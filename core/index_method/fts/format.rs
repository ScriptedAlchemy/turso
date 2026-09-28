//! On-disk format for FTS backing storage: the segment registry.
//!
//! One backing B-tree holds every row of an FTS index. Rows are key-only
//! (the whole `(path, chunk_no, bytes)` record is the index key), so the
//! format is strictly append-only: a row is inserted once and deleted once
//! (by merge), never rewritten in place. Row kinds are
//! distinguished by the `path` prefix:
//!
//! ```text
//! ("fts2/control",              0,      control_blob)     rare index-level facts
//! ("fts2/seg/<uuid>",           0,      descriptor_blob)  segment registry entry
//! ("fts2/chunk/<uuid>/<ord>",   n,      chunk_bytes)      segment file content
//! ("fts2/tomb/<identity>",      0,      [])               document tombstone
//! ```
//!
//! `<uuid>` is Tantivy's 32-char lowercase hex segment id, so two
//! transactions' rows can never collide, and appends by different
//! transactions commute under MVCC. There is no `meta.json` on disk: the
//! visible descriptor rows *are* the meta, and `meta.json` is synthesized
//! per snapshot (see [`synthesize_meta_json`]).
//!
//! `<identity>` is the document's identity as 32 lowercase hex digits. The
//! identity is a u128 that the index assigns when it first indexes the
//! document. The segment stores it as two u64 fast fields (columns that Tantivy
//! reads by document number). A merge copies the identity with the
//! document, so the identity stays the same after every merge. A tombstone
//! (a row that marks a document as deleted) names the document, not the
//! segment that holds it. So a merge that moves the document to a new
//! segment does not break a tombstone that another transaction writes at
//! the same time. A reader hides the document in whichever segment holds
//! it now.
//!
//! The code refuses older stores with a rebuild hint and never converts
//! them. The pre-registry code stored a whole Tantivy directory keyed by
//! file name, without the `fts2/` prefix.

use rustc_hash::FxHashMap as HashMap;
#[cfg(test)]
use rustc_hash::FxHashSet as HashSet;
#[cfg(test)]
use std::collections::BTreeSet;
use tantivy::{
    Index, IndexMeta, IndexSettings, directory::OwnedBytes, index::SegmentId, schema::Schema,
};

use crate::sync::Arc;
use crate::{LimboError, Result};

/// Storage format version stored in the control row.
pub(super) const FTS_STORAGE_FORMAT_VERSION: u32 = 3;

pub(super) const FTS2_CONTROL_PATH: &str = "fts2/control";
pub(super) const FTS2_SEGMENT_PREFIX: &str = "fts2/seg/";
pub(super) const FTS2_CHUNK_PREFIX: &str = "fts2/chunk/";
pub(super) const FTS2_TOMB_PREFIX: &str = "fts2/tomb/";

/// Every row path starts with this; a stored row without it was written
/// by the pre-registry implementation.
pub(super) const FTS2_PATH_PREFIX: &str = "fts2/";

const FTS2_CONTROL_MAGIC: &[u8; 8] = b"TFTSCTL2";
const FTS2_SEGMENT_MAGIC: &[u8; 8] = b"TFTSSEG2";

/// Delete opstamp used for every synthesized delete meta. Tantivy only uses
/// it to name the `.del` file (`<uuid>.<opstamp>.del`); tombstone rows are
/// the real delete state, so one constant value is enough.
pub(super) const FTS2_TOMBSTONE_DELETE_OPSTAMP: u64 = 1;

/// The number the index gives a document when it first indexes it. A merge
/// copies it with the document, so it names the same document in every
/// segment that ever holds it. Segment positions do not: a merge renumbers.
/// Tombstones name documents by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct DocumentIdentity(u128);

impl DocumentIdentity {
    pub fn new(raw: u128) -> Self {
        Self(raw)
    }

    /// The identity `count` documents after this one in the same build.
    pub fn plus(self, count: u32) -> Self {
        Self(self.0.wrapping_add(u128::from(count)))
    }

    /// The number represented by the segment's two identity fast fields.
    pub fn raw(self) -> u128 {
        self.0
    }
}

pub(super) fn segment_registry_path(segment_id: &SegmentId) -> String {
    format!("{FTS2_SEGMENT_PREFIX}{}", segment_id.uuid_string())
}

pub(super) fn segment_chunk_path(segment_id: &SegmentId, file_ord: u32) -> String {
    format!(
        "{FTS2_CHUNK_PREFIX}{}/{file_ord:04}",
        segment_id.uuid_string()
    )
}

pub(super) fn segment_chunk_prefix(segment_id: &SegmentId) -> String {
    format!("{FTS2_CHUNK_PREFIX}{}/", segment_id.uuid_string())
}

pub(super) fn document_tombstone_path(identity: DocumentIdentity) -> String {
    format!("{FTS2_TOMB_PREFIX}{:032x}", identity.raw())
}

pub(super) fn parse_segment_id(hex: &str) -> Result<SegmentId> {
    SegmentId::from_uuid_string(hex)
        .map_err(|_| LimboError::Corrupt(format!("FTS row carries a malformed segment id: {hex}")))
}

pub(super) fn parse_document_identity(hex: &str) -> Result<DocumentIdentity> {
    let malformed = || {
        LimboError::Corrupt(format!(
            "FTS tombstone row carries a malformed document identity: {hex}"
        ))
    };
    if hex.len() != 32 {
        return Err(malformed());
    }
    u128::from_str_radix(hex, 16)
        .map(DocumentIdentity::new)
        .map_err(|_| malformed())
}

fn fts2_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn take<const N: usize>(bytes: &[u8], offset: &mut usize) -> Result<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| LimboError::Corrupt("FTS record offset overflow".into()))?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| LimboError::Corrupt("truncated FTS record".into()))?;
    *offset = end;
    Ok(value.try_into().expect("slice length checked"))
}

pub(super) fn append_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
    let checksum = fts2_checksum(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

fn verify_checksum<'a>(bytes: &'a [u8], what: &str) -> Result<&'a [u8]> {
    if bytes.len() < 8 {
        return Err(LimboError::Corrupt(format!("truncated FTS {what} record")));
    }
    let payload_len = bytes.len() - 8;
    let expected = u64::from_le_bytes(bytes[payload_len..].try_into().expect("length checked"));
    if fts2_checksum(&bytes[..payload_len]) != expected {
        return Err(LimboError::Corrupt(format!(
            "FTS {what} record checksum mismatch"
        )));
    }
    Ok(&bytes[..payload_len])
}

/// Rare index-level facts. Written once when the index is created and
/// never rewritten; its presence marks a registry-format store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FtsControl {
    pub format_version: u32,
    /// Distinguishes drop/recreate lifetimes of the same index name.
    pub index_incarnation: u64,
}

/// A decoded control row. It is either one this code can use, or one that
/// a different format version wrote. The row layout is the same across
/// versions, so the version is always readable. The segment and tombstone
/// rows behind it differ, so the code refuses a store of another version
/// instead of reading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ControlRecord {
    Current(FtsControl),
    OtherVersion(u32),
}

impl FtsControl {
    pub fn new(index_incarnation: u64) -> Self {
        Self {
            format_version: FTS_STORAGE_FORMAT_VERSION,
            index_incarnation,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 4 + 8 + 8);
        bytes.extend_from_slice(FTS2_CONTROL_MAGIC);
        bytes.extend_from_slice(&self.format_version.to_le_bytes());
        bytes.extend_from_slice(&self.index_incarnation.to_le_bytes());
        append_checksum(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<ControlRecord> {
        let payload = verify_checksum(bytes, "control")?;
        let mut offset = 0;
        if take::<8>(payload, &mut offset)? != *FTS2_CONTROL_MAGIC {
            return Err(LimboError::Corrupt(
                "unrecognized FTS control record".into(),
            ));
        }
        let format_version = u32::from_le_bytes(take(payload, &mut offset)?);
        if format_version != FTS_STORAGE_FORMAT_VERSION {
            return Ok(ControlRecord::OtherVersion(format_version));
        }
        let index_incarnation = u64::from_le_bytes(take(payload, &mut offset)?);
        if offset != payload.len() {
            return Err(LimboError::Corrupt(
                "FTS control record has trailing payload bytes".into(),
            ));
        }
        Ok(ControlRecord::Current(Self {
            format_version,
            index_incarnation,
        }))
    }
}

/// One file of an immutable segment, as recorded in its descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SegmentFileEntry {
    /// Tantivy file name, e.g. `<uuid>.term`.
    pub name: String,
    pub size: u64,
    pub num_chunks: u32,
}

/// A segment's registry entry. Inserting this row *is* publishing the
/// segment; deleting it (merge) retires the segment. The segment id lives
/// in the row path, not the blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SegmentDescriptor {
    pub segment_id: SegmentId,
    pub max_doc: u32,
    pub files: Vec<SegmentFileEntry>,
}

impl SegmentDescriptor {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FTS2_SEGMENT_MAGIC);
        bytes.extend_from_slice(&self.max_doc.to_le_bytes());
        let file_count = u32::try_from(self.files.len()).map_err(|_| {
            LimboError::InternalError("FTS segment descriptor has too many files".into())
        })?;
        bytes.extend_from_slice(&file_count.to_le_bytes());
        for file in &self.files {
            let name_len = u32::try_from(file.name.len()).map_err(|_| {
                LimboError::InternalError("FTS segment file name is too long".into())
            })?;
            bytes.extend_from_slice(&name_len.to_le_bytes());
            bytes.extend_from_slice(file.name.as_bytes());
            bytes.extend_from_slice(&file.size.to_le_bytes());
            bytes.extend_from_slice(&file.num_chunks.to_le_bytes());
        }
        Ok(append_checksum(bytes))
    }

    pub fn decode(segment_id: SegmentId, bytes: &[u8]) -> Result<Self> {
        let payload = verify_checksum(bytes, "segment descriptor")?;
        let mut offset = 0;
        if take::<8>(payload, &mut offset)? != *FTS2_SEGMENT_MAGIC {
            return Err(LimboError::Corrupt(
                "unrecognized FTS segment descriptor".into(),
            ));
        }
        let max_doc = u32::from_le_bytes(take(payload, &mut offset)?);
        let file_count = u32::from_le_bytes(take(payload, &mut offset)?) as usize;
        let mut files = Vec::with_capacity(file_count.min(64));
        for _ in 0..file_count {
            let name_len = u32::from_le_bytes(take(payload, &mut offset)?) as usize;
            let name_end = offset
                .checked_add(name_len)
                .ok_or_else(|| LimboError::Corrupt("FTS file name offset overflow".into()))?;
            let name_bytes = payload
                .get(offset..name_end)
                .ok_or_else(|| LimboError::Corrupt("truncated FTS segment file name".into()))?;
            offset = name_end;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| LimboError::Corrupt("FTS segment file name is not UTF-8".into()))?
                .to_string();
            let size = u64::from_le_bytes(take(payload, &mut offset)?);
            let num_chunks = u32::from_le_bytes(take(payload, &mut offset)?);
            if num_chunks == 0 {
                return Err(LimboError::Corrupt(format!(
                    "FTS segment file {name} has zero chunks"
                )));
            }
            files.push(SegmentFileEntry {
                name,
                size,
                num_chunks,
            });
        }
        if offset != payload.len() {
            return Err(LimboError::Corrupt(
                "FTS segment descriptor has trailing payload bytes".into(),
            ));
        }
        Ok(Self {
            segment_id,
            max_doc,
            files,
        })
    }
}

#[derive(Clone)]
pub(super) struct SegmentIdentities {
    hi: tantivy::fastfield::Column<u64>,
    lo: tantivy::fastfield::Column<u64>,
    index: Arc<tantivy::InvertedIndexReader>,
    field: tantivy::schema::Field,
    max_doc: u32,
}

impl std::fmt::Debug for SegmentIdentities {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SegmentIdentities")
            .field("max_doc", &self.max_doc)
            .finish()
    }
}

impl SegmentIdentities {
    pub fn new(
        hi: tantivy::fastfield::Column<u64>,
        lo: tantivy::fastfield::Column<u64>,
        index: Arc<tantivy::InvertedIndexReader>,
        field: tantivy::schema::Field,
        max_doc: u32,
    ) -> Self {
        Self {
            hi,
            lo,
            index,
            field,
            max_doc,
        }
    }

    pub fn identity_of(&self, position: u32) -> Option<DocumentIdentity> {
        if position >= self.max_doc {
            return None;
        }
        self.hi
            .first(position)
            .zip(self.lo.first(position))
            .map(|(hi, lo)| DocumentIdentity::new((u128::from(hi) << 64) | u128::from(lo)))
    }

    pub fn position_of(&self, identity: DocumentIdentity) -> Result<Option<u32>> {
        use tantivy::DocSet;
        let term = tantivy::Term::from_field_u64(self.field, identity.raw() as u64);
        let Some(mut postings) = self
            .index
            .read_postings(&term, tantivy::schema::IndexRecordOption::Basic)
            .map_err(|error| LimboError::Corrupt(format!("FTS identity postings: {error}")))?
        else {
            return Ok(None);
        };
        while postings.doc() != tantivy::TERMINATED {
            let position = postings.doc();
            if self.identity_of(position) == Some(identity) {
                return Ok(Some(position));
            }
            postings.advance();
        }
        Ok(None)
    }

    pub fn tombstoned_positions(
        &self,
        tombstones: impl Iterator<Item = DocumentIdentity>,
        directory: &std::path::Path,
    ) -> Result<DeletedDocs> {
        let mut deleted = DeletedDocs::new(self.max_doc, directory);
        for identity in tombstones {
            if let Some(position) = self.position_of(identity)? {
                deleted.insert(position)?;
            }
        }
        Ok(deleted)
    }
}

/// The mapped bytes of one immutable segment: each file's contents by
/// file name, plus its document identities. Connections share it, keyed by
/// segment id. A segment never changes, so the cache needs no snapshot
/// identity.
#[derive(Debug)]
pub(super) struct SegmentData {
    pub files: HashMap<String, OwnedBytes>,
    pub identities: SegmentIdentities,
    pub total_bytes: usize,
}

impl SegmentData {
    pub fn new(files: HashMap<String, OwnedBytes>, identities: SegmentIdentities) -> Self {
        let total_bytes = files.values().map(|data| data.len()).sum::<usize>();
        Self {
            files,
            identities,
            total_bytes,
        }
    }
}

#[derive(Debug)]
pub(super) struct DeletedDocs {
    max_doc: u32,
    count: usize,
    map: Option<memmap2::MmapMut>,
    directory: std::path::PathBuf,
}

impl DeletedDocs {
    pub fn new(max_doc: u32, directory: &std::path::Path) -> Self {
        Self {
            max_doc,
            count: 0,
            map: None,
            directory: directory.to_path_buf(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        let mut copy = Self::new(self.max_doc, &self.directory);
        if let Some(mapping) = &self.map {
            let target = copy.allocate()?;
            target.copy_from_slice(mapping);
        }
        copy.count = self.count;
        Ok(copy)
    }

    fn allocate(&mut self) -> Result<&mut memmap2::MmapMut> {
        if self.map.is_none() {
            let file = tempfile::tempfile_in(&self.directory)
                .map_err(|error| crate::error::io_error(error, "create FTS deletion mask"))?;
            let bytes = u64::from(self.max_doc).div_ceil(64) * 8;
            file.set_len(bytes)
                .map_err(|error| crate::error::io_error(error, "size FTS deletion mask"))?;
            // SAFETY: this unnamed file has one private writable mapping and
            // its file handle is closed without exposing another writer.
            let map = unsafe { memmap2::MmapMut::map_mut(&file) }
                .map_err(|error| crate::error::io_error(error, "map FTS deletion mask"))?;
            self.map = Some(map);
        }
        self.map
            .as_mut()
            .ok_or_else(|| LimboError::InternalError("FTS deletion mask missing".into()))
    }

    pub fn insert(&mut self, position: u32) -> Result<bool> {
        if position >= self.max_doc {
            return Err(LimboError::Corrupt("FTS deletion outside segment".into()));
        }
        let byte = position as usize / 8;
        let bit = 1u8 << (position % 8);
        let map = self.allocate()?;
        if map[byte] & bit != 0 {
            return Ok(false);
        }
        map[byte] |= bit;
        self.count += 1;
        Ok(true)
    }

    pub fn contains(&self, position: &u32) -> bool {
        *position < self.max_doc
            && self
                .map
                .as_ref()
                .is_some_and(|map| map[*position as usize / 8] & (1 << (*position % 8)) != 0)
    }

    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.max_doc).filter(|position| self.contains(position))
    }

    pub fn alive_file(&self) -> Result<OwnedBytes> {
        let mut file = super::directory::StagedFile::new(&self.directory)?;
        let mut checksum = crc32fast::Hasher::new();
        let header = self.max_doc.to_le_bytes();
        checksum.update(&header);
        file.append(0, &header)?;
        let words = u64::from(self.max_doc).div_ceil(64);
        for word in 0..words {
            let offset = word as usize * 8;
            let deleted = self.map.as_ref().map_or(0, |map| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&map[offset..offset + 8]);
                u64::from_le_bytes(bytes)
            });
            let mut alive = !deleted;
            if word + 1 == words && self.max_doc % 64 != 0 {
                alive &= (1u64 << (self.max_doc % 64)) - 1;
            }
            let bytes = alive.to_le_bytes();
            checksum.update(&bytes);
            file.append(i64::from(file.chunks), &bytes)?;
        }
        let footer =
            serde_json::json!({ "version": tantivy::version(), "crc": checksum.finalize() });
        let payload = serde_json::to_vec(&footer)
            .map_err(|error| LimboError::InternalError(format!("FTS footer synthesis: {error}")))?;
        let length = u32::try_from(payload.len()).map_err(|_| LimboError::TooBig)?;
        for bytes in [
            &payload[..],
            &length.to_le_bytes(),
            &FOOTER_MAGIC_NUMBER.to_le_bytes(),
        ] {
            file.append(i64::from(file.chunks), bytes)?;
        }
        file.finish()
    }
}

/// One segment as seen by a cursor's snapshot: immutable bytes plus the
/// tombstoned doc ids visible at (or created by) this transaction.
#[derive(Debug)]
pub(super) struct LoadedSegment {
    pub descriptor: SegmentDescriptor,
    pub data: Arc<SegmentData>,
    /// Doc ids whose postings are dead at this snapshot. Ordered so cache
    /// identity comparisons and bitset builds are deterministic.
    pub deleted: DeletedDocs,
}

impl LoadedSegment {
    pub fn new(
        descriptor: SegmentDescriptor,
        data: Arc<SegmentData>,
        deleted: DeletedDocs,
    ) -> Self {
        Self {
            descriptor,
            data,
            deleted,
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self::new(
            self.descriptor.clone(),
            Arc::clone(&self.data),
            self.deleted.try_clone()?,
        ))
    }

    pub fn id(&self) -> SegmentId {
        self.descriptor.segment_id
    }

    pub fn live_docs(&self) -> u64 {
        u64::from(self.descriptor.max_doc).saturating_sub(self.deleted.len() as u64)
    }

    /// The identities of the documents that are deleted at this snapshot.
    pub fn tombstoned_identities(&self) -> impl Iterator<Item = DocumentIdentity> + '_ {
        self.deleted
            .iter()
            .filter_map(|position| self.data.identities.identity_of(position))
    }

    pub fn meta_spec(&self) -> SegmentMetaSpec {
        SegmentMetaSpec::new(
            self.id(),
            self.descriptor.max_doc,
            self.deleted.len() as u32,
        )
    }
}

/// What `synthesize_meta_json` records about one segment.
#[derive(Debug, Clone, Copy)]
pub(super) struct SegmentMetaSpec {
    segment_id: SegmentId,
    max_doc: u32,
    num_deleted: u32,
}

impl SegmentMetaSpec {
    pub fn new(segment_id: SegmentId, max_doc: u32, num_deleted: u32) -> Self {
        Self {
            segment_id,
            max_doc,
            num_deleted,
        }
    }

    pub fn id(&self) -> SegmentId {
        self.segment_id
    }

    pub fn max_doc(&self) -> u32 {
        self.max_doc
    }

    pub fn num_deleted(&self) -> u32 {
        self.num_deleted
    }

    pub fn has_deleted_documents(&self) -> bool {
        self.num_deleted > 0
    }
}

/// Serialize an alive bitset in Tantivy's `.del` format:
/// `[u32 max_value LE][ceil(max_value/64) x u64 words LE]`, bit set = alive.
#[cfg(test)]
pub(super) fn alive_bitset_bytes(max_doc: u32, deleted: &BTreeSet<u32>) -> Vec<u8> {
    let words = (max_doc as usize).div_ceil(64);
    let mut bytes = Vec::with_capacity(4 + words * 8);
    bytes.extend_from_slice(&max_doc.to_le_bytes());
    let mut word_buf = vec![u64::MAX; words];
    // Clear bits at or beyond max_doc in the last word so num_alive_docs is
    // exact; every earlier word is fully below max_doc.
    let tail_bits = max_doc % 64;
    if tail_bits != 0 {
        if let Some(last) = word_buf.last_mut() {
            *last = (1u64 << tail_bits) - 1;
        }
    }
    for doc in deleted {
        if *doc < max_doc {
            word_buf[(*doc / 64) as usize] &= !(1u64 << (*doc % 64));
        }
    }
    for word in word_buf {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
pub(super) fn alive_bitset(
    max_doc: u32,
    deleted: &BTreeSet<u32>,
) -> tantivy::fastfield::AliveBitSet {
    tantivy::fastfield::AliveBitSet::open(tantivy::directory::OwnedBytes::new(alive_bitset_bytes(
        max_doc, deleted,
    )))
}

/// Build the `meta.json` bytes for a snapshot's visible segment set.
///
/// `meta.json` survives as an interface, not a file: Tantivy's `Index::open`
/// wants to deserialize an `IndexMeta`, so we serialize one built from the
/// registry rows. Segments with tombstones get a delete meta pointing at a
/// synthesized `.del` file the snapshot directory serves from the same
/// tombstone set (see `directory`), which makes every query path — including
/// `TopDocs` — honor tombstones at the `SegmentReader` level.
///
/// `scratch` is any index with no meaning of its own; it only mints
/// `SegmentMeta` values (their serialized form is independent of the index
/// they were minted from). `IndexSettings` must be identical across every
/// transaction or a mixed searcher could not read old segments; we pin the
/// default everywhere.
pub(super) fn synthesize_meta_json(
    scratch: &Index,
    schema: &Schema,
    segments: &[SegmentMetaSpec],
) -> Result<Vec<u8>> {
    let metas = segments
        .iter()
        .map(|segment| {
            let meta = scratch.new_segment_meta(segment.id(), segment.max_doc());
            if segment.has_deleted_documents() {
                meta.with_delete_meta(segment.num_deleted(), FTS2_TOMBSTONE_DELETE_OPSTAMP)
            } else {
                meta
            }
        })
        .collect();
    let meta = IndexMeta {
        index_settings: IndexSettings::default(),
        segments: metas,
        schema: schema.clone(),
        opstamp: 0,
        payload: None,
    };
    serde_json::to_vec(&meta)
        .map_err(|e| LimboError::InternalError(format!("FTS meta synthesis failed: {e}")))
}

/// The synthesized `.del` file name for a segment
/// (`<uuid>.<FTS2_TOMBSTONE_DELETE_OPSTAMP>.del`).
pub(super) fn tombstone_del_file_name(segment_id: &SegmentId) -> String {
    format!(
        "{}.{}.del",
        segment_id.uuid_string(),
        FTS2_TOMBSTONE_DELETE_OPSTAMP
    )
}

/// Append Tantivy's per-file footer (`<json {version, crc}> <u32 len>
/// <u32 magic>`) to synthesized file bytes.
///
/// Every file served through a Tantivy index is read via `ManagedDirectory`,
/// which validates and strips this footer. Segment files captured from a
/// build already carry one; only files we synthesize ourselves (the `.del`
/// bytes derived from tombstone rows) need it added.
const FOOTER_MAGIC_NUMBER: u32 = 1337;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_identities(values: Vec<DocumentIdentity>) -> SegmentIdentities {
        let mut schema = Schema::builder();
        let hi = schema.add_u64_field("hi", tantivy::schema::FAST);
        let lo = schema.add_u64_field("lo", tantivy::schema::FAST | tantivy::schema::INDEXED);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_with_num_threads(1, 15_000_000).unwrap();
        for identity in &values {
            let mut document = tantivy::TantivyDocument::default();
            document.add_u64(hi, (identity.raw() >> 64) as u64);
            document.add_u64(lo, identity.raw() as u64);
            writer.add_document(document).unwrap();
        }
        writer.commit().unwrap();
        let reader: tantivy::IndexReader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        SegmentIdentities::new(
            segment.fast_fields().u64("hi").unwrap(),
            segment.fast_fields().u64("lo").unwrap(),
            segment.inverted_index(lo).unwrap(),
            lo,
            values.len() as u32,
        )
    }

    fn positions(
        identities: &SegmentIdentities,
        tombstones: &HashSet<DocumentIdentity>,
    ) -> BTreeSet<u32> {
        let directory = tempfile::tempdir().unwrap();
        identities
            .tombstoned_positions(tombstones.iter().copied(), directory.path())
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn mapped_deletions_keep_snapshot_copies_independent_and_encode_tail_bits() {
        let directory = tempfile::tempdir().unwrap();
        let mut deleted = DeletedDocs::new(130, directory.path());
        assert!(!deleted.insert(131).is_ok());
        for position in [0, 63, 64, 129] {
            assert!(deleted.insert(position).unwrap());
        }
        assert!(!deleted.insert(64).unwrap());
        let copy = deleted.try_clone().unwrap();
        deleted.insert(1).unwrap();
        assert!(!copy.contains(&1));
        assert_eq!(copy.iter().collect::<Vec<_>>(), [0, 63, 64, 129]);
        let bytes = copy.alive_file().unwrap();
        let bitset = tantivy::fastfield::AliveBitSet::open(bytes.slice(0..28));
        assert_eq!(bitset.num_alive_docs(), 126);
        for position in 0..130 {
            assert_eq!(bitset.is_alive(position), !copy.contains(&position));
        }
        drop(bitset);
        drop(bytes);
        drop(copy);
        drop(deleted);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn control_record_round_trips_and_detects_corruption() {
        let control = FtsControl::new(0xdead_beef);
        let bytes = control.encode();
        assert_eq!(
            FtsControl::decode(&bytes).unwrap(),
            ControlRecord::Current(control)
        );

        assert!(FtsControl::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut corrupted = bytes;
        corrupted[9] ^= 0xff;
        assert!(FtsControl::decode(&corrupted).is_err());
    }

    #[test]
    fn control_record_of_another_format_version_reports_its_version() {
        let mut older = FtsControl::new(7);
        older.format_version = 1;
        assert_eq!(
            FtsControl::decode(&older.encode()).unwrap(),
            ControlRecord::OtherVersion(1)
        );
    }

    #[test]
    fn document_tombstone_path_round_trips_the_identity() {
        for raw in [0u128, 1, 0x0123456789abcdef_fedcba9876543210, u128::MAX] {
            let identity = DocumentIdentity::new(raw);
            let path = document_tombstone_path(identity);
            let hex = path.strip_prefix(FTS2_TOMB_PREFIX).unwrap();
            assert_eq!(hex.len(), 32);
            assert_eq!(parse_document_identity(hex).unwrap(), identity);
        }
        assert_eq!(
            document_tombstone_path(DocumentIdentity::new(0x0123456789abcdef_fedcba9876543210)),
            "fts2/tomb/0123456789abcdeffedcba9876543210"
        );
        assert!(parse_document_identity("abc").is_err());
        assert!(parse_document_identity("0123456789abcdef").is_err());
        assert!(parse_document_identity("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz").is_err());
    }

    #[test]
    fn document_identity_addition_carries_into_high_bits() {
        let identity = DocumentIdentity::new(0x0123456789abcdef_fffffffffffffffe);
        assert_eq!(identity.plus(1).raw(), 0x0123456789abcdef_ffffffffffffffff);
        assert_eq!(identity.plus(2).raw(), 0x0123456789abcdf0_0000000000000000);
        assert_eq!(DocumentIdentity::new(u128::MAX).plus(1).raw(), 0);
    }

    #[test]
    fn segment_identities_map_both_ways_and_find_tombstoned_positions() {
        let identity = DocumentIdentity::new;
        let identities = test_identities(vec![
            identity(500),
            identity(20),
            identity(9_000),
            identity(3),
        ]);
        assert_eq!(identities.identity_of(2), Some(identity(9_000)));
        assert_eq!(identities.identity_of(4), None);
        assert_eq!(identities.position_of(identity(3)).unwrap(), Some(3));
        assert_eq!(identities.position_of(identity(500)).unwrap(), Some(0));
        assert_eq!(identities.position_of(identity(42)).unwrap(), None);

        let few = HashSet::from_iter([identity(20), identity(42)]);
        assert_eq!(positions(&identities, &few), BTreeSet::from([1]));
        let many = HashSet::from_iter([3, 20, 500, 9_000, 1, 2].map(identity));
        assert_eq!(positions(&identities, &many), BTreeSet::from([0, 1, 2, 3]));
        assert!(positions(&identities, &HashSet::default()).is_empty());
    }

    #[test]
    fn identity_lookup_distinguishes_high_bits() {
        let low = DocumentIdentity::new(7);
        let high = DocumentIdentity::new((1 << 100) | 7);
        let identities = test_identities(vec![high, low]);
        assert_eq!(identities.position_of(high).unwrap(), Some(0));
        assert_eq!(identities.position_of(low).unwrap(), Some(1));
        for tombstones in [
            HashSet::from_iter([high]),
            HashSet::from_iter([high, DocumentIdentity::new(99)]),
        ] {
            assert_eq!(positions(&identities, &tombstones), BTreeSet::from([0]));
        }
    }

    #[test]
    fn segment_descriptor_round_trips() {
        let segment_id = SegmentId::generate_random();
        let descriptor = SegmentDescriptor {
            segment_id,
            max_doc: 42,
            files: vec![
                SegmentFileEntry {
                    name: format!("{}.term", segment_id.uuid_string()),
                    size: 1234,
                    num_chunks: 1,
                },
                SegmentFileEntry {
                    name: format!("{}.store", segment_id.uuid_string()),
                    size: 5 * 1024 * 1024,
                    num_chunks: 10,
                },
            ],
        };
        let bytes = descriptor.encode().unwrap();
        assert_eq!(
            SegmentDescriptor::decode(segment_id, &bytes).unwrap(),
            descriptor
        );

        let mut corrupted = bytes;
        *corrupted.last_mut().unwrap() ^= 0x01;
        assert!(SegmentDescriptor::decode(segment_id, &corrupted).is_err());
    }

    #[test]
    fn alive_bitset_marks_exactly_the_tombstoned_docs() {
        let deleted = BTreeSet::from([0u32, 3, 64, 129]);
        let bitset = alive_bitset(130, &deleted);
        assert_eq!(bitset.num_alive_docs(), 130 - deleted.len());
        for doc in 0..130 {
            assert_eq!(
                bitset.is_deleted(doc),
                deleted.contains(&doc),
                "doc {doc} has the wrong liveness"
            );
        }
    }

    #[test]
    fn alive_bitset_with_no_tombstones_keeps_every_doc() {
        let bitset = alive_bitset(65, &BTreeSet::new());
        assert_eq!(bitset.num_alive_docs(), 65);
        assert!(!bitset.is_deleted(64));
    }
}

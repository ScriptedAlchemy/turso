use super::*;
use crate::{
    index_method::{
        IndexMethodAttachment, IndexMethodConfiguration, IndexMethodCostContext,
        IndexMethodCostEstimate,
    },
    schema::IndexColumn,
};
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::num::NonZeroU32;
use turso_parser::ast::{Expr, Literal, UnaryOperator, Variable};

#[test]
fn unicode_tokenizer_preserves_long_tokens_and_folds_case_and_accents() {
    for mvcc in [false, true] {
        let db = crate::Database::open(
            Arc::new(crate::MemoryIO::new()),
            ":memory:",
            crate::OpenOptions::new(Arc::new(crate::SqliteDialect))
                .db_opts(crate::DatabaseOpts::default().with_index_method(true)),
        )
        .unwrap();
        let conn = db.connect().unwrap();
        if mvcc {
            conn.execute("PRAGMA journal_mode = 'mvcc'").unwrap();
        }
        conn.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
            .unwrap();
        conn.execute(
            "CREATE INDEX docs_fts ON docs USING fts(title, body) WITH (tokenizer = 'unicode')",
        )
        .unwrap();
        let token = "LongIdentifier".repeat(8);
        conn.execute(format!("INSERT INTO docs VALUES (1, 'Résumé', 'CAFÉ quick fox {token}'), (2, 'cafe', 'unrelated')"))
            .unwrap();
        for query in [
            token.to_lowercase(),
            format!("{}*", token[..60].to_lowercase()),
            "body:cafe".to_string(),
            "title:resume".to_string(),
            "body:\"QUICK FOX\"".to_string(),
        ] {
            let rows = conn
                .prepare(format!(
                    "SELECT id FROM docs WHERE fts_match(title, body, '{query}') ORDER BY id"
                ))
                .unwrap()
                .run_collect_rows()
                .unwrap();
            assert_eq!(rows, vec![vec![Value::from_i64(1)]], "{query}, mvcc={mvcc}");
        }
        conn.execute("BEGIN").unwrap();
        conn.execute("UPDATE docs SET body = 'changed' WHERE id = 1")
            .unwrap();
        conn.execute("ROLLBACK").unwrap();
        let rows = conn
            .prepare("SELECT id FROM docs WHERE fts_match(title, body, 'body:cafe')")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert_eq!(rows, vec![vec![Value::from_i64(1)]]);
        conn.execute("DELETE FROM docs WHERE id = 1").unwrap();
        let rows = conn
            .prepare("SELECT id FROM docs WHERE fts_match(title, body, 'body:cafe')")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert!(rows.is_empty());
    }
}

#[test]
fn fts_score_survives_joins_and_negation() {
    let db = crate::Database::open(
        Arc::new(crate::MemoryIO::new()),
        ":memory:",
        crate::OpenOptions::new(Arc::new(crate::SqliteDialect))
            .db_opts(crate::DatabaseOpts::default().with_index_method(true)),
    )
    .unwrap();
    let conn = db.connect().unwrap();
    conn.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT)")
        .unwrap();
    conn.execute("CREATE INDEX docs_fts ON docs USING fts(body)")
        .unwrap();
    conn.execute("INSERT INTO docs VALUES(1, 'relevant text')")
        .unwrap();
    for sql in [
        "SELECT fts_score(body, 'relevant') AS rank FROM docs WHERE fts_match(body, 'relevant') ORDER BY rank DESC",
        "SELECT fts_score(d.body, 'relevant') AS rank FROM docs d JOIN docs other ON other.id = d.id WHERE fts_match(d.body, 'relevant') ORDER BY rank DESC",
        "SELECT -fts_score(body, 'relevant') AS rank FROM docs WHERE fts_match(body, 'relevant') ORDER BY rank",
        "SELECT -fts_score(d.body, 'relevant') AS rank FROM docs d JOIN docs other ON other.id = d.id WHERE fts_match(d.body, 'relevant') ORDER BY rank",
        "SELECT -fts_score(body, 'relevant') AS rank FROM docs WHERE fts_match(body, 'relevant') AND fts_score(body, 'relevant') > 0 ORDER BY rank",
        "SELECT -fts_score(d.body, 'relevant') AS rank FROM docs d JOIN docs other ON other.id = d.id WHERE fts_match(d.body, 'relevant') AND fts_score(d.body, 'relevant') > 0 ORDER BY rank",
    ] {
        let scores = conn
            .prepare(sql)
            .unwrap_or_else(|error| panic!("{sql}: {error:?}"))
            .run_collect_rows()
            .unwrap();
        assert_eq!(scores.len(), 1);
        let expected_negative = sql.starts_with("SELECT -");
        let score = scores[0][0].as_float();
        assert!(score.is_finite() && score != 0.0, "{sql}: {scores:?}");
        assert_eq!(score < 0.0, expected_negative, "{sql}: {scores:?}");
    }
}

#[test]
fn field_weights_reject_non_finite_and_non_positive_values() {
    let mut accepted = Vec::new();
    for weight in [
        "NaN",
        "nan",
        "+NaN",
        "-NaN",
        "inf",
        "+inf",
        "-inf",
        "Infinity",
        "+INFINITY",
        "-infinity",
        "1e39",
        "-1e39",
        "0",
        "-0",
        "-1",
        "1e-46",
    ] {
        let result = FtsIndexAttachment::new(IndexMethodConfiguration {
            table_name: "w".to_string(),
            index_name: "wx".to_string(),
            columns: crate::alloc::vec![IndexColumn::new("title", 0), IndexColumn::new("body", 1)],
            parameters: FxHashMap::from_iter([(
                "weights".to_string(),
                Value::from_text(format!("title={weight},body=1")),
            )]),
        });
        match result {
            Ok(_) => accepted.push(weight),
            Err(error) => assert!(
                matches!(error, LimboError::ParseError(_)),
                "{weight}: {error}"
            ),
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted invalid weights: {accepted:?}"
    );
}

#[test]
fn field_weights_accept_finite_positive_boundaries() {
    let columns = [IndexColumn::new("title", 0), IndexColumn::new("body", 1)];
    for (input, expected) in [
        ("1e-45", f32::from_bits(1)),
        ("1.17549435e-38", f32::MIN_POSITIVE),
        ("0.5", 0.5),
        ("+1", 1.0),
        ("3.4028235e38", f32::MAX),
    ] {
        let weights = parse_field_weights(&format!("title={input},body=2"), &columns).unwrap();
        assert_eq!(weights["title"], expected, "{input}");
        assert_eq!(weights["body"], 2.0);
    }
}

fn test_attachment() -> FtsIndexAttachment {
    FtsIndexAttachment::new(IndexMethodConfiguration {
        table_name: "docs".to_string(),
        index_name: "docs_fts".to_string(),
        columns: crate::alloc::vec![IndexColumn::new("title", 1), IndexColumn::new("body", 2)],
        parameters: FxHashMap::<String, Value>::default(),
    })
    .unwrap()
}

#[test]
fn indexed_text_is_not_duplicated_in_tantivy_document_store() {
    let attachment = test_attachment();
    for (_, field) in attachment.text_fields {
        assert!(
            !attachment.schema.get_field_entry(field).is_stored(),
            "FTS projections come from the base table, so storing indexed text duplicates data"
        );
    }
}

fn estimate_cost(pattern_idx: i64, limit: Option<Expr>) -> IndexMethodCostEstimate {
    let attachment = FtsIndexAttachment::new(IndexMethodConfiguration {
        table_name: "docs".to_string(),
        index_name: "docs_fts".to_string(),
        columns: crate::alloc::vec![IndexColumn::new("body", 1)],
        parameters: FxHashMap::<String, Value>::default(),
    })
    .unwrap();
    let cursor = attachment.init().unwrap();
    let mut arguments = vec![Expr::Literal(Literal::String("'database'".to_string()))];
    arguments.extend(limit);

    cursor
        .estimate_cost(&IndexMethodCostContext {
            pattern_idx: pattern_idx as usize,
            base_table_rows: 100_000.0,
            arguments: &arguments,
        })
        .unwrap()
}

#[test]
fn fts_cost_estimate_applies_literal_limit_to_output_rows() {
    let unlimited = estimate_cost(FTS_PATTERN_MATCH, None);
    assert_eq!(unlimited.estimated_rows, 1_000);

    let limited = estimate_cost(
        FTS_PATTERN_MATCH_LIMIT,
        Some(Expr::Literal(Literal::Numeric("10".to_string()))),
    );
    assert_eq!(limited.estimated_rows, 10);
    assert!(limited.estimated_cost < unlimited.estimated_cost);
    let ranked = estimate_cost(
        FTS_PATTERN_SCORE,
        Some(Expr::Literal(Literal::Numeric("10".to_string()))),
    );
    assert_eq!(ranked.estimated_rows, 10);
    assert!(
        ranked.estimated_cost > limited.estimated_cost,
        "global score ordering must account for scoring all matches"
    );

    let zero = estimate_cost(
        FTS_PATTERN_MATCH_LIMIT,
        Some(Expr::Literal(Literal::Numeric("0".to_string()))),
    );
    assert_eq!(zero.estimated_rows, 0);

    let negative = estimate_cost(
        FTS_PATTERN_MATCH_LIMIT,
        Some(Expr::Unary(
            UnaryOperator::Negative,
            Box::new(Expr::Literal(Literal::Numeric("1".to_string()))),
        )),
    );
    assert_eq!(negative.estimated_rows, unlimited.estimated_rows);

    let dynamic = estimate_cost(
        FTS_PATTERN_MATCH_LIMIT,
        Some(Expr::Variable(Variable::indexed(NonZeroU32::MIN))),
    );
    assert_eq!(dynamic.estimated_rows, unlimited.estimated_rows);
}

#[test]
fn staged_chunks_reject_missing_or_duplicate_numbers_and_share_mapped_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let mut file = StagedFile::new(directory.path()).unwrap();
    assert!(matches!(file.append(-1, &[9]), Err(LimboError::Corrupt(_))));
    assert!(matches!(file.append(1, &[9]), Err(LimboError::Corrupt(_))));
    file.append(0, &[1, 2, 3]).unwrap();
    assert!(matches!(file.append(0, &[9]), Err(LimboError::Corrupt(_))));
    file.append(1, &[4, 5]).unwrap();
    let mapped = file.finish().unwrap();
    assert_eq!(&*mapped, &[1, 2, 3, 4, 5]);
    let tail = mapped.slice(3..5);
    assert_eq!(tail.as_ptr(), mapped[3..].as_ptr());
    drop(mapped);
    assert_eq!(&*tail, &[4, 5]);
    drop(tail);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn query_limit_is_exact_and_bounded_by_live_documents() {
    assert_eq!(bounded_query_limit(None, 1_500_000), 1_500_000);
    assert_eq!(bounded_query_limit(Some(-1), 1_500_000), 1_500_000);
    assert_eq!(bounded_query_limit(Some(i64::MAX), 37), 37);
    assert_eq!(bounded_query_limit(Some(12), 37), 12);
    assert_eq!(bounded_query_limit(Some(0), 37), 0);
    assert_eq!(bounded_query_limit(None, 0), 0);
}

/// Build one segment through the private write path and reopen it through a
/// synthesized snapshot view — the round trip every write and read takes,
/// without a database underneath.
fn build_and_load_segment(
    attachment: &FtsIndexAttachment,
    docs: &[(i64, &str)],
) -> (LoadedSegment, Vec<PendingRow>) {
    let mut cursor_docs = Vec::new();
    for (rowid, text) in docs {
        let mut doc = TantivyDocument::default();
        doc.add_i64(attachment.rowid_field, *rowid);
        doc.add_text(attachment.text_fields[0].1, *text);
        cursor_docs.push(BufferedDoc { rowid: *rowid, doc });
    }
    let mut cursor = FtsCursor::new(attachment);
    cursor.doc_buffer = cursor_docs;
    let (segment, rows) = cursor.build_segment().unwrap();
    (segment.expect("non-empty buffer builds a segment"), rows)
}

#[test]
fn segment_build_round_trips_through_synthesized_snapshot() {
    let attachment = test_attachment();
    let (segment, rows) = build_and_load_segment(
        &attachment,
        &[(1, "hello turso"), (2, "hello world"), (3, "goodbye")],
    );
    assert_eq!(segment.descriptor.max_doc, 3);
    // Rows: chunks for every captured file plus one descriptor row.
    let descriptor_rows = rows
        .iter()
        .filter(|row| row.path.starts_with(FTS2_SEGMENT_PREFIX))
        .count();
    assert_eq!(descriptor_rows, 1);
    assert!(
        rows.iter()
            .all(|row| row.path.starts_with(FTS2_PATH_PREFIX))
    );

    // Reopen through a snapshot view and query it.
    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![segment];
    cursor.snapshot_loaded = true;
    cursor.ensure_searcher().unwrap();
    let searcher = cursor.searcher.as_ref().unwrap();
    assert_eq!(searcher.num_docs(), 3);

    let parser = cursor.cached_parser.as_ref().unwrap();
    let (query, errors) = parser.parse_query_lenient("hello");
    assert!(errors.is_empty());
    let hits = searcher.search(&query, &tantivy::collector::Count).unwrap();
    assert_eq!(hits, 2);
}

#[test]
fn merged_segment_files_can_be_rekeyed_to_a_minted_id() {
    let attachment = test_attachment();
    let (segment, _) = build_and_load_segment(
        &attachment,
        &[(1, "hello turso"), (2, "hello world"), (3, "goodbye")],
    );
    let minted = SegmentId::from_uuid_string("0123456789abcdef0123456789abcdef").unwrap();
    assert_ne!(segment.id(), minted);

    let files: HashMap<PathBuf, OwnedBytes> = segment
        .data
        .files
        .iter()
        .map(|(name, bytes)| (PathBuf::from(name), bytes.clone()))
        .collect();
    let renamed = rename_segment_files(files.clone(), &segment.id(), &minted).unwrap();
    assert_eq!(renamed.len(), files.len());
    assert!(
        renamed
            .keys()
            .all(|path| path.to_str().unwrap().starts_with(&minted.uuid_string()))
    );

    // The renamed files open and answer queries under the new id: the
    // bytes never embed the segment id.
    let (rekeyed, _) = segment_rows_from_files(
        minted,
        segment.descriptor.max_doc,
        renamed,
        segment.data.identities.clone(),
        std::path::Path::new("."),
    )
    .unwrap();
    let rekeyed = rekeyed.expect("non-empty segment");
    assert_eq!(rekeyed.id(), minted);
    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![rekeyed];
    cursor.snapshot_loaded = true;
    cursor.ensure_searcher().unwrap();
    let searcher = cursor.searcher.as_ref().unwrap();
    let (query, _) = cursor
        .cached_parser
        .as_ref()
        .unwrap()
        .parse_query_lenient("hello");
    assert_eq!(
        searcher.search(&query, &tantivy::collector::Count).unwrap(),
        2
    );

    // A file that is not named after the source segment is a bug, not
    // something to rename silently.
    let mut stray = files;
    stray.insert(PathBuf::from("meta.json"), OwnedBytes::empty());
    assert!(matches!(
        rename_segment_files(stray, &segment.id(), &minted),
        Err(LimboError::InternalError(_))
    ));
}

#[test]
fn tombstoned_docs_are_invisible_at_the_reader_level() {
    let attachment = test_attachment();
    let (mut segment, _) = build_and_load_segment(
        &attachment,
        &[(1, "hello turso"), (2, "hello world"), (3, "goodbye")],
    );

    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![segment.try_clone().unwrap()];
    cursor.snapshot_loaded = true;
    let postings = cursor.live_postings_for_rowid(2).unwrap();
    assert_eq!(postings.len(), 1);
    let (segment_id, doc_id) = postings[0];
    assert_eq!(segment_id, segment.id());

    // Tombstone rowid 2 and rebuild the view: the posting must disappear
    // from every query path, including counts.
    segment.deleted.insert(doc_id).unwrap();
    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![segment];
    cursor.snapshot_loaded = true;
    cursor.ensure_searcher().unwrap();
    let searcher = cursor.searcher.as_ref().unwrap();
    assert_eq!(searcher.num_docs(), 2);
    let parser = cursor.cached_parser.as_ref().unwrap();
    let (query, _) = parser.parse_query_lenient("hello");
    let hits = searcher.search(&query, &tantivy::collector::Count).unwrap();
    assert_eq!(hits, 1, "the tombstoned posting must not match");
    assert!(cursor.live_postings_for_rowid(2).unwrap().is_empty());
}

#[test]
fn rowid_lookup_uses_current_deletes_with_reordered_segments() {
    let attachment = test_attachment();
    let (first, _) = build_and_load_segment(&attachment, &[(7, "first"), (9, "other")]);
    let (second, _) = build_and_load_segment(&attachment, &[(11, "other"), (7, "second")]);
    let first_id = first.id();
    let second_id = second.id();
    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![first, second];
    cursor.snapshot_loaded = true;
    cursor.ensure_searcher().unwrap();
    cursor.segments.reverse();
    let hits: HashSet<_> = cursor
        .live_postings_for_rowid(7)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(hits, HashSet::from_iter([(first_id, 0), (second_id, 1)]));
    cursor.segments[0].deleted.insert(1).unwrap();
    assert_eq!(
        cursor.live_postings_for_rowid(7).unwrap(),
        vec![(first_id, 0)]
    );
    assert!(cursor.live_postings_for_rowid(99).unwrap().is_empty());
}

fn identities_of(segment: &LoadedSegment) -> Vec<DocumentIdentity> {
    (0..segment.descriptor.max_doc)
        .map(|position| segment.data.identities.identity_of(position).unwrap())
        .collect()
}

#[test]
fn segment_load_reads_the_identities_the_build_wrote() {
    let attachment = test_attachment();
    let (segment, _) = build_and_load_segment(
        &attachment,
        &[(1, "hello turso"), (2, "hello world"), (3, "goodbye")],
    );
    let written = identities_of(&segment);
    assert_eq!(written.len(), 3);
    assert!(
        written
            .iter()
            .all(|identity| identity.raw() > u128::from(u64::MAX))
    );
    assert!(
        written.windows(2).all(|pair| pair[0] != pair[1]),
        "every document gets its own identity"
    );

    // A segment loaded from storage reads its identities from the fast
    // field. A merged segment and every cache miss do the same.
    let files: HashMap<PathBuf, OwnedBytes> = segment
        .data
        .files
        .iter()
        .map(|(name, bytes)| (PathBuf::from(name), bytes.clone()))
        .collect();
    let scratch = attachment.shared.scratch_index(&attachment.schema).unwrap();
    let read_back = read_segment_identities(
        &scratch,
        &attachment.schema,
        segment.id(),
        segment.descriptor.max_doc,
        files,
    )
    .unwrap();
    for position in 0..segment.descriptor.max_doc {
        assert_eq!(
            read_back.identity_of(position),
            segment.data.identities.identity_of(position)
        );
    }

    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![segment];
    cursor.ensure_searcher().unwrap();
    let reader = &cursor.searcher.as_ref().unwrap().segment_readers()[0];
    let hi = reader.fast_fields().u64(IDENTITY_HI_FIELD).unwrap();
    let lo = reader.fast_fields().u64(IDENTITY_LO_FIELD).unwrap();
    for (position, identity) in written.iter().enumerate() {
        assert_eq!(
            hi.first(position as u32),
            Some((identity.raw() >> 64) as u64)
        );
        assert_eq!(lo.first(position as u32), Some(identity.raw() as u64));
    }

    let (other, _) = build_and_load_segment(&attachment, &[(4, "other")]);
    assert!(
        !written.contains(&other.data.identities.identity_of(0).unwrap()),
        "identities of different builds must not collide"
    );
}

#[test]
fn segment_load_rejects_the_old_identity_field() {
    let attachment = test_attachment();
    let mut schema = Schema::builder();
    let rowid = schema.add_i64_field(
        ROWID_FIELD,
        tantivy::schema::INDEXED | tantivy::schema::FAST,
    );
    let identity = schema.add_u64_field("doc_identity", tantivy::schema::FAST);
    let directory = BuildDirectory::default();
    let index = Index::create(directory.clone(), schema.build(), IndexSettings::default()).unwrap();
    let id = SegmentId::generate_random();
    let mut writer = SegmentWriter::for_segment(
        DEFAULT_MEMORY_BUDGET_BYTES,
        index.segment(index.new_segment_meta(id, 0)),
    )
    .unwrap();
    let mut document = TantivyDocument::default();
    document.add_i64(rowid, 7);
    document.add_u64(identity, 902);
    writer
        .add_document(AddOperation {
            opstamp: 0,
            document,
        })
        .unwrap();
    writer.finalize().unwrap();

    let scratch = attachment.shared.scratch_index(&attachment.schema).unwrap();
    let error = read_segment_identities(
        &scratch,
        &attachment.schema,
        id,
        1,
        directory.captured_files(),
    )
    .unwrap_err();
    assert!(matches!(&error, LimboError::Corrupt(_)));
    assert!(error.to_string().contains("rebuild the index"), "{error}");
}

#[test]
fn merge_keeps_document_identities_and_retires_only_dropped_tombstones() {
    let attachment = test_attachment();
    let (mut first, _) = build_and_load_segment(&attachment, &[(1, "alpha one"), (2, "alpha two")]);
    let (mut second, _) =
        build_and_load_segment(&attachment, &[(3, "alpha three"), (4, "alpha four")]);
    let first_ids = identities_of(&first);
    let second_ids = identities_of(&second);

    // Delete rowids 2 and 3: one document in each input segment.
    first.deleted.insert(1).unwrap();
    second.deleted.insert(0).unwrap();
    let dropped = [first_ids[1], second_ids[0]];
    let kept = [first_ids[0], second_ids[1]];

    let mut cursor = FtsCursor::new(&attachment);
    cursor.segments = vec![first.try_clone().unwrap(), second.try_clone().unwrap()];
    cursor.snapshot_loaded = true;
    let candidates: HashSet<SegmentId> = [first.id(), second.id()].into_iter().collect();
    cursor.stage_merge_of_segments(&candidates).unwrap();
    let publish = cursor.publish.take().expect("merge staged a publication");

    let PublishApply::ReplaceSegments(merged) = publish.apply else {
        panic!("a merge replaces the visible segment set");
    };
    assert_eq!(merged.len(), 1);
    let merged = merged.into_iter().next().unwrap();
    assert_eq!(merged.descriptor.max_doc, 2);
    assert_eq!(
        identities_of(&merged).into_iter().collect::<BTreeSet<_>>(),
        kept.into_iter().collect::<BTreeSet<_>>(),
        "surviving documents keep the identity they were indexed with"
    );

    // The rowid of each survivor still maps to its original identity.
    cursor.segments = vec![merged.try_clone().unwrap()];
    cursor.invalidate_snapshot_view();
    for (rowid, identity) in [(1, kept[0]), (4, kept[1])] {
        let postings = cursor.live_postings_for_rowid(rowid).unwrap();
        assert_eq!(postings.len(), 1);
        assert_eq!(
            merged.data.identities.identity_of(postings[0].1),
            Some(identity)
        );
    }
    // A tombstone written against an input segment still hides the
    // document in the merged one.
    assert_eq!(
        merged
            .data
            .identities
            .tombstoned_positions(std::iter::once(kept[1]), std::path::Path::new("."))
            .unwrap()
            .len(),
        1
    );

    // The merge deletes only the tombstone rows of the dropped documents.
    // It also deletes the chunk rows of both inputs. The registry rows are
    // not its job: the claim that runs before it deleted them already.
    let targets = publish
        .deleter
        .expect("merge retires rows")
        .targets()
        .to_vec();
    let tombstone_targets: Vec<&PathTarget> = targets
        .iter()
        .filter(|target| matches!(target, PathTarget::Exact(path) if path.starts_with(FTS2_TOMB_PREFIX)))
        .collect();
    assert_eq!(
        tombstone_targets,
        dropped
            .iter()
            .map(|identity| PathTarget::Exact(document_tombstone_path(*identity)))
            .collect::<Vec<_>>()
            .iter()
            .collect::<Vec<_>>()
    );
    for input in [&first, &second] {
        assert!(!targets.contains(&PathTarget::Exact(segment_registry_path(&input.id()))));
        assert!(targets.contains(&PathTarget::Prefix(segment_chunk_prefix(&input.id()))));
    }
}

#[test]
fn snapshots_with_different_segment_sets_do_not_share_searchers() {
    let attachment = test_attachment();
    let (segment_a, _) = build_and_load_segment(&attachment, &[(1, "alpha")]);
    let (segment_b, _) = build_and_load_segment(&attachment, &[(2, "beta")]);

    let key_a = searcher_key(std::slice::from_ref(&segment_a)).unwrap();
    let key_ab = searcher_key(&[segment_a.try_clone().unwrap(), segment_b]).unwrap();
    assert_ne!(key_a, key_ab);

    // Tombstone state is part of the identity.
    let mut tombstoned = segment_a.try_clone().unwrap();
    tombstoned.deleted.insert(0).unwrap();
    assert_ne!(
        searcher_key(std::slice::from_ref(&segment_a)).unwrap(),
        searcher_key(std::slice::from_ref(&tombstoned)).unwrap()
    );
}

#[test]
fn segment_byte_cache_bounds_retention_reuses_mappings_and_preserves_active_pins() {
    let (segment, _) = build_and_load_segment(&test_attachment(), &[(1, "sample")]);
    let identities = segment.data.identities.clone();
    let directory = tempfile::tempdir().unwrap();
    let make_data = |bytes: usize| {
        let mut staged = StagedFile::new(directory.path()).unwrap();
        staged.append(0, &vec![7u8; bytes]).unwrap();
        let mut files = HashMap::default();
        files.insert("f".to_string(), staged.finish().unwrap());
        Arc::new(SegmentData::new(files, identities.clone()))
    };
    let mut cache = SegmentByteCache::default();
    let a = SegmentId::generate_random();
    let b = SegmentId::generate_random();
    let c = SegmentId::generate_random();
    let active = cache.put(a, make_data(100), 250);
    let active_weak = Arc::downgrade(&active);
    let retained = cache.put(b, make_data(100), 250);
    let retained_weak = Arc::downgrade(&retained);
    drop(retained);
    cache.put(c, make_data(100), 250);
    assert_eq!(cache.total_bytes(), 200);
    assert!(cache.get(&a).is_none());
    assert_eq!(&active.files["f"][..], &[7u8; 100]);
    assert!(
        active_weak.upgrade().is_some(),
        "active pin outlives eviction"
    );

    let duplicate = make_data(100);
    let duplicate_weak = Arc::downgrade(&duplicate);
    let canonical = cache.put(b, duplicate, 250);
    assert!(Arc::ptr_eq(&canonical, &retained_weak.upgrade().unwrap()));
    assert!(duplicate_weak.upgrade().is_none());
    drop(canonical);

    let oversized = cache.put(a, make_data(1000), 250);
    assert!(cache.get(&a).is_none());
    assert_eq!(cache.total_bytes(), 200);
    assert_eq!(oversized.files["f"].len(), 1000);
    drop(oversized);
    assert!(cache.get(&b).is_some());
    assert!(cache.get(&c).is_some());
    drop(cache);
    assert!(retained_weak.upgrade().is_none());
    assert!(active_weak.upgrade().is_some());
    drop(active);
    assert!(active_weak.upgrade().is_none());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn query_rowid_readers_follow_cached_searcher_order_and_snapshot() {
    let attachment = test_attachment();
    let (first, _) = build_and_load_segment(
        &attachment,
        &[(91, "alpha alpha"), (-7, "alpha beta gamma")],
    );
    let (second, _) =
        build_and_load_segment(&attachment, &[(400, "beta alpha"), (13, "beta delta")]);
    let mut original = FtsCursor::new(&attachment);
    original.segments = vec![first.try_clone().unwrap(), second.try_clone().unwrap()];
    original.ensure_searcher().unwrap();

    let mut cached = FtsCursor::new(&attachment);
    cached.segments = vec![second, first];
    cached.ensure_searcher().unwrap();
    assert!(Arc::ptr_eq(&original.rowid_readers, &cached.rowid_readers));

    for cursor in [&mut original, &mut cached] {
        for pattern in [FTS_PATTERN_MATCH, FTS_PATTERN_COMBINED] {
            let mut hits = query_hits(cursor, pattern, "alpha", -1);
            hits.sort_by_key(|hit| hit.0);
            assert_eq!(
                hits.iter().map(|hit| hit.0).collect::<Vec<_>>(),
                [-7, 91, 400]
            );
            if pattern == FTS_PATTERN_COMBINED {
                assert!(hits.iter().all(|hit| hit.1 != Value::from_f64(0.0)));
            }
        }
        for query in ["\"alpha beta\"", "alpha AND gamma"] {
            assert_eq!(
                query_hits(cursor, FTS_PATTERN_MATCH, query, -1),
                vec![(-7, Value::from_i64(1))]
            );
        }
        assert!(query_hits(cursor, FTS_PATTERN_MATCH_LIMIT, "alpha", 0).is_empty());
        assert_eq!(
            query_hits(cursor, FTS_PATTERN_MATCH_LIMIT, "alpha", 1).len(),
            1
        );
    }

    let ranked = query_hits(&mut original, FTS_PATTERN_COMBINED_ORDERED, "alpha", -1);
    assert_eq!(
        ranked.iter().map(|hit| hit.0).collect::<Vec<_>>(),
        [91, 400, -7]
    );
    assert_eq!(
        query_hits(&mut cached, FTS_PATTERN_COMBINED_ORDERED_LIMIT, "alpha", 2),
        ranked[..2]
    );

    cached.segments[1].deleted.insert(0).unwrap();
    cached.invalidate_snapshot_view();
    assert!(cached.rowid_readers.is_empty());
    let deleted = query_hits(&mut cached, FTS_PATTERN_COMBINED_ORDERED_LIMIT, "alpha", 1);
    assert_eq!(deleted[0].0, 400);
    assert!(!Arc::ptr_eq(&original.rowid_readers, &cached.rowid_readers));
    assert_eq!(
        query_hits(&mut original, FTS_PATTERN_COMBINED_ORDERED, "alpha", -1),
        ranked
    );

    let (replacement, _) = build_and_load_segment(&attachment, &[(999, "alpha")]);
    cached.segments = vec![replacement];
    cached.invalidate_snapshot_view();
    cached.build_snapshot_view(false).unwrap();
    assert_eq!(
        query_hits(&mut cached, FTS_PATTERN_MATCH, "alpha", -1),
        vec![(999, Value::from_i64(1))]
    );
}

fn query_hits(cursor: &mut FtsCursor, pattern: i64, query: &str, limit: i64) -> Vec<(i64, Value)> {
    let values = [
        Register::Value(Value::from_i64(pattern)),
        Register::Value(Value::from_text(query.to_owned())),
        Register::Value(Value::from_i64(limit)),
    ];
    let mut hits = Vec::new();
    let mut next = cursor.query_start(&values).unwrap();
    while let IOResult::Done(true) = next {
        let IOResult::Done(Some(rowid)) = cursor.query_rowid().unwrap() else {
            panic!("query must have a rowid");
        };
        let IOResult::Done(score) = cursor.query_column(0).unwrap() else {
            panic!("query column must not yield");
        };
        hits.push((rowid, score));
        next = cursor.query_next().unwrap();
    }
    assert!(matches!(next, IOResult::Done(false)));
    hits
}

#[test]
fn fts_write_errors_do_not_infer_out_of_memory_from_the_message() {
    let directory = BuildDirectory::default();
    let other = std::io::Error::other("memory allocation failed");
    assert!(matches!(
        directory.write_error(other.into(), "FTS build"),
        LimboError::InternalError(message) if message == "FTS build: An IO error occurred: 'memory allocation failed'"
    ));
}

#[cfg(nightly)]
mod allocation_failures {
    use super::*;
    use crate::DatabaseAllocators;
    use crate::alloc::{AllocError, ApiAllocator, Global, Layout};
    use std::io::{ErrorKind, Write};
    use std::ptr::NonNull;
    use std::sync::atomic::AtomicIsize;
    use tantivy::directory::{Directory, TerminatingWrite};

    #[test]
    fn atomic_write_failure_preserves_previous_metadata() {
        let allocator = FailingAllocator {
            #[cfg(feature = "allocation_metric")]
            expected_site: Some(crate::alloc::FtsAllocationSite::AtomicMetadata.into()),
            ..Default::default()
        };
        let directory = BuildDirectory::new(DynAllocator::new(allocator.clone()));
        let path = std::path::Path::new("meta.json");

        allocator.fail_after(0);
        assert_eq!(
            directory.atomic_write(path, b"first").unwrap_err().kind(),
            ErrorKind::OutOfMemory
        );
        assert!(!directory.exists(path).unwrap());
        directory.atomic_write(path, b"first").unwrap();

        allocator.fail_after(0);
        assert_eq!(
            directory
                .atomic_write(path, b"replacement")
                .unwrap_err()
                .kind(),
            ErrorKind::OutOfMemory
        );
        assert_eq!(directory.atomic_read(path).unwrap(), b"first");
        directory.atomic_write(path, b"replacement").unwrap();
        assert_eq!(directory.atomic_read(path).unwrap(), b"replacement");
    }

    #[test]
    fn capture_growth_failure_preserves_bytes_and_does_not_publish() {
        let allocator = FailingAllocator {
            #[cfg(feature = "allocation_metric")]
            expected_site: Some(crate::alloc::FtsAllocationSite::CaptureBuffer.into()),
            ..Default::default()
        };
        let directory = BuildDirectory::new(DynAllocator::new(allocator.clone()));
        let path = std::path::Path::new("segment.idx");
        let mut writer = directory.open_write(path).unwrap();
        writer.get_mut().write_all(b"prefix").unwrap();

        allocator.fail_after(0);
        let suffix = [37; 16_384];
        assert_eq!(
            writer.get_mut().write_all(&suffix).unwrap_err().kind(),
            ErrorKind::OutOfMemory
        );
        assert!(!directory.exists(path).unwrap());
        writer.get_mut().write_all(b"suffix").unwrap();
        writer.terminate().unwrap();
        assert_eq!(&*directory.captured_files()[path], b"prefixsuffix");

        let abandoned = std::path::Path::new("abandoned.idx");
        let mut writer = directory.open_write(abandoned).unwrap();
        writer.get_mut().write_all(b"partial").unwrap();
        allocator.fail_after(0);
        assert!(writer.get_mut().write_all(&suffix).is_err());
        drop(writer);
        assert!(!directory.exists(abandoned).unwrap());
    }

    #[test]
    fn failed_fts_allocations_roll_back_statements_and_allow_retry() {
        for merge in [false, true] {
            for mvcc in [false, true] {
                let (conn, allocator) = database_with_failing_fts_allocator(mvcc);
                if merge {
                    conn.execute("INSERT INTO docs VALUES (19, 'hello world')")
                        .unwrap();
                }
                let sql = if merge {
                    "OPTIMIZE INDEX docs_fts"
                } else {
                    "INSERT INTO docs VALUES (19, 'hello world')"
                };
                allocator.allocations.store(0, Ordering::Relaxed);
                conn.execute(sql).unwrap();
                let allocation_count = allocator.allocations.load(Ordering::Relaxed);
                assert!(allocation_count > 0);

                for fail_at in 0..allocation_count {
                    let (conn, allocator) = database_with_failing_fts_allocator(mvcc);
                    if merge {
                        conn.execute("INSERT INTO docs VALUES (19, 'hello world')")
                            .unwrap();
                    }
                    allocator.fail_after(fail_at);
                    let result = conn.execute(sql);
                    assert_eq!(
                        allocator.remaining.load(Ordering::Relaxed),
                        -1,
                        "{sql}, mvcc={mvcc}, fail_at={fail_at}"
                    );
                    assert!(
                        matches!(result, Err(LimboError::OutOfMemory)),
                        "{sql}, mvcc={mvcc}, fail_at={fail_at}: {result:?}"
                    );

                    let expected = if merge { vec![7, 19] } else { vec![7] };
                    assert_fts_rows(&conn, &expected);
                    conn.execute(sql).unwrap();
                    assert_fts_rows(&conn, &[7, 19]);
                }
            }
        }
    }

    fn database_with_failing_fts_allocator(mvcc: bool) -> (Arc<Connection>, FailingAllocator) {
        let allocator = FailingAllocator::default();
        let db = crate::Database::open(
            Arc::new(crate::MemoryIO::new()),
            ":memory:",
            crate::OpenOptions::new(Arc::new(crate::SqliteDialect))
                .db_opts(crate::DatabaseOpts::default().with_index_method(true))
                .allocators(DatabaseAllocators {
                    fts: DynAllocator::new(allocator.clone()),
                    ..Default::default()
                }),
        )
        .unwrap();
        let conn = db.connect().unwrap();
        if mvcc {
            conn.execute("PRAGMA journal_mode = 'mvcc'").unwrap();
        }
        conn.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT)")
            .unwrap();
        conn.execute("CREATE INDEX docs_fts ON docs USING fts(body)")
            .unwrap();
        conn.execute("INSERT INTO docs VALUES (7, 'hello turso')")
            .unwrap();
        (conn, allocator)
    }

    fn assert_fts_rows(conn: &Arc<Connection>, ids: &[i64]) {
        let expected: Vec<Vec<Value>> = ids.iter().map(|id| vec![Value::from_i64(*id)]).collect();
        for sql in [
            "SELECT id FROM docs ORDER BY id",
            "SELECT id FROM docs WHERE fts_match(body, 'hello') ORDER BY id",
        ] {
            let rows = conn.prepare(sql).unwrap().run_collect_rows().unwrap();
            assert_eq!(rows, expected, "{sql}");
        }
    }

    #[derive(Clone)]
    struct FailingAllocator {
        remaining: Arc<AtomicIsize>,
        allocations: Arc<AtomicUsize>,
        #[cfg(feature = "allocation_metric")]
        expected_site: Option<crate::alloc::AllocationSite>,
    }

    impl Default for FailingAllocator {
        fn default() -> Self {
            Self {
                remaining: Arc::new(AtomicIsize::new(-1)),
                allocations: Arc::new(AtomicUsize::new(0)),
                #[cfg(feature = "allocation_metric")]
                expected_site: None,
            }
        }
    }

    impl FailingAllocator {
        fn fail_after(&self, allocations: usize) {
            self.remaining
                .store(allocations.try_into().unwrap(), Ordering::Relaxed);
        }
    }

    unsafe impl ApiAllocator for FailingAllocator {
        fn allocate(&self, layout: Layout) -> std::result::Result<NonNull<[u8]>, AllocError> {
            #[cfg(feature = "allocation_metric")]
            if let Some(expected) = self.expected_site {
                assert_eq!(crate::alloc::current_allocation_site(), Some(expected));
            }
            self.allocations.fetch_add(1, Ordering::Relaxed);
            let previous =
                self.remaining
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                        (remaining >= 0).then(|| remaining - 1)
                    });
            if previous == Ok(0) {
                return Err(AllocError);
            }
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { Global.deallocate(ptr, layout) }
        }
    }
}

#[test]
fn prefix_queries_preserve_boolean_fields_unicode_and_long_tokens() {
    for mvcc in [false, true] {
        let db = crate::Database::open(
            Arc::new(crate::MemoryIO::new()),
            ":memory:",
            crate::OpenOptions::new(Arc::new(crate::SqliteDialect))
                .db_opts(crate::DatabaseOpts::default().with_index_method(true)),
        )
        .unwrap();
        let conn = db.connect().unwrap();
        if mvcc {
            conn.execute("PRAGMA journal_mode = 'mvcc'").unwrap();
        }
        conn.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
            .unwrap();
        conn.execute(
            "CREATE INDEX docs_fts ON docs USING fts(title, body) WITH (tokenizer = 'unicode')",
        )
        .unwrap();
        let long = "LongIdentifier".repeat(8);
        conn.execute(format!("INSERT INTO docs VALUES (1, 'Résumé', 'CAFÉ quick fox zebra {long}'), (2, 'cafe', 'quick food'), (3, 'unrelated', 'quicker forest')")).unwrap();
        for (query, expected) in [
            ("quick*".to_owned(), vec![1, 2, 3]),
            ("\"quick\"*".to_owned(), vec![1, 2, 3]),
            ("quick".to_owned(), vec![1, 2]),
            ("body:\"CAF\"*".to_owned(), vec![1]),
            ("title:\"RÉSU\"*".to_owned(), vec![1]),
            ("\"z\"*".to_owned(), vec![1]),
            (format!("\"{}\"*", &long[..60]), vec![1]),
            ("\"quick\"* AND \"fox\"*".to_owned(), vec![1]),
            ("\"quick\"* NOT \"food\"*".to_owned(), vec![1, 3]),
            ("body:\"quick fo\"*".to_owned(), vec![1, 2]),
            ("title:\"quick\"*".to_owned(), vec![]),
            ("\"absent\"* OR \"fox\"*".to_owned(), vec![1]),
        ] {
            let rows = conn
                .prepare(format!(
                    "SELECT id FROM docs WHERE fts_match(title, body, '{query}') ORDER BY id"
                ))
                .unwrap()
                .run_collect_rows()
                .unwrap();
            let expected: Vec<_> = expected
                .into_iter()
                .map(|id| vec![Value::from_i64(id)])
                .collect();
            assert_eq!(rows, expected, "{query}, mvcc={mvcc}");
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "isolated cold-process FTS memory and concurrency measurement"]
fn fts_query_memory_profile() {
    const TEST: &str = "index_method::fts::tests::fts_query_memory_profile";
    const PATH_ENV: &str = "TURSO_FTS_PROFILE_FIXTURE";
    const CLIENTS_ENV: &str = "TURSO_FTS_PROFILE_CLIENTS";
    fn memory(stage: &str) {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let fields: Vec<_> = status
            .lines()
            .filter(|line| {
                ["VmHWM:", "VmRSS:", "RssAnon:", "RssFile:"]
                    .iter()
                    .any(|key| line.starts_with(key))
            })
            .collect();
        println!("FTS_MEMORY {stage} {}", fields.join(" "));
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            let io: Vec<_> = io
                .lines()
                .filter(|line| line.starts_with("read_bytes:") || line.starts_with("write_bytes:"))
                .collect();
            println!("FTS_IO {stage} {}", io.join(" "));
        }
        if let Ok(groups) = std::fs::read_to_string("/proc/self/cgroup") {
            if let Some(group) = groups.lines().find_map(|line| line.strip_prefix("0::")) {
                let root =
                    std::path::Path::new("/sys/fs/cgroup").join(group.trim_start_matches('/'));
                if let Ok(bytes) = std::fs::read_to_string(root.join("memory.current")) {
                    println!(
                        "FTS_CGROUP {stage} current_bytes={} shared_with_other_processes=true",
                        bytes.trim()
                    );
                }
            }
        }
    }
    fn open(path: &str) -> Arc<crate::Database> {
        crate::Database::open(
            Arc::new(crate::PlatformIO::new().unwrap()),
            path,
            crate::OpenOptions::new(Arc::new(crate::SqliteDialect))
                .db_opts(crate::DatabaseOpts::default().with_index_method(true)),
        )
        .unwrap()
    }
    if let Ok(path) = std::env::var(PATH_ENV) {
        let clients: usize = std::env::var(CLIENTS_ENV).unwrap().parse().unwrap();
        memory("before_open");
        let db = open(&path);
        memory("before_query");
        let connections: Vec<_> = (0..clients).map(|_| db.connect().unwrap()).collect();
        let ready = std::sync::Barrier::new(clients);
        let started = std::time::Instant::now();
        let waves = std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for conn in &connections {
                let ready = &ready;
                workers.push(scope.spawn(move || {
                    let mut timings = Vec::new();
                    ready.wait();
                    for _ in 0..11 {
                        let wave = std::time::Instant::now();
                        for (query, count) in [("needle00000123", 1), ("common", 10)] {
                            let rows = conn.prepare(format!(
                                "SELECT id FROM docs WHERE fts_match(body, '{query}') ORDER BY id LIMIT 10"
                            )).unwrap().run_collect_rows().unwrap();
                            assert_eq!(rows.len(), count);
                            if count == 1 { assert_eq!(rows[0], vec![Value::from_i64(123)]); }
                        }
                        timings.push(wave.elapsed().as_micros());
                    }
                    timings
                }));
            }
            let timings: Vec<_> = workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect();
            (0..11)
                .map(|wave| timings.iter().map(|client| client[wave]).max().unwrap())
                .collect::<Vec<_>>()
        });
        for (wave, elapsed) in waves.iter().enumerate() {
            println!("FTS_QUERY clients={clients} wave={wave} elapsed_us={elapsed}");
        }
        let mut warm = waves[1..].to_vec();
        warm.sort_unstable();
        println!(
            "FTS_WARM clients={clients} p50_us={} p95_us={} total_ms={}",
            warm[4],
            warm[9],
            started.elapsed().as_millis()
        );
        memory("queries_complete_connections_live");
        drop(connections);
        drop(db);
        memory("after_close");
        return;
    }
    let fixture = tempfile::tempdir().unwrap();
    let path = fixture.path().join("fts.db");
    {
        let db = open(path.to_str().unwrap());
        let conn = db.connect().unwrap();
        conn.execute("PRAGMA journal_mode = 'mvcc'").unwrap();
        conn.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT)")
            .unwrap();
        conn.execute("CREATE INDEX docs_fts ON docs USING fts(body)")
            .unwrap();
        let mut random = 0x45fa_9621_77a3_619bu64;
        for batch in 0..128 {
            let mut sql = String::from("INSERT INTO docs VALUES ");
            for offset in 0..32 {
                if offset != 0 {
                    sql.push(',');
                }
                let id = batch * 32 + offset;
                sql.push_str(&format!("({id}, 'common needle{id:08} "));
                for _ in 0..256 {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    sql.push_str(&format!("word{random:016x} "));
                }
                sql.push_str("')");
            }
            conn.execute(sql).unwrap();
        }
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    }
    for clients in [1, 8, 20] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                TEST,
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PATH_ENV, &path)
            .env(CLIENTS_ENV, clients.to_string())
            .env("HOME", fixture.path())
            .env("XDG_CACHE_HOME", fixture.path())
            .env("XDG_DATA_HOME", fixture.path())
            .output()
            .unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

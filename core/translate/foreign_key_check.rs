//! Read-only foreign-key audit compiled from the same resolved schema as enforcement.

use crate::sync::Arc;
use crate::translate::emitter::Resolver;
use crate::translate::plan::{Plan, QueryDestination, SelectPlan};
use crate::translate::select::{emit_select_plan, prepare_select_plan};
use crate::util::{escape_sql_string_literal, quote_identifier};
use crate::vdbe::builder::ProgramBuilder;
use crate::{Connection, LimboError, Result};
use turso_parser::{ast, parser::Parser};

fn expose_child_rowid(plan: &mut SelectPlan, database_id: usize) {
    // Bind the physical rowid directly, including when every SQL rowid spelling
    // is shadowed by a declared column. WITHOUT ROWID tables retain NULL.
    if let Some(child) = plan.joined_tables().first() {
        if child.table.btree().is_some_and(|table| table.has_rowid) {
            plan.result_columns[1].expr = ast::Expr::RowId {
                database: Some(database_id),
                table: child.internal_id,
            };
        }
    }
}

pub(super) fn translate_foreign_key_check(
    resolver: &Resolver,
    database_id: usize,
    table_name: Option<&str>,
    program: &mut ProgramBuilder,
    connection: &Arc<Connection>,
) -> Result<()> {
    let database_name = resolver
        .get_database_name_by_index(database_id)
        .ok_or_else(|| LimboError::InternalError("foreign key audit database missing".into()))?;
    let database = quote_identifier(&database_name);
    let queries = resolver.with_schema(database_id, |schema| -> Result<Vec<String>> {
        let mut tables = match table_name {
            Some(name) => vec![schema.get_btree_table(name)
                .ok_or_else(|| LimboError::ParseError(format!("no such table: {name}")))?],
            None => schema.tables.values().filter_map(|table| table.btree()).collect(),
        };
        tables.sort_by(|a, b| a.name.cmp(&b.name));
        let mut queries = Vec::new();
        for child in tables {
            let mut foreign_keys = child.foreign_keys.iter().collect::<Vec<_>>();
            foreign_keys.sort_by_key(|fk| std::cmp::Reverse(fk.decl_order));
            for (id, fk) in foreign_keys.into_iter().enumerate() {
                let mut predicates = fk.child_columns.iter().map(|name| {
                    format!("child.{} IS NOT NULL", quote_identifier(name))
                }).collect::<Vec<_>>();
                if let Some(parent) = schema.get_btree_table(&fk.parent_table) {
                    let resolved = schema.resolve_fk(fk, &child, &parent, true)?;
                    let comparisons = resolved.parent_cols.iter().zip(fk.child_columns.iter())
                        .map(|(parent, child)| {
                            // Parent affinity and collation govern FK equality. Unary
                            // plus removes child-column affinity without changing its value.
                            format!("parent.{} = +child.{}", quote_identifier(parent), quote_identifier(child))
                        }).collect::<Vec<_>>().join(" AND ");
                    predicates.push(format!("NOT EXISTS (SELECT 1 FROM {database}.{} AS parent WHERE {comparisons})", quote_identifier(&parent.name)));
                }
                queries.push(format!(
                    "SELECT '{}' AS \"table\", NULL AS rowid, '{}' AS parent, {id} AS fkid FROM {database}.{} AS child WHERE {}",
                    escape_sql_string_literal(&child.name), escape_sql_string_literal(&fk.parent_table),
                    quote_identifier(&child.name), predicates.join(" AND ")
                ));
            }
        }
        Ok(queries)
    })?;
    let sql = if queries.is_empty() {
        "SELECT NULL AS \"table\", NULL AS rowid, NULL AS parent, NULL AS fkid WHERE 0".to_owned()
    } else {
        queries.join(" UNION ALL ")
    };
    let mut parser = Parser::new(sql.as_bytes());
    let Some(ast::Cmd::Stmt(ast::Stmt::Select(select))) = parser.next_cmd()? else {
        return Err(LimboError::InternalError(
            "foreign key audit SELECT missing".into(),
        ));
    };
    let mut plan = prepare_select_plan(
        select,
        resolver,
        program,
        &[],
        QueryDestination::ResultRows,
        connection,
    )?;
    match &mut plan {
        Plan::Select(select) => expose_child_rowid(select, database_id),
        Plan::CompoundSelect {
            left, right_most, ..
        } => {
            for (select, _) in left {
                expose_child_rowid(select, database_id);
            }
            expose_child_rowid(right_most, database_id);
        }
        _ => {
            return Err(LimboError::InternalError(
                "unexpected foreign key audit plan".into(),
            ))
        }
    }
    emit_select_plan(plan, resolver, program, connection)?;
    Ok(())
}

use tantivy::Index;
use tantivy::query::{Query, QueryParser};
use tantivy::query_grammar::{Delimiter, Occur, UserInputAst, UserInputLeaf};
use tantivy::schema::Field;

/// Extend Tantivy's phrase-prefix syntax to single tokens using its native
/// regex query. Normalize with the index analyzer so Unicode and long terms
/// have exactly the same representation at indexing and query time.
pub(super) fn parse_query(
    parser: &QueryParser,
    index: &Index,
    default_fields: &[Field],
    boosts: &[(Field, f32)],
    text: &str,
) -> Result<Box<dyn Query>, String> {
    let (ast, errors) = tantivy::query_grammar::parse_query_lenient(text);
    if let Some(error) = errors.first() {
        return Err(format!("{error:?}"));
    }
    let ast = expand(ast, index, default_fields, boosts)?;
    parser
        .build_query_from_user_input_ast(ast)
        .map_err(|e| e.to_string())
}

fn expand(
    ast: UserInputAst,
    index: &Index,
    default_fields: &[Field],
    boosts: &[(Field, f32)],
) -> Result<UserInputAst, String> {
    Ok(match ast {
        UserInputAst::Clause(children) => UserInputAst::Clause(
            children
                .into_iter()
                .map(|(occur, child)| Ok((occur, expand(child, index, default_fields, boosts)?)))
                .collect::<Result<_, String>>()?,
        ),
        UserInputAst::Boost(child, boost) => UserInputAst::Boost(
            Box::new(expand(*child, index, default_fields, boosts)?),
            boost,
        ),
        UserInputAst::Leaf(leaf) => {
            let UserInputLeaf::Literal(mut literal) = *leaf else {
                return Ok(UserInputAst::Leaf(leaf));
            };
            if literal.delimiter == Delimiter::None && literal.phrase.ends_with('*') {
                literal.phrase.pop();
                literal.prefix = true;
            }
            if !literal.prefix {
                return Ok(UserInputAst::Leaf(Box::new(UserInputLeaf::Literal(
                    literal,
                ))));
            }
            let schema = index.schema();
            let fields = match &literal.field_name {
                Some(name) => vec![schema.get_field(name).map_err(|e| e.to_string())?],
                None => default_fields.to_vec(),
            };
            let mut clauses = Vec::with_capacity(fields.len());
            for field in fields {
                let mut analyzer = index
                    .tokenizer_for_field(field)
                    .map_err(|e| e.to_string())?;
                let mut terms = Vec::new();
                analyzer
                    .token_stream(&literal.phrase)
                    .process(&mut |token| terms.push(token.text.clone()));
                let child = if terms.len() == 1 {
                    let term = &terms[0];
                    let regex = UserInputLeaf::Regex {
                        field: Some(schema.get_field_name(field).to_owned()),
                        pattern: format!("{}.*", regex::escape(term)),
                    };
                    let child = UserInputAst::Leaf(Box::new(regex));
                    match boosts.iter().find(|(f, _)| *f == field) {
                        Some((_, boost)) => {
                            UserInputAst::Boost(Box::new(child), f64::from(*boost).into())
                        }
                        None => child,
                    }
                } else {
                    let mut field_literal = literal.clone();
                    field_literal.field_name = Some(schema.get_field_name(field).to_owned());
                    UserInputAst::Leaf(Box::new(UserInputLeaf::Literal(field_literal)))
                };
                clauses.push((Some(Occur::Should), child));
            }
            UserInputAst::Clause(clauses)
        }
    })
}

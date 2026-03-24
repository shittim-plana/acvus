//! AST-level placeholder substitution.
//!
//! Replaces dummy ident nodes (`__acvus_ph_<name>__`) with provided Expr values.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::ast::*;

/// Substitute placeholder idents in a Script AST.
pub fn substitute_script(script: Script, subs: &FxHashMap<Astr, Expr>) -> Script {
    Script {
        stmts: script.stmts.into_iter().map(|s| sub_stmt(s, subs)).collect(),
        tail: script.tail.map(|e| Box::new(sub_expr(*e, subs))),
        span: script.span,
    }
}

/// Substitute placeholder idents in a Template AST.
pub fn substitute_template(template: Template, subs: &FxHashMap<Astr, Expr>) -> Template {
    Template {
        body: template.body.into_iter().map(|n| sub_node(n, subs)).collect(),
        span: template.span,
    }
}

fn sub_expr(expr: Expr, subs: &FxHashMap<Astr, Expr>) -> Expr {
    match expr {
        Expr::Ident { name, ref_kind: RefKind::Value, .. } => {
            if let Some(replacement) = subs.get(&name) {
                replacement.clone()
            } else {
                expr
            }
        }
        Expr::Ident { .. } | Expr::Literal { .. } => expr,
        Expr::BinaryOp { left, op, right, span } => Expr::BinaryOp {
            left: Box::new(sub_expr(*left, subs)),
            op,
            right: Box::new(sub_expr(*right, subs)),
            span,
        },
        Expr::UnaryOp { op, operand, span } => Expr::UnaryOp {
            op,
            operand: Box::new(sub_expr(*operand, subs)),
            span,
        },
        Expr::FieldAccess { object, field, span } => Expr::FieldAccess {
            object: Box::new(sub_expr(*object, subs)),
            field,
            span,
        },
        Expr::FuncCall { func, args, span } => Expr::FuncCall {
            func: Box::new(sub_expr(*func, subs)),
            args: args.into_iter().map(|a| sub_expr(a, subs)).collect(),
            span,
        },
        Expr::Pipe { left, right, span } => Expr::Pipe {
            left: Box::new(sub_expr(*left, subs)),
            right: Box::new(sub_expr(*right, subs)),
            span,
        },
        Expr::Lambda { params, body, span } => Expr::Lambda {
            params,
            body: Box::new(sub_expr(*body, subs)),
            span,
        },
        Expr::Paren { inner, span } => Expr::Paren {
            inner: Box::new(sub_expr(*inner, subs)),
            span,
        },
        Expr::List { head, rest, tail, span } => Expr::List {
            head: head.into_iter().map(|e| sub_expr(e, subs)).collect(),
            rest,
            tail: tail.into_iter().map(|e| sub_expr(e, subs)).collect(),
            span,
        },
        Expr::Group { elements, span } => Expr::Group {
            elements: elements.into_iter().map(|e| sub_expr(e, subs)).collect(),
            span,
        },
        Expr::Object { fields, span } => Expr::Object {
            fields: fields
                .into_iter()
                .map(|f| ObjectExprField {
                    key: f.key,
                    value: sub_expr(f.value, subs),
                    span: f.span,
                })
                .collect(),
            span,
        },
        Expr::Range { start, end, kind, span } => Expr::Range {
            start: Box::new(sub_expr(*start, subs)),
            end: Box::new(sub_expr(*end, subs)),
            kind,
            span,
        },
        Expr::Tuple { elements, span } => Expr::Tuple {
            elements: elements
                .into_iter()
                .map(|e| match e {
                    TupleElem::Expr(expr) => TupleElem::Expr(sub_expr(expr, subs)),
                    w @ TupleElem::Wildcard(_) => w,
                })
                .collect(),
            span,
        },
        Expr::Block { stmts, tail, span } => Expr::Block {
            stmts: stmts.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            tail: Box::new(sub_expr(*tail, subs)),
            span,
        },
        Expr::Variant { enum_name, tag, payload, span } => Expr::Variant {
            enum_name,
            tag,
            payload: payload.map(|p| Box::new(sub_expr(*p, subs))),
            span,
        },
    }
}

fn sub_stmt(stmt: Stmt, subs: &FxHashMap<Astr, Expr>) -> Stmt {
    match stmt {
        Stmt::Bind { name, expr, span } => Stmt::Bind {
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::ContextStore { name, expr, span } => Stmt::ContextStore {
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::Expr(expr) => Stmt::Expr(sub_expr(expr, subs)),
        Stmt::MatchBind { pattern, source, body, span } => Stmt::MatchBind {
            pattern,
            source: sub_expr(source, subs),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
        Stmt::Iterate { pattern, source, body, span } => Stmt::Iterate {
            pattern,
            source: sub_expr(source, subs),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
    }
}

fn sub_node(node: Node, subs: &FxHashMap<Astr, Expr>) -> Node {
    match node {
        Node::Text { .. } | Node::Comment { .. } => node,
        Node::InlineExpr { expr, span } => Node::InlineExpr {
            expr: sub_expr(expr, subs),
            span,
        },
        Node::MatchBlock(mb) => Node::MatchBlock(MatchBlock {
            source: sub_expr(mb.source, subs),
            arms: mb
                .arms
                .into_iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern,
                    body: arm.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                    tag_span: arm.tag_span,
                })
                .collect(),
            catch_all: mb.catch_all.map(|ca| CatchAll {
                body: ca.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                tag_span: ca.tag_span,
            }),
            indent: mb.indent,
            span: mb.span,
        }),
        Node::IterBlock(ib) => Node::IterBlock(IterBlock {
            pattern: ib.pattern,
            source: sub_expr(ib.source, subs),
            body: ib.body.into_iter().map(|n| sub_node(n, subs)).collect(),
            catch_all: ib.catch_all.map(|ca| CatchAll {
                body: ca.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                tag_span: ca.tag_span,
            }),
            indent: ib.indent,
            span: ib.span,
        }),
    }
}

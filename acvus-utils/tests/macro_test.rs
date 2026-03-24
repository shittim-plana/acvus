use acvus_macro::{acvus_script, acvus_template};

#[test]
fn script_no_placeholder() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("1 + 2");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_context_read() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@x + 1");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_pipe_with_lambda() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@data | map(|x| -> x + 1)");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_lambda_pipe_in_body() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@data | map(|x| -> x | to_string)");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_placeholder_substitution() {
    let interner = acvus_utils::Interner::new();

    // Create an Expr to substitute: literal 42
    let replacement = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(42),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("%result + 1");
    let script = make(&interner, replacement);

    // The tail should be BinaryOp(Literal(42), Add, Literal(1))
    let tail = script.tail.as_ref().unwrap();
    match tail.as_ref() {
        acvus_ast::Expr::BinaryOp { left, op, .. } => {
            assert_eq!(*op, acvus_ast::BinOp::Add);
            match left.as_ref() {
                acvus_ast::Expr::Literal { value: acvus_ast::Literal::Int(42), .. } => {}
                other => panic!("expected Literal(42), got {other:?}"),
            }
        }
        other => panic!("expected BinaryOp, got {other:?}"),
    }
}

#[test]
fn script_multiple_placeholders() {
    let interner = acvus_utils::Interner::new();

    let a = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(10),
        span: acvus_ast::Span::ZERO,
    };
    let b = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(20),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("%a + %b");
    let script = make(&interner, a, b);

    let tail = script.tail.as_ref().unwrap();
    match tail.as_ref() {
        acvus_ast::Expr::BinaryOp { left, right, .. } => {
            match left.as_ref() {
                acvus_ast::Expr::Literal { value: acvus_ast::Literal::Int(10), .. } => {}
                other => panic!("expected Literal(10), got {other:?}"),
            }
            match right.as_ref() {
                acvus_ast::Expr::Literal { value: acvus_ast::Literal::Int(20), .. } => {}
                other => panic!("expected Literal(20), got {other:?}"),
            }
        }
        other => panic!("expected BinaryOp, got {other:?}"),
    }
}

#[test]
fn script_placeholder_with_context() {
    let interner = acvus_utils::Interner::new();

    let result = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::String("hello".into()),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("@history = append(@history, %result); @history");
    let script = make(&interner, result);

    // Should have 1 stmt (context store) + tail (@history)
    assert_eq!(script.stmts.len(), 1);
    assert!(script.tail.is_some());
}

#[test]
fn template_no_placeholder() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_template!("hello world");
    let template = make(&interner);
    assert!(!template.body.is_empty());
}

#[test]
fn template_with_placeholder() {
    let interner = acvus_utils::Interner::new();

    let value = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(99),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_template!("result: {{ %value }}");
    let template = make(&interner, value);
    assert!(!template.body.is_empty());
}

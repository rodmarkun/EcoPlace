//! AST-level API tests (TODO 1.8): typed subtree crossover for the same-DSL GP baseline
//! (C19) and numeric-literal patching for ShinkaEvolve.

use placementscript_core::astapi::*;
use placementscript_core::{compile, PsError};

const A: &str = "def f(a: int, b: int) -> int:\n    x = a + b * 2\n    return x\n";
const B: &str = "def g(a: int, c: int) -> int:\n    y = a - c\n    return y * y\n";

#[test]
fn enumeration_is_deterministic_and_typed() {
    let s1 = enumerate_subtrees(A).unwrap();
    let s2 = enumerate_subtrees(A).unwrap();
    assert_eq!(s1, s2);
    assert!(s1
        .iter()
        .any(|s| s.kind == "binary" && s.ty.as_deref() == Some("int")));
    assert!(s1.iter().any(|s| !s.is_expr && s.kind == "assign"));
    // Every expression site carries a type.
    assert!(s1.iter().filter(|s| s.is_expr).all(|s| s.ty.is_some()));
}

#[test]
fn typed_crossover_produces_compilable_offspring() {
    let sites_a = enumerate_subtrees(A).unwrap();
    let sites_b = enumerate_subtrees(B).unwrap();
    // Swap A's `a + b * 2` with B's `a - c`... `c` is not in scope in A → must fail
    // as a Type error (invalid offspring), not silently.
    let a_expr = sites_a
        .iter()
        .find(|s| s.kind == "binary" && s.is_expr)
        .unwrap();
    let b_expr = sites_b
        .iter()
        .find(|s| s.kind == "binary" && s.is_expr)
        .unwrap();
    let err = crossover(A, B, a_expr.id, b_expr.id).unwrap_err();
    assert!(matches!(err, PsError::Type { .. }));
    // A donor that only uses names present in A compiles.
    let b_name = sites_b
        .iter()
        .find(|s| s.kind == "name" && s.ty.as_deref() == Some("int"))
        .unwrap();
    let child = crossover(A, B, a_expr.id, b_name.id).unwrap();
    compile(&child).unwrap();
    assert!(child.contains("x = (a)"));
}

#[test]
fn crossover_type_mismatch_rejected() {
    let src_f = "def h(a: float) -> float:\n    return a * 0.5\n";
    let sa = enumerate_subtrees(A).unwrap();
    let sf = enumerate_subtrees(src_f).unwrap();
    let int_site = sa.iter().find(|s| s.ty.as_deref() == Some("int")).unwrap();
    let float_site = sf
        .iter()
        .find(|s| s.ty.as_deref() == Some("float"))
        .unwrap();
    let err = crossover(A, src_f, int_site.id, float_site.id).unwrap_err();
    assert!(matches!(err, PsError::Type { .. }));
}

#[test]
fn replace_subtree_mutation_hook() {
    let sites = enumerate_subtrees(A).unwrap();
    let site = sites.iter().find(|s| s.kind == "binary").unwrap();
    let child = replace_subtree(A, site.id, "min(a, b) * 3").unwrap();
    compile(&child).unwrap();
    assert!(child.contains("(min(a, b) * 3)"));
    // Garbage replacement text is rejected with a structured error.
    assert!(replace_subtree(A, site.id, "def nope").is_err());
}

#[test]
fn numeric_literal_enumeration_and_patching() {
    let src = "def f(a: float) -> float:\n    if a > 0.75:\n        return a * 2.0 - 1.5\n    return a + 42.0\n";
    let lits = enumerate_numeric_literals(src).unwrap();
    let texts: Vec<&str> = lits.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(texts, ["0.75", "2.0", "1.5", "42.0"]);
    assert!(lits.iter().all(|l| l.is_float));

    let patched = patch_numeric_literals(
        src,
        &[
            (lits[0].id, "0.9".to_string()),
            (lits[3].id, "13.5".to_string()),
        ],
    )
    .unwrap();
    assert!(patched.contains("a > 0.9") && patched.contains("a + 13.5"));
    compile(&patched).unwrap();

    // Kind mismatch (float literal patched with int text) is rejected.
    let err = patch_numeric_literals(src, &[(lits[0].id, "7".to_string())]).unwrap_err();
    assert!(matches!(err, PsError::Type { .. }));
    // Negative replacement rejected (sign lives in the AST, not the literal).
    let err = patch_numeric_literals(src, &[(lits[0].id, "-0.5".to_string())]).unwrap_err();
    assert!(matches!(err, PsError::Type { .. }));
}

#[test]
fn int_literal_patching() {
    let src = "def f(n: int) -> int:\n    acc = 0\n    for i in range(n):\n        acc += i % 10\n    return acc\n";
    let lits = enumerate_numeric_literals(src).unwrap();
    assert!(lits.iter().all(|l| !l.is_float));
    let ten = lits.iter().find(|l| l.text == "10").unwrap();
    let patched = patch_numeric_literals(src, &[(ten.id, "7".to_string())]).unwrap();
    assert!(patched.contains("acc += i % 7"));
    // Out-of-range replacement rejected.
    let err =
        patch_numeric_literals(src, &[(ten.id, "99999999999999999999".to_string())]).unwrap_err();
    assert!(matches!(err, PsError::Type { .. }));
}

#[test]
fn statement_crossover() {
    let src_a = "def f(a: int) -> int:\n    x = a + 1\n    return x\n";
    let src_b = "def g(a: int) -> int:\n    x = a * 3\n    return x\n";
    let sa = enumerate_subtrees(src_a).unwrap();
    let sb = enumerate_subtrees(src_b).unwrap();
    let a_stmt = sa
        .iter()
        .find(|s| !s.is_expr && s.kind == "assign")
        .unwrap();
    let b_stmt = sb
        .iter()
        .find(|s| !s.is_expr && s.kind == "assign")
        .unwrap();
    let child = crossover(src_a, src_b, a_stmt.id, b_stmt.id).unwrap();
    compile(&child).unwrap();
    assert!(child.contains("x = a * 3"));
    // A donor statement reading names unavailable at the target site is rejected on
    // recompile (definite assignment), i.e. an invalid offspring — not a broken program.
    let src_c = "def h(q: int) -> int:\n    x = q * 3\n    return x\n";
    let sc = enumerate_subtrees(src_c).unwrap();
    let c_stmt = sc
        .iter()
        .find(|s| !s.is_expr && s.kind == "assign")
        .unwrap();
    assert!(matches!(
        crossover(src_a, src_c, a_stmt.id, c_stmt.id).unwrap_err(),
        PsError::Type { .. }
    ));
}

// --- Round-1 review R1.10: full-grammar GP sites ---

const WITH_BLOCKS: &str = "def f(a: int) -> int:\n    acc = 0\n    for i in range(a):\n        if i % 2 == 0:\n            acc += i\n    return acc\n";
const WITH_BLOCKS_B: &str =
    "def g(a: int) -> int:\n    t = 0\n    if a > 3:\n        t = a * 2\n    return t\n";

#[test]
fn block_statements_and_functions_are_sites() {
    let sites = enumerate_subtrees(WITH_BLOCKS).unwrap();
    assert!(sites.iter().any(|s| s.kind == "for" && !s.is_expr));
    assert!(sites.iter().any(|s| s.kind == "if" && !s.is_expr));
    assert!(sites.iter().any(|s| s.kind == "func" && !s.is_expr));
    // The for-site's span covers its nested if-body.
    let for_site = sites.iter().find(|s| s.kind == "for").unwrap();
    let text = &WITH_BLOCKS[for_site.start as usize..for_site.end as usize];
    assert!(text.contains("if i % 2 == 0:") && text.ends_with("acc += i"));
}

#[test]
fn sites_carry_parent_and_depth() {
    let sites = enumerate_subtrees(WITH_BLOCKS).unwrap();
    let func = sites.iter().find(|s| s.kind == "func").unwrap();
    assert_eq!((func.parent, func.depth), (None, 0));
    let for_site = sites.iter().find(|s| s.kind == "for").unwrap();
    assert_eq!(for_site.parent, Some(func.id));
    assert_eq!(for_site.depth, 1);
    let if_site = sites.iter().find(|s| s.kind == "if").unwrap();
    assert_eq!(if_site.parent, Some(for_site.id));
    assert_eq!(if_site.depth, 2);
    let aug = sites.iter().find(|s| s.kind == "aug_assign").unwrap();
    assert_eq!(aug.parent, Some(if_site.id));
    assert_eq!(aug.depth, 3);
}

#[test]
fn block_statement_crossover_reindents() {
    // Swap A's nested `if` (depth 2, 8-space body) with B's top-level `if`
    // (depth 1, 4-space body): the donor must be reindented, and vice versa.
    let sa = enumerate_subtrees(WITH_BLOCKS).unwrap();
    let sb = enumerate_subtrees(WITH_BLOCKS_B).unwrap();
    let a_if = sa.iter().find(|s| s.kind == "if").unwrap();
    let b_if = sb.iter().find(|s| s.kind == "if").unwrap();
    let child = crossover(WITH_BLOCKS, WITH_BLOCKS_B, a_if.id, b_if.id).unwrap();
    compile(&child).unwrap();
    // Donor references `t`, undefined in f → wait: `t = a * 2` assigns; `if a > 3:`
    // guard reads a — both fine inside f. The reindented body must sit at 12 spaces.
    assert!(child.contains("        if a > 3:\n            t = a * 2\n"));

    // Statement × block-statement is also legal (both statements).
    let b_assign = sb
        .iter()
        .find(|s| s.kind == "assign" && s.depth == 1)
        .unwrap();
    let child2 = crossover(WITH_BLOCKS, WITH_BLOCKS_B, a_if.id, b_assign.id).unwrap();
    compile(&child2).unwrap();

    // Expression × statement stays rejected.
    let a_expr = sa.iter().find(|s| s.is_expr).unwrap();
    assert!(crossover(WITH_BLOCKS, WITH_BLOCKS_B, a_expr.id, b_if.id).is_err());
}

#[test]
fn function_crossover_swaps_whole_definitions() {
    let sa = enumerate_subtrees(WITH_BLOCKS).unwrap();
    let sb = enumerate_subtrees(WITH_BLOCKS_B).unwrap();
    let a_fn = sa.iter().find(|s| s.kind == "func").unwrap();
    let b_fn = sb.iter().find(|s| s.kind == "func").unwrap();
    let child = crossover(WITH_BLOCKS, WITH_BLOCKS_B, a_fn.id, b_fn.id).unwrap();
    compile(&child).unwrap();
    assert!(child.starts_with("def g(a: int) -> int:"));
    // Function sites cross only with function sites.
    let b_if = sb.iter().find(|s| s.kind == "if").unwrap();
    assert!(crossover(WITH_BLOCKS, WITH_BLOCKS_B, a_fn.id, b_if.id).is_err());
}

#[test]
fn multiline_statement_replacement_reindented() {
    let sites = enumerate_subtrees(WITH_BLOCKS).unwrap();
    let if_site = sites.iter().find(|s| s.kind == "if").unwrap();
    // Replacement written at column 0; must be shifted to the site's column (8).
    let child = replace_subtree(WITH_BLOCKS, if_site.id, "if i > 0:\n    acc += 1").unwrap();
    compile(&child).unwrap();
    assert!(child.contains("        if i > 0:\n            acc += 1\n"));
}

#[test]
fn module_constant_literals_enumerable_and_patchable() {
    let src = "THRESH: float = 0.25\nBOOST: int = -3\ndef f(a: float) -> float:\n    if a > THRESH:\n        return a * 2.0\n    return a\n";
    let lits = enumerate_numeric_literals(src).unwrap();
    let consts: Vec<_> = lits.iter().filter(|l| l.is_const).collect();
    assert_eq!(consts.len(), 2);
    assert_eq!(consts[0].text, "0.25");
    assert_eq!(consts[1].text, "-3");
    // Patch the float const (and a body literal) in one shot.
    let patches = vec![
        (consts[0].id, "0.4".to_string()),
        (
            lits.iter().find(|l| l.text == "2.0").unwrap().id,
            "1.5".to_string(),
        ),
    ];
    let out = patch_numeric_literals(src, &patches).unwrap();
    assert!(out.contains("THRESH: float = 0.4") && out.contains("a * 1.5"));
    // Negative replacement is legal for constants, not for expression literals.
    let out2 = patch_numeric_literals(src, &[(consts[1].id, "-7".into())]).unwrap();
    assert!(out2.contains("BOOST: int = -7"));
    let body_lit = lits.iter().find(|l| l.text == "2.0").unwrap();
    assert!(patch_numeric_literals(src, &[(body_lit.id, "-1.0".into())]).is_err());
}

#[test]
fn underscore_literals_validated_through_the_lexer() {
    let src = "def f() -> int:\n    return 1000\n";
    let lits = enumerate_numeric_literals(src).unwrap();
    // "1_000" is a legal literal per spec §1 — the old ad-hoc validation rejected it.
    let out = patch_numeric_literals(src, &[(lits[0].id, "1_000".into())]).unwrap();
    assert!(out.contains("return 1_000"));
    // Smuggled trailing content is rejected even though it lexes to one literal.
    assert!(patch_numeric_literals(src, &[(lits[0].id, "1#x".into())]).is_err());
    assert!(patch_numeric_literals(src, &[(lits[0].id, "1 ".into())]).is_err());
    // Kind mismatch still rejected.
    assert!(patch_numeric_literals(src, &[(lits[0].id, "1.0".into())]).is_err());
}

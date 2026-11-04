//! AST-level API (TODO 1.8): typed subtree enumeration, source-splicing crossover and
//! mutation hooks for the same-DSL GP baseline (C19), and numeric-literal enumeration /
//! patching for ShinkaEvolve's numeric mutation operator.
//!
//! Everything works at the *source* level using the parser's byte spans: an edit is
//! "replace span `[start, end)` of the original text", and every produced source is
//! re-compiled before it is returned — an edit that breaks parsing, typing, or scoping
//! comes back as the corresponding `PsError` (which the GP loop treats as an invalid
//! offspring), never as a silently broken program.
//!
//! **Coverage (C19):** every grammar node is a crossover/mutation site — expressions
//! (typed), simple statements, block statements (`if`/`for`, spliced with
//! indentation-aware reindenting), and whole function definitions. Sites carry
//! `parent`/`depth` so GP operators can navigate the tree. Module-constant numeric
//! literals are enumerable/patchable alongside expression literals.
//!
//! Id scheme: AST nodes use their dense parser ids (`0..node_count`). Constants and
//! functions have no AST node id, so they get stable synthetic ids above `node_count`:
//! constant *i* → `node_count + i`; function *fi* → `node_count + n_consts + fi`.

use crate::ast::*;
use crate::lexer::{lex, Tok};
use crate::typecheck::Program;
use crate::{compile, PsError};

/// A crossover/mutation site: an expression subtree, a statement (simple or block), or
/// a whole function definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subtree {
    /// Node id (dense, parse pre-order — stable for a given source text; functions use
    /// synthetic ids above `node_count`, see module docs).
    pub id: NodeId,
    /// Byte span in the source.
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
    /// 1-based line.
    pub line: u32,
    /// Node kind (e.g. `binary`, `call`, `assign`, `if`, `for`, `func`).
    pub kind: &'static str,
    /// Rendered static type for expressions (`None` for statements and functions).
    pub ty: Option<String>,
    /// Enclosing function name (the function's own name for `func` sites).
    pub func: String,
    /// True for expression nodes, false for statements/functions.
    pub is_expr: bool,
    /// Parent site id (`None` for top-level statements and `func` sites).
    pub parent: Option<NodeId>,
    /// Tree depth (0 for `func` sites, 1 for top-level statements in a body, …).
    pub depth: u32,
}

/// A numeric literal occurrence (expression literal or module-constant literal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericLiteral {
    /// Node id (synthetic `node_count + i` for constant literals, see module docs).
    pub id: NodeId,
    /// Byte span of the literal token(s) in the source.
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
    /// 1-based line.
    pub line: u32,
    /// True if a float literal, false if int.
    pub is_float: bool,
    /// True for a module-constant literal (may carry a leading `-`; expression
    /// literals are non-negative by grammar).
    pub is_const: bool,
    /// Literal text exactly as it appears in the source.
    pub text: String,
}

fn expr_kind(e: &Expr) -> &'static str {
    match e {
        Expr::Int(_) => "int_lit",
        Expr::Float(_) => "float_lit",
        Expr::Str(_) => "str_lit",
        Expr::Bool(_) => "bool_lit",
        Expr::Name(_) => "name",
        Expr::Neg(_) => "neg",
        Expr::Not(_) => "not",
        Expr::Binary(..) => "binary",
        Expr::Compare(..) => "compare",
        Expr::BoolChain(..) => "boolop",
        Expr::Index(..) => "index",
        Expr::Field(..) => "field",
        Expr::Call(..) => "call",
        Expr::ListDisplay(_) => "list",
    }
}

fn stmt_kind(s: &Stmt) -> &'static str {
    match s {
        Stmt::Assign { .. } => "assign",
        Stmt::AugAssign { .. } => "aug_assign",
        Stmt::Append { .. } => "append",
        Stmt::Return(_) => "return",
        Stmt::Pass => "pass",
        Stmt::If { .. } => "if",
        Stmt::For { .. } => "for",
    }
}

struct Walker<'a> {
    prog: &'a Program,
    out: Vec<Subtree>,
    lits: Vec<NumericLiteral>,
    func: String,
    parent: Option<NodeId>,
    depth: u32,
}

impl<'a> Walker<'a> {
    fn expr(&mut self, e: &ExprNode) {
        let ty = self.prog.expr_type(e.id).map(|t| t.render());
        self.out.push(Subtree {
            id: e.id,
            start: e.span.start,
            end: e.span.end,
            line: e.line,
            kind: expr_kind(&e.kind),
            ty,
            func: self.func.clone(),
            is_expr: true,
            parent: self.parent,
            depth: self.depth,
        });
        if matches!(e.kind, Expr::Int(_) | Expr::Float(_)) {
            let text = self.prog.source()[e.span.start as usize..e.span.end as usize].to_string();
            self.lits.push(NumericLiteral {
                id: e.id,
                start: e.span.start,
                end: e.span.end,
                line: e.line,
                is_float: matches!(e.kind, Expr::Float(_)),
                is_const: false,
                text,
            });
        }
        let saved = (self.parent, self.depth);
        self.parent = Some(e.id);
        self.depth += 1;
        match &e.kind {
            Expr::Neg(x) | Expr::Not(x) => self.expr(x),
            Expr::Binary(_, l, r) | Expr::Compare(_, l, r) | Expr::BoolChain(_, l, r) => {
                self.expr(l);
                self.expr(r);
            }
            Expr::Index(b, i) => {
                self.expr(b);
                self.expr(i);
            }
            Expr::Field(b, _) => self.expr(b),
            Expr::Call(_, args) => args.iter().for_each(|a| self.expr(a)),
            Expr::ListDisplay(items) => items.iter().for_each(|i| self.expr(i)),
            _ => {}
        }
        (self.parent, self.depth) = saved;
    }

    fn stmt(&mut self, s: &StmtNode) {
        // Every statement is a splice site (C19): simple statements splice verbatim;
        // block statements (`if`/`for`) splice with indentation-aware reindenting.
        self.out.push(Subtree {
            id: s.id,
            start: s.span.start,
            end: s.span.end,
            line: s.line,
            kind: stmt_kind(&s.kind),
            ty: None,
            func: self.func.clone(),
            is_expr: false,
            parent: self.parent,
            depth: self.depth,
        });
        let saved = (self.parent, self.depth);
        self.parent = Some(s.id);
        self.depth += 1;
        match &s.kind {
            Stmt::Assign { value, .. }
            | Stmt::AugAssign { value, .. }
            | Stmt::Append { value, .. }
            | Stmt::Return(value) => self.expr(value),
            Stmt::Pass => {}
            Stmt::If { arms, orelse } => {
                for (c, b) in arms {
                    self.expr(c);
                    b.iter().for_each(|s| self.stmt(s));
                }
                if let Some(b) = orelse {
                    b.iter().for_each(|s| self.stmt(s));
                }
            }
            Stmt::For { iter, body, .. } => {
                match iter {
                    ForIter::Range(a, b) => {
                        if let Some(a) = a {
                            self.expr(a);
                        }
                        self.expr(b);
                    }
                    ForIter::List(e) => self.expr(e),
                }
                body.iter().for_each(|s| self.stmt(s));
            }
        }
        (self.parent, self.depth) = saved;
    }
}

fn walk(prog: &Program) -> Walker<'_> {
    let mut w = Walker {
        prog,
        out: Vec::new(),
        lits: Vec::new(),
        func: String::new(),
        parent: None,
        depth: 0,
    };
    let module = prog.module();
    // Module-constant numeric literals (synthetic ids node_count + i).
    for (i, c) in module.consts.iter().enumerate() {
        if matches!(c.value, ConstLit::Int(_) | ConstLit::Float(_)) {
            let (s, e) = (c.value_span.start, c.value_span.end);
            w.lits.push(NumericLiteral {
                id: module.node_count + i as u32,
                start: s,
                end: e,
                line: c.line,
                is_float: matches!(c.value, ConstLit::Float(_)),
                is_const: true,
                text: prog.source()[s as usize..e as usize].to_string(),
            });
        }
    }
    let func_id_base = module.node_count + module.consts.len() as u32;
    for (fi, f) in module.funcs.iter().enumerate() {
        // The whole function definition is a crossover site (synthetic id).
        w.out.push(Subtree {
            id: func_id_base + fi as u32,
            start: f.span.start,
            end: f.span.end,
            line: f.line,
            kind: "func",
            ty: None,
            func: f.name.clone(),
            is_expr: false,
            parent: None,
            depth: 0,
        });
        w.func = f.name.clone();
        w.parent = Some(func_id_base + fi as u32);
        w.depth = 1;
        f.body.iter().for_each(|s| w.stmt(s));
        w.parent = None;
        w.depth = 0;
    }
    w
}

/// Enumerate crossover/mutation sites: expressions (typed), simple statements, block
/// statements (`if`/`for`), and whole function definitions.
///
/// Deterministic: ids and order are parse pre-order over the given source text
/// (functions and their synthetic ids included).
pub fn enumerate_subtrees(source: &str) -> Result<Vec<Subtree>, PsError> {
    let prog = compile(source)?;
    Ok(walk(&prog).out)
}

/// Enumerate numeric literals (ShinkaEvolve numeric-patching operator, C19/5.8),
/// including module-constant literals.
pub fn enumerate_numeric_literals(source: &str) -> Result<Vec<NumericLiteral>, PsError> {
    let prog = compile(source)?;
    Ok(walk(&prog).lits)
}

fn find_site(sites: &[Subtree], id: NodeId) -> Result<&Subtree, PsError> {
    sites.iter().find(|s| s.id == id).ok_or(PsError::Type {
        line: 1,
        message: format!("node id {id} is not a valid splice site"),
    })
}

fn splice(source: &str, start: u32, end: u32, replacement: &str) -> String {
    let mut out = String::with_capacity(source.len() + replacement.len());
    out.push_str(&source[..start as usize]);
    out.push_str(replacement);
    out.push_str(&source[end as usize..]);
    out
}

/// Indentation column of the line containing byte offset `at` (count of spaces between
/// the preceding newline and `at`; the grammar allows spaces only).
fn line_indent(source: &str, at: u32) -> usize {
    let bytes = source.as_bytes();
    let mut line_start = at as usize;
    while line_start > 0 && bytes[line_start - 1] != b'\n' {
        line_start -= 1;
    }
    bytes[line_start..at as usize]
        .iter()
        .take_while(|&&b| b == b' ')
        .count()
}

/// Shift a (possibly multi-line) statement text from indentation column `from` to
/// column `to`. The first line is left as-is (the splice target supplies its
/// indentation); every subsequent non-blank line must be indented at least `from`
/// spaces, which are replaced by `to` spaces (relative nesting preserved).
fn reindent(donor: &str, from: usize, to: usize, line: u32) -> Result<String, PsError> {
    let mut out = String::with_capacity(donor.len() + 16);
    for (i, l) in donor.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
            if !l.trim().is_empty() {
                if l.len() < from || !l.as_bytes()[..from].iter().all(|&b| b == b' ') {
                    return Err(PsError::Type {
                        line,
                        message: "donor statement has inconsistent indentation".into(),
                    });
                }
                out.push_str(&" ".repeat(to));
                out.push_str(&l[from..]);
                continue;
            }
        }
        if i > 0 {
            continue; // blank line preserved empty
        }
        out.push_str(l);
    }
    Ok(out)
}

/// Replace subtree `id` in `source` with `new_text` and re-compile.
///
/// Expressions are parenthesized on splice (always syntactically safe) and must be
/// single-line. Statement/function sites accept multi-line `new_text` written at
/// indentation column 0; continuation lines are shifted to the site's column. The
/// result is returned only if it compiles; otherwise the compile error is returned
/// (invalid offspring).
pub fn replace_subtree(source: &str, id: NodeId, new_text: &str) -> Result<String, PsError> {
    let sites = enumerate_subtrees(source)?;
    let site = find_site(&sites, id)?;
    let replacement = if site.is_expr {
        if new_text.contains('\n') {
            return Err(PsError::Type {
                line: site.line,
                message: "expression replacement text must be single-line".into(),
            });
        }
        format!("({new_text})")
    } else {
        let to = line_indent(source, site.start);
        reindent(new_text, 0, to, site.line)?
    };
    let out = splice(source, site.start, site.end, &replacement);
    compile(&out)?;
    Ok(out)
}

/// Typed subtree crossover (C19): replace subtree `id_a` of `src_a` with the text of
/// subtree `id_b` from `src_b`.
///
/// Compatibility: expressions cross with expressions of equal static type; statements
/// (simple or block) cross with statements; whole functions cross with whole
/// functions. Multi-line donors are reindented to the target site's column. The
/// offspring is re-compiled before being returned; scoping violations (e.g. the donor
/// references a variable that does not exist at the target site) surface as the
/// corresponding `PsError`.
pub fn crossover(src_a: &str, src_b: &str, id_a: NodeId, id_b: NodeId) -> Result<String, PsError> {
    let sites_a = enumerate_subtrees(src_a)?;
    let sites_b = enumerate_subtrees(src_b)?;
    let a = find_site(&sites_a, id_a)?;
    let b = find_site(&sites_b, id_b)?;
    if a.is_expr != b.is_expr || ((a.kind == "func") != (b.kind == "func")) {
        return Err(PsError::Type {
            line: a.line,
            message: "crossover sites must be expression×expression, statement×statement, \
                      or function×function"
                .into(),
        });
    }
    if a.is_expr && a.ty != b.ty {
        return Err(PsError::Type {
            line: a.line,
            message: format!(
                "crossover type mismatch: {} vs {}",
                a.ty.as_deref().unwrap_or("?"),
                b.ty.as_deref().unwrap_or("?")
            ),
        });
    }
    let donor = &src_b[b.start as usize..b.end as usize];
    let replacement = if a.is_expr {
        if donor.contains('\n') {
            // Multi-line expressions (bracket continuation) don't splice across lines.
            return Err(PsError::Type {
                line: b.line,
                message: "donor expression spans multiple lines".into(),
            });
        }
        format!("({donor})")
    } else {
        let from = line_indent(src_b, b.start);
        let to = line_indent(src_a, a.start);
        reindent(donor, from, to, b.line)?
    };
    let out = splice(src_a, a.start, a.end, &replacement);
    compile(&out)?;
    Ok(out)
}

/// Validate a numeric-literal replacement through the *actual lexer*: it must lex to
/// exactly one literal token of the required kind (underscore separators and all
/// grammar forms accepted). Constant literals may carry one leading `-`.
fn valid_literal(text: &str, lit: &NumericLiteral) -> bool {
    let body = match (lit.is_const, text.strip_prefix('-')) {
        (true, Some(rest)) => rest,
        (_, Some(_)) => return false, // expression literals are non-negative by grammar
        (_, None) => text,
    };
    // Small literal bodies, but budget the lexer anyway (never an unbounded path).
    let Ok(toks) = lex(body, crate::Limits::default_load().fuel) else {
        return false;
    };
    // Expect: literal token spanning the whole text, Newline (synthesized at EOF),
    // Eof. The full-span requirement rejects smuggled trailing content (comments,
    // whitespace) that would still lex to one literal token.
    if toks.len() != 3 || toks[0].span.start != 0 || toks[0].span.end != body.len() as u32 {
        return false;
    }
    matches!(
        (&toks[0].tok, lit.is_float),
        (Tok::Float(_), true) | (Tok::Int(_), false)
    )
}

/// Apply numeric-literal patches: `(node id, new literal text)` pairs.
///
/// Each replacement must be a literal of the same kind (int → int, float → float),
/// validated through the actual lexer (finite, in range, underscores allowed) —
/// checked before splicing; the patched program is re-compiled before being returned.
pub fn patch_numeric_literals(
    source: &str,
    patches: &[(NodeId, String)],
) -> Result<String, PsError> {
    let lits = enumerate_numeric_literals(source)?;
    let mut resolved: Vec<(&NumericLiteral, &str)> = Vec::with_capacity(patches.len());
    for (id, text) in patches {
        let lit = lits.iter().find(|l| l.id == *id).ok_or(PsError::Type {
            line: 1,
            message: format!("node id {id} is not a numeric literal"),
        })?;
        if !valid_literal(text, lit) {
            return Err(PsError::Type {
                line: lit.line,
                message: format!(
                    "'{text}' is not a valid {}{} literal",
                    if lit.is_const { "" } else { "non-negative " },
                    if lit.is_float { "float" } else { "int" }
                ),
            });
        }
        if resolved.iter().any(|(l, _)| l.id == lit.id) {
            return Err(PsError::Type {
                line: lit.line,
                message: format!("duplicate patch for node id {}", lit.id),
            });
        }
        resolved.push((lit, text));
    }
    // Apply right-to-left so earlier spans stay valid.
    resolved.sort_by(|a, b| b.0.start.cmp(&a.0.start));
    let mut out = source.to_string();
    for (lit, text) in resolved {
        out = splice(&out, lit.start, lit.end, text);
    }
    compile(&out)?;
    Ok(out)
}

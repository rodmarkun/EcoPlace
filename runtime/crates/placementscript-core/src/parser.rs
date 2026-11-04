//! Recursive-descent parser (spec §2). Assigns dense node ids in pre-order.

use crate::ast::*;
use crate::lexer::{lex, Kw, Op, Tok, Token};
use crate::PsError;

/// Maximum expression nesting depth (spec §8.1 structural limits), enforced two ways
/// that must both hold (Round-3 review):
///
/// 1. **Built AST height, compositionally.** Every expression node records its exact
///    subtree height (`1 + max(child heights)`) at construction in `mk`; a node whose
///    height would exceed the limit is rejected *before it is built*. This is the
///    normative bound — it holds across precedence layers and parenthesization, so
///    typechecker/interpreter recursion (and recursive `Drop`) over any accepted tree
///    is exactly bounded on every build target.
/// 2. **Open parser contexts** (`enter_expr`): each parenthesized group, call
///    argument, list element, index expression, `not`, unary `-`, comparison, and
///    each additional operand of an operator/postfix chain charges one level while
///    open. This bounds the *parser's own* native recursion (which the finished-tree
///    height cannot, e.g. `((((x))))` collapses to height 1 but recurses per paren).
pub const MAX_EXPR_DEPTH: u32 = 32;

/// Maximum type nesting depth (spec §8.1): each `list[...]` level counts one.
pub const MAX_TYPE_DEPTH: u32 = 8;

/// Maximum block nesting depth (spec §8.1): each `if`/`elif`/`else`/`for` body level.
pub const MAX_BLOCK_DEPTH: u32 = 8;

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    next_id: NodeId,
    expr_depth: u32,
    type_depth: u32,
    block_depth: u32,
    /// Exact subtree height per expression node id (`heights[id]`), recorded at
    /// construction — see `MAX_EXPR_DEPTH` point 1. Statement ids share the counter
    /// and leave their slot 0.
    heights: Vec<u32>,
}

/// Parse `source` into a `Module` under a load-fuel budget (spec §8.1: one fuel per
/// source byte, charged during lexing).
pub fn parse(source: &str, load_fuel: u64) -> Result<Module, PsError> {
    let toks = lex(source, load_fuel)?;
    let mut p = Parser {
        toks,
        pos: 0,
        next_id: 0,
        expr_depth: 0,
        type_depth: 0,
        block_depth: 0,
        heights: Vec::new(),
    };
    p.parse_module()
}

impl Parser {
    fn cur(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn line(&self) -> u32 {
        self.cur().line
    }

    fn err(&self, message: impl Into<String>) -> PsError {
        PsError::Parse {
            line: self.line(),
            message: message.into(),
        }
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn at(&self, t: &Tok) -> bool {
        &self.cur().tok == t
    }

    fn at_op(&self, op: Op) -> bool {
        matches!(&self.cur().tok, Tok::Op(o) if *o == op)
    }

    fn at_kw(&self, kw: Kw) -> bool {
        matches!(&self.cur().tok, Tok::Kw(k) if *k == kw)
    }

    fn eat_op(&mut self, op: Op) -> bool {
        if self.at_op(op) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_op(&mut self, op: Op, what: &str) -> Result<(), PsError> {
        if self.eat_op(op) {
            Ok(())
        } else {
            Err(self.err(format!("expected {what}")))
        }
    }

    fn expect_newline(&mut self) -> Result<(), PsError> {
        if self.at(&Tok::Newline) {
            self.bump();
            Ok(())
        } else {
            Err(self.err("expected end of line"))
        }
    }

    fn expect_name(&mut self, what: &str) -> Result<(String, Span, u32), PsError> {
        match &self.cur().tok {
            Tok::Name(n) => {
                let n = n.clone();
                let t = self.bump();
                Ok((n, t.span, t.line))
            }
            _ => Err(self.err(format!("expected {what}"))),
        }
    }

    fn id(&mut self) -> NodeId {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    // ---- module ----

    fn parse_module(&mut self) -> Result<Module, PsError> {
        let mut consts = Vec::new();
        let mut funcs = Vec::new();
        loop {
            if self.at(&Tok::Eof) {
                break;
            }
            if self.at(&Tok::Newline) {
                self.bump();
                continue;
            }
            if self.at_kw(Kw::Def) {
                funcs.push(self.parse_func()?);
            } else if matches!(&self.cur().tok, Tok::Name(_)) {
                consts.push(self.parse_const()?);
            } else {
                return Err(self.err("expected 'def' or a module-level constant"));
            }
        }
        Ok(Module {
            consts,
            funcs,
            node_count: self.next_id,
        })
    }

    fn parse_const(&mut self) -> Result<ConstDef, PsError> {
        let (name, _, line) = self.expect_name("constant name")?;
        self.expect_op(Op::Colon, "':' after constant name")?;
        let ty = self.parse_type()?;
        self.expect_op(Op::Assign, "'=' in constant definition")?;
        let value_start = self.cur().span.start;
        let neg = self.eat_op(Op::Minus);
        let value = match self.bump().tok {
            Tok::Int(v) => {
                if neg {
                    ConstLit::Int(-v)
                } else {
                    ConstLit::Int(v)
                }
            }
            Tok::Float(v) => ConstLit::Float(if neg { -v } else { v }),
            Tok::Str(s) if !neg => ConstLit::Str(s),
            Tok::Kw(Kw::True) if !neg => ConstLit::Bool(true),
            Tok::Kw(Kw::False) if !neg => ConstLit::Bool(false),
            _ => return Err(self.err("module constants must be literals")),
        };
        let value_span = Span {
            start: value_start,
            end: self.prev_end(),
        };
        self.expect_newline()?;
        Ok(ConstDef {
            name,
            ty,
            value,
            value_span,
            line,
        })
    }

    fn parse_func(&mut self) -> Result<FuncDef, PsError> {
        let line = self.line();
        let start = self.cur().span.start;
        self.bump(); // def
        let (name, _, _) = self.expect_name("function name")?;
        self.expect_op(Op::LParen, "'('")?;
        let mut params = Vec::new();
        if !self.at_op(Op::RParen) {
            loop {
                let (pname, _, _) = self.expect_name("parameter name")?;
                self.expect_op(Op::Colon, "':' after parameter name (types are required)")?;
                let ty = self.parse_type()?;
                params.push((pname, ty));
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
        }
        self.expect_op(Op::RParen, "')'")?;
        self.expect_op(Op::Arrow, "'->' (return types are required)")?;
        let ret = self.parse_type()?;
        self.expect_op(Op::Colon, "':'")?;
        let body = self.parse_block()?;
        // Span ends at the last body statement (nested statements included: a block
        // statement's span already extends through its innermost children).
        let end = body.last().map(|s| s.span.end).unwrap_or(self.prev_end());
        Ok(FuncDef {
            name,
            params,
            ret,
            body,
            span: Span { start, end },
            line,
        })
    }

    fn parse_type(&mut self) -> Result<Type, PsError> {
        let (name, _, _) = self.expect_name("a type")?;
        match name.as_str() {
            "int" => Ok(Type::Int),
            "float" => Ok(Type::Float),
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            "list" => {
                self.type_depth += 1;
                if self.type_depth > MAX_TYPE_DEPTH {
                    return Err(self.err(format!(
                        "type nesting too deep (max {MAX_TYPE_DEPTH} levels; spec §8.1)"
                    )));
                }
                self.expect_op(Op::LBracket, "'[' after 'list'")?;
                let inner = self.parse_type();
                self.type_depth -= 1;
                let inner = inner?;
                self.expect_op(Op::RBracket, "']'")?;
                Ok(Type::List(Box::new(inner)))
            }
            other => match RecordType::by_name(other) {
                Some(rt) => Ok(Type::Record(rt)),
                None => Err(self.err(format!("unknown type '{other}'"))),
            },
        }
    }

    // ---- statements ----

    fn parse_block(&mut self) -> Result<Vec<StmtNode>, PsError> {
        self.block_depth += 1;
        if self.block_depth > MAX_BLOCK_DEPTH {
            return Err(self.err(format!(
                "block nesting too deep (max {MAX_BLOCK_DEPTH} levels; spec §8.1)"
            )));
        }
        self.expect_newline()?;
        if !self.at(&Tok::Indent) {
            return Err(self.err("expected an indented block"));
        }
        self.bump();
        let mut stmts = Vec::new();
        while !self.at(&Tok::Dedent) {
            if self.at(&Tok::Eof) {
                return Err(self.err("unexpected end of input in block"));
            }
            stmts.push(self.parse_stmt()?);
        }
        self.bump(); // dedent
        self.block_depth -= 1;
        if stmts.is_empty() {
            return Err(self.err("empty block"));
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<StmtNode, PsError> {
        let line = self.line();
        let start = self.cur().span.start;
        let id = self.id();
        if self.at_kw(Kw::If) {
            return self.parse_if(id, start, line);
        }
        if self.at_kw(Kw::For) {
            return self.parse_for(id, start, line);
        }
        let kind = if self.at_kw(Kw::Return) {
            self.bump();
            let e = self.parse_expr()?;
            Stmt::Return(e)
        } else if self.at_kw(Kw::Pass) {
            self.bump();
            Stmt::Pass
        } else if matches!(&self.cur().tok, Tok::Name(_)) {
            let (name, _, _) = self.expect_name("name")?;
            if self.at_op(Op::Dot) {
                self.bump();
                let (m, _, _) = self.expect_name("method name")?;
                if m != "append" {
                    return Err(self.err(format!(
                        "unknown statement method '.{m}(...)' (only 'append')"
                    )));
                }
                self.expect_op(Op::LParen, "'('")?;
                let e = self.parse_expr()?;
                self.expect_op(Op::RParen, "')'")?;
                Stmt::Append { name, value: e }
            } else if self.at_op(Op::Colon) {
                self.bump();
                let ty = self.parse_type()?;
                self.expect_op(Op::Assign, "'=' after annotation")?;
                let e = self.parse_expr()?;
                Stmt::Assign {
                    name,
                    ann: Some(ty),
                    value: e,
                }
            } else if self.at_op(Op::Assign) {
                self.bump();
                let e = self.parse_expr()?;
                Stmt::Assign {
                    name,
                    ann: None,
                    value: e,
                }
            } else {
                let op = match &self.cur().tok {
                    Tok::Op(Op::PlusEq) => BinOp::Add,
                    Tok::Op(Op::MinusEq) => BinOp::Sub,
                    Tok::Op(Op::StarEq) => BinOp::Mul,
                    Tok::Op(Op::SlashEq) => BinOp::Div,
                    Tok::Op(Op::SlashSlashEq) => BinOp::FloorDiv,
                    Tok::Op(Op::PercentEq) => BinOp::Mod,
                    _ => {
                        return Err(self.err(
                            "expected assignment, augmented assignment, or '.append(...)' \
                             (bare expressions are not statements)",
                        ))
                    }
                };
                self.bump();
                let e = self.parse_expr()?;
                Stmt::AugAssign { name, op, value: e }
            }
        } else {
            return Err(self.err("expected a statement"));
        };
        let end = self.prev_end();
        self.expect_newline()?;
        Ok(StmtNode {
            id,
            span: Span { start, end },
            line,
            kind,
        })
    }

    fn prev_end(&self) -> u32 {
        self.toks[self.pos.saturating_sub(1)].span.end
    }

    fn parse_if(&mut self, id: NodeId, start: u32, line: u32) -> Result<StmtNode, PsError> {
        self.bump(); // if
        let mut arms = Vec::new();
        let cond = self.parse_expr()?;
        self.expect_op(Op::Colon, "':'")?;
        let block = self.parse_block()?;
        arms.push((cond, block));
        let mut orelse = None;
        loop {
            if self.at_kw(Kw::Elif) {
                self.bump();
                let c = self.parse_expr()?;
                self.expect_op(Op::Colon, "':'")?;
                let b = self.parse_block()?;
                arms.push((c, b));
            } else if self.at_kw(Kw::Else) {
                self.bump();
                self.expect_op(Op::Colon, "':'")?;
                orelse = Some(self.parse_block()?);
                break;
            } else {
                break;
            }
        }
        // Span ends at the last statement of the last block — *not* `prev_end()`,
        // which is the Dedent token and would drag the span through the next
        // line's indentation (breaking source splicing at this site).
        let end = orelse
            .as_deref()
            .and_then(<[StmtNode]>::last)
            .or_else(|| arms.last().and_then(|(_, b)| b.last()))
            .map(|s| s.span.end)
            .unwrap_or_else(|| self.prev_end());
        Ok(StmtNode {
            id,
            span: Span { start, end },
            line,
            kind: Stmt::If { arms, orelse },
        })
    }

    fn parse_for(&mut self, id: NodeId, start: u32, line: u32) -> Result<StmtNode, PsError> {
        self.bump(); // for
        let (var, _, _) = self.expect_name("loop variable")?;
        if !self.at_kw(Kw::In) {
            return Err(self.err("expected 'in'"));
        }
        self.bump();
        let iter = if self.at_kw(Kw::Range) {
            self.bump();
            self.expect_op(Op::LParen, "'(' after 'range'")?;
            let first = self.parse_expr()?;
            let iter = if self.eat_op(Op::Comma) {
                let second = self.parse_expr()?;
                ForIter::Range(Some(Box::new(first)), Box::new(second))
            } else {
                ForIter::Range(None, Box::new(first))
            };
            self.expect_op(Op::RParen, "')'")?;
            iter
        } else {
            ForIter::List(Box::new(self.parse_expr()?))
        };
        self.expect_op(Op::Colon, "':'")?;
        let body = self.parse_block()?;
        // See parse_if: end at the last body statement, not the Dedent token.
        let end = body
            .last()
            .map(|s| s.span.end)
            .unwrap_or_else(|| self.prev_end());
        Ok(StmtNode {
            id,
            span: Span { start, end },
            line,
            kind: Stmt::For { var, iter, body },
        })
    }

    // ---- expressions ----

    fn parse_expr(&mut self) -> Result<ExprNode, PsError> {
        self.enter_expr()?;
        let e = self.parse_or();
        self.expr_depth -= 1;
        e
    }

    /// Charge one expression nesting level (spec §8.1 structural limits). Every
    /// recursion-deepening path in the expression grammar passes through here:
    /// `parse_expr` re-entry (parens, call args, list elements, index expressions)
    /// and the self-recursive `not` / unary `-` arms.
    fn enter_expr(&mut self) -> Result<(), PsError> {
        self.expr_depth += 1;
        if self.expr_depth > MAX_EXPR_DEPTH {
            return Err(self.err(format!(
                "expression nesting too deep (max {MAX_EXPR_DEPTH} levels; spec §8.1)"
            )));
        }
        Ok(())
    }

    /// Recorded height of an already-built child expression.
    fn h(&self, e: &ExprNode) -> u32 {
        self.heights[e.id as usize]
    }

    /// Build an expression node, computing its exact subtree height compositionally
    /// (`1 + max(child heights)`) and rejecting it *before construction* if the
    /// height would exceed `MAX_EXPR_DEPTH` — the open-context counter alone does not
    /// bound finished-tree height across precedence layers (Round-3 review).
    fn mk(&mut self, id: NodeId, start: u32, line: u32, kind: Expr) -> Result<ExprNode, PsError> {
        let h = 1 + match &kind {
            Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Name(_) => 0,
            Expr::Neg(e) | Expr::Not(e) | Expr::Field(e, _) => self.h(e),
            Expr::Binary(_, l, r)
            | Expr::Compare(_, l, r)
            | Expr::BoolChain(_, l, r)
            | Expr::Index(l, r) => self.h(l).max(self.h(r)),
            Expr::Call(_, args) => args.iter().map(|a| self.h(a)).max().unwrap_or(0),
            Expr::ListDisplay(items) => items.iter().map(|a| self.h(a)).max().unwrap_or(0),
        };
        if h > MAX_EXPR_DEPTH {
            return Err(PsError::Parse {
                line,
                message: format!(
                    "expression nesting too deep (expression tree height {h} exceeds \
                     max {MAX_EXPR_DEPTH}; spec §8.1)"
                ),
            });
        }
        if self.heights.len() <= id as usize {
            self.heights.resize(id as usize + 1, 0);
        }
        self.heights[id as usize] = h;
        Ok(ExprNode {
            id,
            span: Span {
                start,
                end: self.prev_end(),
            },
            line,
            kind,
        })
    }

    fn parse_or(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let mut lhs = self.parse_and()?;
        let mut charged = 0;
        while self.at_kw(Kw::Or) {
            // Each chained operand wraps another BoolChain node around `lhs`: charge
            // one nesting level per operator so the built tree height stays bounded
            // (spec §8.1) — a long flat-looking chain is a deep left-leaning tree.
            // (On error the whole parse aborts, so no depth rebalancing is needed.)
            self.enter_expr()?;
            charged += 1;
            let id = self.id();
            self.bump();
            let rhs = self.parse_and()?;
            lhs = self.mk(
                id,
                start,
                line,
                Expr::BoolChain(BoolOp::Or, Box::new(lhs), Box::new(rhs)),
            )?;
        }
        self.expr_depth -= charged;
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let mut lhs = self.parse_not()?;
        let mut charged = 0;
        while self.at_kw(Kw::And) {
            self.enter_expr()?; // see parse_or: one level per chained operand
            charged += 1;
            let id = self.id();
            self.bump();
            let rhs = self.parse_not()?;
            lhs = self.mk(
                id,
                start,
                line,
                Expr::BoolChain(BoolOp::And, Box::new(lhs), Box::new(rhs)),
            )?;
        }
        self.expr_depth -= charged;
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<ExprNode, PsError> {
        if self.at_kw(Kw::Not) {
            let start = self.cur().span.start;
            let line = self.line();
            let id = self.id();
            self.bump();
            self.enter_expr()?;
            let e = self.parse_not();
            self.expr_depth -= 1;
            return self.mk(id, start, line, Expr::Not(Box::new(e?)));
        }
        self.parse_comparison()
    }

    fn cmp_op(&self) -> Option<CmpOp> {
        match &self.cur().tok {
            Tok::Op(Op::Lt) => Some(CmpOp::Lt),
            Tok::Op(Op::Le) => Some(CmpOp::Le),
            Tok::Op(Op::Gt) => Some(CmpOp::Gt),
            Tok::Op(Op::Ge) => Some(CmpOp::Ge),
            Tok::Op(Op::EqEq) => Some(CmpOp::Eq),
            Tok::Op(Op::Ne) => Some(CmpOp::Ne),
            _ => None,
        }
    }

    fn parse_comparison(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let lhs = self.parse_arith()?;
        if let Some(op) = self.cmp_op() {
            self.enter_expr()?; // the Compare node adds one level of tree height
            let id = self.id();
            self.bump();
            let rhs = self.parse_arith()?;
            if self.cmp_op().is_some() {
                return Err(self.err("chained comparisons are not supported"));
            }
            self.expr_depth -= 1;
            return self.mk(
                id,
                start,
                line,
                Expr::Compare(op, Box::new(lhs), Box::new(rhs)),
            );
        }
        Ok(lhs)
    }

    fn parse_arith(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let mut lhs = self.parse_term()?;
        let mut charged = 0;
        loop {
            let op = match &self.cur().tok {
                Tok::Op(Op::Plus) => BinOp::Add,
                Tok::Op(Op::Minus) => BinOp::Sub,
                _ => break,
            };
            self.enter_expr()?; // see parse_or: one level per chained operand
            charged += 1;
            let id = self.id();
            self.bump();
            let rhs = self.parse_term()?;
            lhs = self.mk(
                id,
                start,
                line,
                Expr::Binary(op, Box::new(lhs), Box::new(rhs)),
            )?;
        }
        self.expr_depth -= charged;
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let mut lhs = self.parse_unary()?;
        let mut charged = 0;
        loop {
            let op = match &self.cur().tok {
                Tok::Op(Op::Star) => BinOp::Mul,
                Tok::Op(Op::Slash) => BinOp::Div,
                Tok::Op(Op::SlashSlash) => BinOp::FloorDiv,
                Tok::Op(Op::Percent) => BinOp::Mod,
                _ => break,
            };
            self.enter_expr()?; // see parse_or: one level per chained operand
            charged += 1;
            let id = self.id();
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = self.mk(
                id,
                start,
                line,
                Expr::Binary(op, Box::new(lhs), Box::new(rhs)),
            )?;
        }
        self.expr_depth -= charged;
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<ExprNode, PsError> {
        if self.at_op(Op::Minus) {
            let start = self.cur().span.start;
            let line = self.line();
            let id = self.id();
            self.bump();
            self.enter_expr()?;
            let e = self.parse_unary();
            self.expr_depth -= 1;
            return self.mk(id, start, line, Expr::Neg(Box::new(e?)));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        let mut e = self.parse_atom()?;
        let mut charged = 0;
        loop {
            if self.at_op(Op::Dot) {
                // Each postfix suffix wraps another Field/Index node around `e`: charge
                // one nesting level per suffix so built-tree height stays bounded (spec
                // §8.1) — `x[0][0]...` is a deep left spine exactly like operator
                // chains (post-Round-2 self-review of R1.1).
                self.enter_expr()?;
                charged += 1;
                let id = self.id();
                self.bump();
                let (field, _, _) = self.expect_name("field name")?;
                if self.at_op(Op::LParen) {
                    return Err(self.err(format!(
                        "method calls are not expressions ('.{field}(...)'); \
                         only 'xs.append(v)' as a statement"
                    )));
                }
                e = self.mk(id, start, line, Expr::Field(Box::new(e), field))?;
            } else if self.at_op(Op::LBracket) {
                self.enter_expr()?; // see above: one level per postfix suffix
                charged += 1;
                let id = self.id();
                self.bump();
                let idx = self.parse_expr()?;
                self.expect_op(Op::RBracket, "']'")?;
                e = self.mk(id, start, line, Expr::Index(Box::new(e), Box::new(idx)))?;
            } else {
                break;
            }
        }
        self.expr_depth -= charged;
        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<ExprNode, PsError> {
        let start = self.cur().span.start;
        let line = self.line();
        match self.cur().tok.clone() {
            Tok::Int(v) => {
                let id = self.id();
                self.bump();
                self.mk(id, start, line, Expr::Int(v))
            }
            Tok::Float(v) => {
                let id = self.id();
                self.bump();
                self.mk(id, start, line, Expr::Float(v))
            }
            Tok::Str(s) => {
                let id = self.id();
                self.bump();
                self.mk(id, start, line, Expr::Str(s))
            }
            Tok::Kw(Kw::True) => {
                let id = self.id();
                self.bump();
                self.mk(id, start, line, Expr::Bool(true))
            }
            Tok::Kw(Kw::False) => {
                let id = self.id();
                self.bump();
                self.mk(id, start, line, Expr::Bool(false))
            }
            Tok::Name(n) => {
                let id = self.id();
                self.bump();
                if self.at_op(Op::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if !self.at_op(Op::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.eat_op(Op::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect_op(Op::RParen, "')'")?;
                    self.mk(id, start, line, Expr::Call(n, args))
                } else {
                    self.mk(id, start, line, Expr::Name(n))
                }
            }
            Tok::Op(Op::LParen) => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect_op(Op::RParen, "')'")?;
                Ok(e)
            }
            Tok::Op(Op::LBracket) => {
                let id = self.id();
                self.bump();
                let mut items = Vec::new();
                if !self.at_op(Op::RBracket) {
                    loop {
                        items.push(self.parse_expr()?);
                        if !self.eat_op(Op::Comma) {
                            break;
                        }
                    }
                }
                self.expect_op(Op::RBracket, "']'")?;
                self.mk(id, start, line, Expr::ListDisplay(items))
            }
            Tok::Kw(Kw::Range) => Err(self.err("'range' is only valid as a for-loop iterable")),
            _ => Err(self.err("expected an expression")),
        }
    }
}

//! Indentation-aware lexer (spec §1).
//!
//! Produces a flat token stream with explicit `Newline`/`Indent`/`Dedent` tokens.
//! Numeric-literal finiteness/range checks happen here: NaN/inf are unrepresentable
//! downstream by construction (C18).

use crate::ast::Span;
use crate::PsError;

/// Keywords (spec §1). Type and record names are reserved but lex as `Name`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Kw {
    Def,
    Return,
    If,
    Elif,
    Else,
    For,
    In,
    Range,
    And,
    Or,
    Not,
    True,
    False,
    Pass,
}

/// Punctuation and operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Op {
    LParen,
    RParen,
    LBracket,
    RBracket,
    Colon,
    Comma,
    Dot,
    Plus,
    Minus,
    Star,
    Slash,
    SlashSlash,
    Percent,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    SlashSlashEq,
    PercentEq,
    Arrow,
}

/// Token kinds.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// Integer literal (fits i64, non-negative).
    Int(i64),
    /// Float literal (finite, non-negative).
    Float(f64),
    /// String literal (decoded).
    Str(String),
    /// Identifier (including reserved type/record/builtin names).
    Name(String),
    /// Keyword.
    Kw(Kw),
    /// Operator/punctuation.
    Op(Op),
    /// Logical end of line.
    Newline,
    /// Indentation increase.
    Indent,
    /// Indentation decrease.
    Dedent,
    /// End of input.
    Eof,
}

/// A token with location.
#[derive(Debug, Clone)]
pub struct Token {
    /// Kind.
    pub tok: Tok,
    /// Byte span.
    pub span: Span,
    /// 1-based line.
    pub line: u32,
}

fn kw_of(s: &str) -> Option<Kw> {
    Some(match s {
        "def" => Kw::Def,
        "return" => Kw::Return,
        "if" => Kw::If,
        "elif" => Kw::Elif,
        "else" => Kw::Else,
        "for" => Kw::For,
        "in" => Kw::In,
        "range" => Kw::Range,
        "and" => Kw::And,
        "or" => Kw::Or,
        "not" => Kw::Not,
        "True" => Kw::True,
        "False" => Kw::False,
        "pass" => Kw::Pass,
        _ => return None,
    })
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    paren_depth: u32,
    indents: Vec<usize>,
    out: Vec<Token>,
    /// Load-fuel budget in *source bytes* (spec §8.1: load fuel is charged one per
    /// source byte, checked during lexing before the per-AST-node/per-constant
    /// charges). Checked before every token, so lexing allocates O(budget) even on a
    /// pathologically token-dense input — an over-budget source is a structured
    /// `fuel` outcome, never compiler memory exhaustion (Round-3 review).
    fuel: u64,
}

/// Tokenize `source` under a load-fuel budget of `load_fuel` source bytes.
/// Grammar errors are `PsError::Parse`; budget exhaustion is `PsError::FuelExhausted`.
pub fn lex(source: &str, load_fuel: u64) -> Result<Vec<Token>, PsError> {
    // Reject over-budget sources before scanning a single byte (Round-4 review: the
    // between-token checks alone let one long token — identifier, number, string,
    // comment — scan far past the budget). Outcome-equivalent to the final
    // accounting check (total load fuel = source bytes + AST nodes + constants, so
    // `len > fuel` can never succeed anyway); this makes lexer work and allocation
    // O(budget) unconditionally.
    if source.len() as u64 > load_fuel {
        return Err(PsError::FuelExhausted);
    }
    let mut lx = Lexer {
        src: source.as_bytes(),
        pos: 0,
        line: 1,
        paren_depth: 0,
        indents: vec![0],
        out: Vec::new(),
        fuel: load_fuel,
    };
    lx.run()?;
    Ok(lx.out)
}

impl<'a> Lexer<'a> {
    fn err(&self, message: impl Into<String>) -> PsError {
        PsError::Parse {
            line: self.line,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn push(&mut self, tok: Tok, start: usize) {
        self.out.push(Token {
            tok,
            span: Span {
                start: start as u32,
                end: self.pos as u32,
            },
            line: self.line,
        });
    }

    /// Charge scanned source bytes against the load-fuel budget (one fuel per byte,
    /// spec §8.1). Called before every token, so between checks the lexer can
    /// overshoot by at most one token — allocation stays O(budget + one token).
    fn charge(&self) -> Result<(), PsError> {
        if self.pos as u64 > self.fuel {
            return Err(PsError::FuelExhausted);
        }
        Ok(())
    }

    fn run(&mut self) -> Result<(), PsError> {
        loop {
            self.charge()?;
            // At a physical line start outside brackets: handle indentation.
            if self.paren_depth == 0 && !self.handle_line_start()? {
                break; // EOF
            }
            // Lex tokens until end of logical line (or EOF).
            let more = self.lex_line()?;
            if !more {
                break;
            }
        }
        // Close the file: final NEWLINE is emitted by lex_line at EOF when needed.
        while self.indents.len() > 1 {
            self.indents.pop();
            self.push(Tok::Dedent, self.pos);
        }
        self.push(Tok::Eof, self.pos);
        Ok(())
    }

    /// Skip a comment body (`#` to end of line). Tabs and carriage returns are parse
    /// errors *everywhere* outside string literals (spec §1) — including comments.
    fn skip_comment(&mut self) -> Result<(), PsError> {
        while let Some(c) = self.peek() {
            match c {
                b'\n' => break,
                b'\t' => return Err(self.err("tab character (spaces only)")),
                b'\r' => return Err(self.err("carriage return (LF line endings only)")),
                _ => self.pos += 1,
            }
        }
        Ok(())
    }

    /// Skip blank/comment lines and emit Indent/Dedent for the first contentful line.
    /// Returns false at EOF.
    fn handle_line_start(&mut self) -> Result<bool, PsError> {
        loop {
            let mut col = 0usize;
            loop {
                match self.peek() {
                    Some(b' ') => {
                        self.pos += 1;
                        col += 1;
                    }
                    Some(b'\t') => return Err(self.err("tab character (spaces only)")),
                    _ => break,
                }
            }
            match self.peek() {
                None => return Ok(false),
                Some(b'\n') => {
                    self.pos += 1;
                    self.line += 1;
                    continue; // blank line
                }
                Some(b'\r') => {
                    return Err(self.err("carriage return (LF line endings only)"));
                }
                Some(b'#') => {
                    self.skip_comment()?;
                    continue; // comment-only line; loop consumes the newline next pass
                }
                Some(_) => {
                    let cur = *self.indents.last().unwrap();
                    match col.cmp(&cur) {
                        std::cmp::Ordering::Greater => {
                            self.indents.push(col);
                            self.push(Tok::Indent, self.pos);
                        }
                        std::cmp::Ordering::Less => {
                            while *self.indents.last().unwrap() > col {
                                self.indents.pop();
                                self.push(Tok::Dedent, self.pos);
                            }
                            if *self.indents.last().unwrap() != col {
                                return Err(self.err("unindent does not match any outer level"));
                            }
                        }
                        std::cmp::Ordering::Equal => {}
                    }
                    return Ok(true);
                }
            }
        }
    }

    /// Lex tokens until a logical line ends (NEWLINE emitted) or EOF.
    /// Returns false at EOF.
    fn lex_line(&mut self) -> Result<bool, PsError> {
        loop {
            self.charge()?;
            let Some(c) = self.peek() else {
                // EOF mid-line: emit the closing NEWLINE.
                if self.paren_depth > 0 {
                    return Err(self.err("unclosed bracket at end of input"));
                }
                self.push(Tok::Newline, self.pos);
                return Ok(false);
            };
            match c {
                b' ' => {
                    self.pos += 1;
                }
                b'\t' => return Err(self.err("tab character (spaces only)")),
                b'\r' => return Err(self.err("carriage return (LF line endings only)")),
                b'#' => self.skip_comment()?,
                b'\n' => {
                    self.pos += 1;
                    if self.paren_depth == 0 {
                        let start = self.pos - 1;
                        self.push(Tok::Newline, start);
                        self.line += 1;
                        return Ok(true);
                    }
                    self.line += 1; // implicit continuation inside brackets
                }
                b'0'..=b'9' => self.lex_number()?,
                b'.' => {
                    if matches!(self.peek2(), Some(b'0'..=b'9')) {
                        self.lex_number()?;
                    } else {
                        let s = self.pos;
                        self.pos += 1;
                        self.push(Tok::Op(Op::Dot), s);
                    }
                }
                b'"' | b'\'' => self.lex_string()?,
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.lex_name(),
                _ => self.lex_op()?,
            }
        }
    }

    fn lex_name(&mut self) {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        let tok = match kw_of(text) {
            Some(k) => Tok::Kw(k),
            None => Tok::Name(text.to_string()),
        };
        self.push(tok, start);
    }

    fn lex_number(&mut self) -> Result<(), PsError> {
        let start = self.pos;
        let digits = |lx: &mut Self| -> Result<bool, PsError> {
            // consume [0-9_]+ validating underscore placement; returns whether any digit seen
            let mut any = false;
            let mut prev_us = false;
            let mut first = true;
            while let Some(c) = lx.peek() {
                match c {
                    b'0'..=b'9' => {
                        any = true;
                        prev_us = false;
                        first = false;
                        lx.pos += 1;
                    }
                    b'_' => {
                        if first || prev_us {
                            return Err(lx.err("misplaced underscore in numeric literal"));
                        }
                        prev_us = true;
                        lx.pos += 1;
                    }
                    _ => break,
                }
            }
            if prev_us {
                return Err(lx.err("misplaced underscore in numeric literal"));
            }
            Ok(any)
        };

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            // .5 form
            is_float = true;
            self.pos += 1;
            if !digits(self)? {
                return Err(self.err("malformed float literal"));
            }
        } else {
            digits(self)?;
            if self.peek() == Some(b'.') {
                is_float = true;
                self.pos += 1;
                digits(self)?; // optional fraction ("1." is valid)
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !digits(self)? {
                return Err(self.err("malformed float exponent"));
            }
        }
        // A literal immediately followed by an identifier character is malformed (e.g. 1x).
        if let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                return Err(self.err("malformed numeric literal"));
            }
        }
        let text: String = std::str::from_utf8(&self.src[start..self.pos])
            .unwrap()
            .chars()
            .filter(|c| *c != '_')
            .collect();
        if is_float {
            let v: f64 = text
                .parse()
                .map_err(|_| self.err("malformed float literal"))?;
            if !v.is_finite() {
                return Err(self.err("float literal is not finite (NaN/inf are rejected)"));
            }
            self.push(Tok::Float(v), start);
        } else {
            let v: i64 = text
                .parse()
                .map_err(|_| self.err("integer literal out of i64 range"))?;
            self.push(Tok::Int(v), start);
        }
        Ok(())
    }

    fn lex_string(&mut self) -> Result<(), PsError> {
        let start = self.pos;
        let quote = self.bump().unwrap();
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string literal")),
                Some(b'\n') => return Err(self.err("newline in string literal")),
                Some(c) if c == quote => break,
                Some(b'\\') => match self.bump() {
                    Some(b'\\') => s.push('\\'),
                    Some(b'"') => s.push('"'),
                    Some(b'\'') => s.push('\''),
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    _ => return Err(self.err("unsupported escape sequence")),
                },
                Some(c) => {
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        // The source arrived as `&str`, so its bytes are valid UTF-8
                        // and this position is a character boundary (ASCII paths
                        // advance one whole character at a time). Decode this one
                        // character in O(1) from its leading byte — re-validating the
                        // entire remaining suffix per character made lexing quadratic
                        // in the string length (Round-4 review).
                        let len = match c {
                            0xC0..=0xDF => 2,
                            0xE0..=0xEF => 3,
                            _ => 4,
                        };
                        let seq = &self.src[self.pos - 1..self.pos - 1 + len];
                        let ch = std::str::from_utf8(seq)
                            .expect("source is &str: char sequences are valid UTF-8")
                            .chars()
                            .next()
                            .unwrap();
                        s.push(ch);
                        self.pos += len - 1;
                    }
                }
            }
        }
        self.push(Tok::Str(s), start);
        Ok(())
    }

    fn lex_op(&mut self) -> Result<(), PsError> {
        let start = self.pos;
        let c = self.bump().unwrap();
        let eat = |lx: &mut Self, c: u8| -> bool {
            if lx.peek() == Some(c) {
                lx.pos += 1;
                true
            } else {
                false
            }
        };
        let op = match c {
            b'(' => {
                self.paren_depth += 1;
                Op::LParen
            }
            b')' => {
                if self.paren_depth == 0 {
                    return Err(self.err("unmatched ')'"));
                }
                self.paren_depth -= 1;
                Op::RParen
            }
            b'[' => {
                self.paren_depth += 1;
                Op::LBracket
            }
            b']' => {
                if self.paren_depth == 0 {
                    return Err(self.err("unmatched ']'"));
                }
                self.paren_depth -= 1;
                Op::RBracket
            }
            b':' => Op::Colon,
            b',' => Op::Comma,
            b'+' => {
                if eat(self, b'=') {
                    Op::PlusEq
                } else {
                    Op::Plus
                }
            }
            b'-' => {
                if eat(self, b'>') {
                    Op::Arrow
                } else if eat(self, b'=') {
                    Op::MinusEq
                } else {
                    Op::Minus
                }
            }
            b'*' => {
                if eat(self, b'*') {
                    return Err(self.err("'**' is not supported"));
                } else if eat(self, b'=') {
                    Op::StarEq
                } else {
                    Op::Star
                }
            }
            b'/' => {
                if eat(self, b'/') {
                    if eat(self, b'=') {
                        Op::SlashSlashEq
                    } else {
                        Op::SlashSlash
                    }
                } else if eat(self, b'=') {
                    Op::SlashEq
                } else {
                    Op::Slash
                }
            }
            b'%' => {
                if eat(self, b'=') {
                    Op::PercentEq
                } else {
                    Op::Percent
                }
            }
            b'<' => {
                if eat(self, b'=') {
                    Op::Le
                } else {
                    Op::Lt
                }
            }
            b'>' => {
                if eat(self, b'=') {
                    Op::Ge
                } else {
                    Op::Gt
                }
            }
            b'=' => {
                if eat(self, b'=') {
                    Op::EqEq
                } else {
                    Op::Assign
                }
            }
            b'!' => {
                if eat(self, b'=') {
                    Op::Ne
                } else {
                    return Err(self.err("unexpected '!'"));
                }
            }
            other => {
                return Err(self.err(format!("unexpected character {:?}", char::from(other))));
            }
        };
        self.push(Tok::Op(op), start);
        Ok(())
    }
}

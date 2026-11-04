//! Tree-walking interpreter with fuel metering and memory accounting (spec §6–§8).
//!
//! Fuel is charged *before* each step; memory is cumulative allocation (spec §8.2).
//! Every failure maps to exactly one `PsError` class — the taxonomy is evolution
//! feedback (TODO 5.4) and must stay distinguishable.

use crate::ast::*;
use crate::typecheck::{Builtin, Program, Res};
use crate::value::{RecordVal, Value};
use crate::{Limits, PsError};
use std::rc::Rc;

/// Result of a successful call: the value plus resource-use diagnostics.
#[derive(Debug, Clone)]
pub struct CallOutcome {
    /// The returned value.
    pub value: Value,
    /// Fuel consumed by the invocation.
    pub fuel_used: u64,
    /// Bytes of (cumulative) allocation charged to the invocation.
    pub mem_used: u64,
}

/// Errors at the host boundary — a host bug or bad embedding input, never the policy's
/// fault (spec §9). Kept out of `PsError` so evolution feedback can't misattribute them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// The named function does not exist in the program.
    UnknownFunction(String),
    /// Arguments do not match the function signature (arity/type/finiteness).
    BadArguments(String),
}

impl core::fmt::Display for HostError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HostError::UnknownFunction(n) => write!(f, "unknown function '{n}'"),
            HostError::BadArguments(m) => write!(f, "bad arguments: {m}"),
        }
    }
}

impl std::error::Error for HostError {}

/// Either a language-level error (the policy's fault) or a host-boundary error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// Language-level failure (spec §9 taxonomy).
    Ps(PsError),
    /// Host-boundary failure.
    Host(HostError),
}

impl core::fmt::Display for CallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CallError::Ps(e) => write!(f, "{e}"),
            CallError::Host(e) => write!(f, "host error: {e}"),
        }
    }
}

impl std::error::Error for CallError {}

impl From<PsError> for CallError {
    fn from(e: PsError) -> Self {
        CallError::Ps(e)
    }
}

/// Call `func` in `prog` with `args` under `limits`.
///
/// Arguments are deep-validated against the function signature (a mismatch is a
/// `CallError::Host`, not a policy failure).
pub fn call(
    prog: &Program,
    func: &str,
    args: &[Value],
    limits: Limits,
) -> Result<CallOutcome, CallError> {
    let fi = resolve(prog, func, args)?;
    let f = &prog.module.funcs[fi];
    for (a, (pname, pty)) in args.iter().zip(f.params.iter()) {
        if !a.conforms(pty) {
            return Err(CallError::Host(HostError::BadArguments(format!(
                "argument '{pname}' of '{func}' must be {} (with finite floats), got {}",
                pty.render(),
                a.type_name()
            ))));
        }
    }
    run(prog, fi, args, limits)
}

/// Like [`call`], but skips the deep per-argument validation walk (arity is still
/// checked). For host paths whose inputs were already validated on construction
/// (`Value::record`) or deserialization (`wire::WireValue::to_value`) — e.g. the
/// per-node loop of `bundle::score_batch`, where re-walking the shared `ctx` on every
/// node would make a batch O(N²) in cluster size.
pub fn call_prevalidated(
    prog: &Program,
    func: &str,
    args: &[Value],
    limits: Limits,
) -> Result<CallOutcome, CallError> {
    let fi = resolve(prog, func, args)?;
    run(prog, fi, args, limits)
}

/// Like [`call`], but reports resource use for **failures too** (spec §6.4: fuel and
/// memory accounting are part of the observable, cross-target-deterministic outcome —
/// the conformance suite compares them on error paths as well). Host-boundary errors
/// (unknown function / bad arguments) carry zero cost: the invocation never ran.
pub fn call_costed(
    prog: &Program,
    func: &str,
    args: &[Value],
    limits: Limits,
) -> (Result<Value, CallError>, u64, u64) {
    let fi = match resolve(prog, func, args) {
        Ok(fi) => fi,
        Err(e) => return (Err(e), 0, 0),
    };
    let f = &prog.module.funcs[fi];
    for (a, (pname, pty)) in args.iter().zip(f.params.iter()) {
        if !a.conforms(pty) {
            let e = CallError::Host(HostError::BadArguments(format!(
                "argument '{pname}' of '{func}' must be {} (with finite floats), got {}",
                pty.render(),
                a.type_name()
            )));
            return (Err(e), 0, 0);
        }
    }
    let mut interp = Interp {
        prog,
        fuel_left: limits.fuel,
        mem_left: limits.memory_bytes,
        limits,
    };
    let result = interp.call_fn(fi, args.to_vec());
    (
        result.map_err(CallError::Ps),
        limits.fuel - interp.fuel_left,
        limits.memory_bytes - interp.mem_left,
    )
}

fn resolve(prog: &Program, func: &str, args: &[Value]) -> Result<usize, CallError> {
    let Some(&fi) = prog.fn_index.get(func) else {
        return Err(CallError::Host(HostError::UnknownFunction(func.into())));
    };
    let f = &prog.module.funcs[fi];
    if args.len() != f.params.len() {
        return Err(CallError::Host(HostError::BadArguments(format!(
            "'{func}' expects {} argument(s), got {}",
            f.params.len(),
            args.len()
        ))));
    }
    Ok(fi)
}

fn run(
    prog: &Program,
    fi: usize,
    args: &[Value],
    limits: Limits,
) -> Result<CallOutcome, CallError> {
    let mut interp = Interp {
        prog,
        fuel_left: limits.fuel,
        mem_left: limits.memory_bytes,
        limits,
    };
    let value = interp.call_fn(fi, args.to_vec())?;
    Ok(CallOutcome {
        value,
        fuel_used: limits.fuel - interp.fuel_left,
        mem_used: limits.memory_bytes - interp.mem_left,
    })
}

enum Flow {
    Normal,
    Return(Value),
}

struct Interp<'a> {
    prog: &'a Program,
    fuel_left: u64,
    mem_left: u64,
    #[allow(dead_code)]
    limits: Limits,
}

fn trap(message: impl Into<String>) -> PsError {
    PsError::Trap {
        message: message.into(),
    }
}

impl<'a> Interp<'a> {
    #[inline]
    fn fuel(&mut self, cost: u64) -> Result<(), PsError> {
        if self.fuel_left < cost {
            self.fuel_left = 0;
            return Err(PsError::FuelExhausted);
        }
        self.fuel_left -= cost;
        Ok(())
    }

    #[inline]
    fn mem(&mut self, bytes: u64) -> Result<(), PsError> {
        if self.mem_left < bytes {
            self.mem_left = 0;
            return Err(PsError::OutOfMemory);
        }
        self.mem_left -= bytes;
        Ok(())
    }

    fn res(&self, id: NodeId) -> Res {
        self.prog.res[id as usize]
            .clone()
            .expect("resolution missing: typecheck bug")
    }

    fn call_fn(&mut self, fi: usize, args: Vec<Value>) -> Result<Value, PsError> {
        let f = &self.prog.module.funcs[fi];
        let nslots = self.prog.fn_nslots[fi] as usize;
        let mut locals: Vec<Option<Value>> = vec![None; nslots];
        for (i, a) in args.into_iter().enumerate() {
            locals[i] = Some(a);
        }
        match self.exec_block(&f.body, &mut locals)? {
            Flow::Return(v) => Ok(v),
            Flow::Normal => unreachable!("missing return: typecheck bug"),
        }
    }

    fn exec_block(
        &mut self,
        stmts: &[StmtNode],
        locals: &mut Vec<Option<Value>>,
    ) -> Result<Flow, PsError> {
        for s in stmts {
            match self.exec_stmt(s, locals)? {
                Flow::Normal => {}
                r => return Ok(r),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(
        &mut self,
        s: &StmtNode,
        locals: &mut Vec<Option<Value>>,
    ) -> Result<Flow, PsError> {
        self.fuel(1)?; // statement dispatch (spec §8.1)
        match &s.kind {
            Stmt::Assign { value, .. } => {
                let v = self.eval(value, locals)?;
                let Res::Local(slot) = self.res(s.id) else {
                    unreachable!()
                };
                locals[slot as usize] = Some(v);
                Ok(Flow::Normal)
            }
            Stmt::AugAssign { op, value, .. } => {
                let Res::Local(slot) = self.res(s.id) else {
                    unreachable!()
                };
                let cur = locals[slot as usize].clone().expect("definite assignment");
                let rhs = self.eval(value, locals)?;
                let v = self.binop(*op, cur, rhs)?;
                locals[slot as usize] = Some(v);
                Ok(Flow::Normal)
            }
            Stmt::Append { value, .. } => {
                let Res::Local(slot) = self.res(s.id) else {
                    unreachable!()
                };
                let v = self.eval(value, locals)?;
                let Some(Value::List(xs)) = &locals[slot as usize] else {
                    unreachable!("append target: typecheck bug")
                };
                if xs.frozen() {
                    // Host-provided state is read-only (spec §5.2): mutating it through
                    // an alias would leak across batch candidates and violate C2.
                    return Err(trap("append to a host-provided list (read-only)"));
                }
                let xs = xs.clone();
                self.mem(16)?;
                xs.push(v);
                Ok(Flow::Normal)
            }
            Stmt::Return(e) => {
                let v = self.eval(e, locals)?;
                Ok(Flow::Return(v))
            }
            Stmt::Pass => Ok(Flow::Normal),
            Stmt::If { arms, orelse } => {
                for (cond, block) in arms {
                    let Value::Bool(b) = self.eval(cond, locals)? else {
                        unreachable!("non-bool condition: typecheck bug")
                    };
                    if b {
                        return self.exec_block(block, locals);
                    }
                }
                if let Some(block) = orelse {
                    return self.exec_block(block, locals);
                }
                Ok(Flow::Normal)
            }
            Stmt::For { iter, body, .. } => {
                let Res::Local(slot) = self.res(s.id) else {
                    unreachable!()
                };
                match iter {
                    ForIter::Range(a, b) => {
                        self.fuel(1)?; // range header (spec §8.1)
                        let lo = match a {
                            Some(a) => {
                                let Value::Int(v) = self.eval(a, locals)? else {
                                    unreachable!()
                                };
                                v
                            }
                            None => 0,
                        };
                        let Value::Int(hi) = self.eval(b, locals)? else {
                            unreachable!()
                        };
                        let mut i = lo;
                        loop {
                            self.fuel(1)?; // per-iteration check (incl. the exiting one)
                            if i >= hi {
                                break;
                            }
                            locals[slot as usize] = Some(Value::Int(i));
                            match self.exec_block(body, locals)? {
                                Flow::Normal => {}
                                r => return Ok(r),
                            }
                            // hi <= i64::MAX and i < hi, so i + 1 cannot overflow.
                            i += 1;
                        }
                    }
                    ForIter::List(e) => {
                        let Value::List(xs) = self.eval(e, locals)? else {
                            unreachable!()
                        };
                        let n = xs.borrow().len();
                        let mut i = 0usize;
                        loop {
                            self.fuel(1)?;
                            if xs.borrow().len() != n {
                                return Err(trap("list mutated during iteration"));
                            }
                            if i >= n {
                                break;
                            }
                            let item = xs.borrow()[i].clone();
                            locals[slot as usize] = Some(item);
                            match self.exec_block(body, locals)? {
                                Flow::Normal => {}
                                r => return Ok(r),
                            }
                            i += 1;
                        }
                    }
                }
                Ok(Flow::Normal)
            }
        }
    }

    fn eval(&mut self, e: &ExprNode, locals: &mut Vec<Option<Value>>) -> Result<Value, PsError> {
        match &e.kind {
            Expr::Int(v) => {
                self.fuel(1)?;
                Ok(Value::Int(*v))
            }
            Expr::Float(v) => {
                self.fuel(1)?;
                Ok(Value::Float(*v))
            }
            Expr::Bool(v) => {
                self.fuel(1)?;
                Ok(Value::Bool(*v))
            }
            Expr::Str(_) => {
                self.fuel(1)?;
                // Interned at load (spec §8.2): clone the interned Rc — no allocation,
                // no per-evaluation memory charge.
                let Res::Str(idx) = self.res(e.id) else {
                    unreachable!("string literal not interned: typecheck bug")
                };
                Ok(self.prog.str_vals[idx as usize].clone())
            }
            Expr::Name(_) => {
                self.fuel(1)?;
                match self.res(e.id) {
                    Res::Local(slot) => {
                        Ok(locals[slot as usize].clone().expect("definite assignment"))
                    }
                    Res::Const(idx) => Ok(self.prog.const_vals[idx as usize].clone()),
                    _ => unreachable!(),
                }
            }
            Expr::Neg(inner) => {
                self.fuel(1)?;
                match self.eval(inner, locals)? {
                    Value::Int(v) => v
                        .checked_neg()
                        .map(Value::Int)
                        .ok_or_else(|| trap("integer overflow in unary '-'")),
                    Value::Float(v) => Ok(Value::Float(-v)), // finite in => finite out
                    _ => unreachable!(),
                }
            }
            Expr::Not(inner) => {
                self.fuel(1)?;
                let Value::Bool(b) = self.eval(inner, locals)? else {
                    unreachable!()
                };
                Ok(Value::Bool(!b))
            }
            Expr::Binary(op, l, r) => {
                self.fuel(1)?;
                let lv = self.eval(l, locals)?;
                let rv = self.eval(r, locals)?;
                self.binop(*op, lv, rv)
            }
            Expr::Compare(op, l, r) => {
                self.fuel(1)?;
                let lv = self.eval(l, locals)?;
                let rv = self.eval(r, locals)?;
                Ok(Value::Bool(compare(*op, &lv, &rv)))
            }
            Expr::BoolChain(op, l, r) => {
                self.fuel(1)?;
                let Value::Bool(lv) = self.eval(l, locals)? else {
                    unreachable!()
                };
                match (op, lv) {
                    (BoolOp::And, false) => Ok(Value::Bool(false)),
                    (BoolOp::Or, true) => Ok(Value::Bool(true)),
                    _ => {
                        let Value::Bool(rv) = self.eval(r, locals)? else {
                            unreachable!()
                        };
                        Ok(Value::Bool(rv))
                    }
                }
            }
            Expr::Index(base, idx) => {
                self.fuel(1)?;
                let Value::List(xs) = self.eval(base, locals)? else {
                    unreachable!()
                };
                let Value::Int(i) = self.eval(idx, locals)? else {
                    unreachable!()
                };
                let xs = xs.borrow();
                if i < 0 || (i as usize) >= xs.len() {
                    return Err(trap(format!(
                        "list index {i} out of bounds (len {})",
                        xs.len()
                    )));
                }
                Ok(xs[i as usize].clone())
            }
            Expr::Field(base, _) => {
                self.fuel(1)?;
                let Value::Record(r) = self.eval(base, locals)? else {
                    unreachable!()
                };
                let Res::Field(idx) = self.res(e.id) else {
                    unreachable!()
                };
                Ok(r.fields[idx as usize].clone())
            }
            Expr::Call(_, args) => {
                let mut argv = Vec::with_capacity(args.len());
                for a in args {
                    argv.push(self.eval(a, locals)?);
                }
                match self.res(e.id) {
                    Res::CallBuiltin(b) => {
                        self.fuel(1)?;
                        self.builtin(b, argv)
                    }
                    Res::CallUser(fi) => {
                        self.fuel(10)?; // call overhead (spec §8.1)
                        self.call_fn(fi as usize, argv)
                    }
                    _ => unreachable!(),
                }
            }
            Expr::ListDisplay(items) => {
                self.fuel(1 + items.len() as u64)?;
                self.mem(32 + 16 * items.len() as u64)?;
                let mut xs = Vec::with_capacity(items.len());
                for item in items {
                    xs.push(self.eval(item, locals)?);
                }
                Ok(Value::list(xs))
            }
        }
    }

    fn binop(&mut self, op: BinOp, l: Value, r: Value) -> Result<Value, PsError> {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => {
                let v = match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::FloorDiv => {
                        if b == 0 {
                            return Err(trap("integer division by zero"));
                        }
                        floor_div(a, b)
                    }
                    BinOp::Mod => {
                        if b == 0 {
                            return Err(trap("integer modulo by zero"));
                        }
                        py_mod(a, b)
                    }
                    BinOp::Div => unreachable!("'/' on ints: typecheck bug"),
                };
                v.map(Value::Int)
                    .ok_or_else(|| trap(format!("integer overflow in '{}'", op.token())))
            }
            (Value::Float(a), Value::Float(b)) => {
                let v = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    _ => unreachable!("int-only op on floats: typecheck bug"),
                };
                if !v.is_finite() {
                    return Err(trap(format!("non-finite float result in '{}'", op.token())));
                }
                Ok(Value::Float(v))
            }
            _ => unreachable!("mixed operand types: typecheck bug"),
        }
    }

    fn builtin(&mut self, b: Builtin, mut args: Vec<Value>) -> Result<Value, PsError> {
        let one = |args: &mut Vec<Value>| args.pop().unwrap();
        match b {
            Builtin::Len => {
                let Value::List(xs) = one(&mut args) else {
                    unreachable!()
                };
                let n = xs.borrow().len() as i64;
                Ok(Value::Int(n))
            }
            Builtin::AbsInt => {
                let Value::Int(v) = one(&mut args) else {
                    unreachable!()
                };
                v.checked_abs()
                    .map(Value::Int)
                    .ok_or_else(|| trap("integer overflow in abs()"))
            }
            Builtin::AbsFloat => {
                let Value::Float(v) = one(&mut args) else {
                    unreachable!()
                };
                Ok(Value::Float(v.abs()))
            }
            Builtin::MinInt | Builtin::MaxInt => {
                let (Value::Int(bv), Value::Int(av)) = (args.pop().unwrap(), args.pop().unwrap())
                else {
                    unreachable!()
                };
                let take_b = if matches!(b, Builtin::MinInt) {
                    bv < av
                } else {
                    bv > av
                };
                Ok(Value::Int(if take_b { bv } else { av })) // returns a on ties (spec §7)
            }
            Builtin::MinFloat | Builtin::MaxFloat => {
                let (Value::Float(bv), Value::Float(av)) =
                    (args.pop().unwrap(), args.pop().unwrap())
                else {
                    unreachable!()
                };
                let take_b = if matches!(b, Builtin::MinFloat) {
                    bv < av
                } else {
                    bv > av
                };
                Ok(Value::Float(if take_b { bv } else { av }))
            }
            Builtin::Sqrt => {
                let Value::Float(v) = one(&mut args) else {
                    unreachable!()
                };
                if v < 0.0 {
                    return Err(trap("sqrt() of a negative number (result would be NaN)"));
                }
                Ok(Value::Float(v.sqrt()))
            }
            Builtin::CastFloat => {
                let Value::Int(v) = one(&mut args) else {
                    unreachable!()
                };
                Ok(Value::Float(v as f64)) // round-to-nearest; always finite
            }
            Builtin::CastInt => {
                let Value::Float(v) = one(&mut args) else {
                    unreachable!()
                };
                let t = v.trunc();
                // Exact-boundary check: 2^63 as f64 is exactly 9223372036854775808.0.
                if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&t) {
                    return Err(trap("int() result out of i64 range"));
                }
                Ok(Value::Int(t as i64))
            }
            Builtin::MovePod => {
                let (Value::Int(dest), Value::Int(pod)) =
                    (args.pop().unwrap(), args.pop().unwrap())
                else {
                    unreachable!()
                };
                self.mem(32)?;
                Ok(Value::Record(Rc::new(RecordVal {
                    rt: RecordType::Move,
                    fields: vec![Value::Int(0), Value::Int(pod), Value::Int(dest)],
                })))
            }
            Builtin::Wake => {
                let Value::Int(node) = one(&mut args) else {
                    unreachable!()
                };
                self.mem(32)?;
                Ok(Value::Record(Rc::new(RecordVal {
                    rt: RecordType::Move,
                    fields: vec![Value::Int(1), Value::Int(-1), Value::Int(node)],
                })))
            }
        }
    }
}

/// Python floor division (round toward negative infinity). `None` on overflow.
fn floor_div(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    let r = a % b; // safe: b != 0 and not (MIN, -1) past checked_div
    if r != 0 && ((r < 0) != (b < 0)) {
        Some(q - 1) // cannot overflow: q > i64::MIN here
    } else {
        Some(q)
    }
}

/// Python modulo (sign of the divisor). `None` on overflow.
fn py_mod(a: i64, b: i64) -> Option<i64> {
    if a == i64::MIN && b == -1 {
        return None; // match `//` trap behavior so the div/mod identity holds
    }
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        Some(r + b)
    } else {
        Some(r)
    }
}

fn compare(op: CmpOp, l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => cmp_ord(op, a.cmp(b)),
        // No NaN exists: partial_cmp is total here (spec §6.3).
        (Value::Float(a), Value::Float(b)) => cmp_ord(op, a.partial_cmp(b).unwrap()),
        (Value::Bool(a), Value::Bool(b)) => match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            _ => unreachable!("bool ordering: typecheck bug"),
        },
        (Value::Str(a), Value::Str(b)) => match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            _ => unreachable!("str ordering: typecheck bug"),
        },
        _ => unreachable!("mixed comparison: typecheck bug"),
    }
}

fn cmp_ord(op: CmpOp, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match op {
        CmpOp::Lt => ord == Less,
        CmpOp::Le => ord != Greater,
        CmpOp::Gt => ord == Greater,
        CmpOp::Ge => ord != Less,
        CmpOp::Eq => ord == Equal,
        CmpOp::Ne => ord != Equal,
    }
}

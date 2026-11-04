//! Static checks (spec §3–§5): types, definite assignment, return paths, the
//! no-recursion rule, reserved names, and load-budget accounting. Produces a `Program`
//! with side tables (expression types, name/call/field resolutions) consumed by the
//! interpreter and the AST-level API.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::*;
use crate::value::Value;
use crate::{Limits, PsError};

/// Resolved builtin, with overloads already selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Builtin {
    Len,
    AbsInt,
    AbsFloat,
    MinInt,
    MinFloat,
    MaxInt,
    MaxFloat,
    Sqrt,
    CastFloat,
    CastInt,
    MovePod,
    Wake,
}

/// Per-node resolution, filled during type checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Res {
    /// Local variable slot (params, locals, loop vars; also assignment targets).
    Local(u32),
    /// Module constant index.
    Const(u32),
    /// Call to a user function (index into `Module::funcs`).
    CallUser(u32),
    /// Call to a builtin.
    CallBuiltin(Builtin),
    /// Record field index.
    Field(u16),
    /// Interned string literal (index into `Program::str_vals`). Interning happens at
    /// load (spec §8.2): evaluation clones the same `Rc<str>`, never re-allocates.
    Str(u32),
}

/// Maximum number of function definitions per module (spec §8.1 structural limits).
pub const MAX_FUNCTIONS: usize = 64;

/// Maximum static call-chain depth (spec §8.1): the call graph is a DAG of
/// earlier-defined functions, so this is computable exactly at load and bounds the
/// interpreter's dynamic call depth on every build target.
pub const MAX_CALL_DEPTH: u32 = 16;

/// A compiled (parsed + type-checked) PlacementScript program.
#[derive(Debug, Clone)]
pub struct Program {
    pub(crate) module: Module,
    pub(crate) source: String,
    pub(crate) expr_ty: Vec<Option<Type>>,
    pub(crate) res: Vec<Option<Res>>,
    pub(crate) fn_index: BTreeMap<String, usize>,
    pub(crate) fn_nslots: Vec<u32>,
    pub(crate) const_vals: Vec<Value>,
    pub(crate) const_index: BTreeMap<String, usize>,
    /// Interned expression string literals (spec §8.2), indexed by `Res::Str`.
    pub(crate) str_vals: Vec<Value>,
}

impl Program {
    /// Original source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The parsed module (read-only).
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Static type of an expression node, if `id` names an expression.
    pub fn expr_type(&self, id: NodeId) -> Option<&Type> {
        self.expr_ty.get(id as usize).and_then(|t| t.as_ref())
    }

    /// Signature of a function by name: (param types, return type).
    pub fn signature(&self, name: &str) -> Option<(Vec<Type>, Type)> {
        let f = &self.module.funcs[*self.fn_index.get(name)?];
        Some((
            f.params.iter().map(|(_, t)| t.clone()).collect(),
            f.ret.clone(),
        ))
    }

    /// Value of a module constant by name.
    pub fn const_value(&self, name: &str) -> Option<&Value> {
        self.const_index.get(name).map(|i| &self.const_vals[*i])
    }
}

const TYPE_NAMES: [&str; 5] = ["int", "float", "bool", "str", "list"];
const BUILTIN_NAMES: [&str; 9] = [
    "len", "abs", "min", "max", "sqrt", "int", "float", "move_pod", "wake",
];

fn is_reserved(name: &str) -> bool {
    TYPE_NAMES.contains(&name)
        || BUILTIN_NAMES.contains(&name)
        || RecordType::by_name(name).is_some()
}

fn terr(line: u32, message: impl Into<String>) -> PsError {
    PsError::Type {
        line,
        message: message.into(),
    }
}

/// Flow-sensitive state: which names are visible / definitely assigned.
#[derive(Clone)]
struct Flow {
    visible: BTreeSet<String>,
    assigned: BTreeSet<String>,
}

struct FnScope {
    /// Function-wide name → (slot, type). Monotonic: one type per name per function.
    slots: BTreeMap<String, (u32, Type)>,
    n_slots: u32,
}

struct Checker<'a> {
    module: &'a Module,
    expr_ty: Vec<Option<Type>>,
    res: Vec<Option<Res>>,
    /// Functions checked so far (callable). The current function is added *after* its
    /// body is checked — recursion is unresolvable by construction (spec §5.4).
    callable: BTreeMap<String, usize>,
    all_funcs: BTreeSet<String>,
    const_types: BTreeMap<String, (u32, Type)>,
    string_bytes: u64,
    /// Interned expression string literals, built during checking.
    str_vals: Vec<Value>,
    /// Static call-chain depth per already-checked function (1 = calls no user fn).
    fn_depth: Vec<u32>,
    /// Max callee chain depth seen while checking the current function body.
    cur_max_callee: u32,
}

/// Type-check a parsed module under `load_limits` (spec §8: module-constant and
/// string-literal costs are charged to the load budget).
pub fn check(module: Module, source: String, load_limits: Limits) -> Result<Program, PsError> {
    let mut const_vals = Vec::new();
    let mut const_index = BTreeMap::new();
    let mut const_types = BTreeMap::new();
    let mut string_bytes: u64 = 0;

    for (i, c) in module.consts.iter().enumerate() {
        if is_reserved(&c.name) {
            return Err(terr(c.line, format!("'{}' is a reserved name", c.name)));
        }
        if const_index.contains_key(&c.name) {
            return Err(terr(c.line, format!("duplicate constant '{}'", c.name)));
        }
        let (v, ty) = match (&c.value, &c.ty) {
            (ConstLit::Int(v), Type::Int) => (Value::Int(*v), Type::Int),
            (ConstLit::Float(v), Type::Float) => (Value::Float(*v), Type::Float),
            (ConstLit::Str(s), Type::Str) => {
                string_bytes += 32 + s.len() as u64;
                (Value::str(s), Type::Str)
            }
            (ConstLit::Bool(v), Type::Bool) => (Value::Bool(*v), Type::Bool),
            _ => {
                return Err(terr(
                    c.line,
                    format!("constant '{}' does not match its annotation", c.name),
                ))
            }
        };
        const_index.insert(c.name.clone(), i);
        const_types.insert(c.name.clone(), (i as u32, ty));
        const_vals.push(v);
    }

    // Load-budget accounting (spec §8.1): one fuel per source byte (already charged
    // incrementally during lexing, which bounds compiler allocation on token-dense
    // input — Round-3 review) + one per constant + one per AST node. This bounds
    // total program size — an oversized module fails at *load* with a structured
    // error instead of stressing the interpreter later.
    if (source.len() as u64) + (module.consts.len() as u64) + (module.node_count as u64)
        > load_limits.fuel
    {
        return Err(PsError::FuelExhausted);
    }

    // Structural limit (spec §8.1): function count. Together with MAX_CALL_DEPTH and
    // the parser's nesting limits this statically bounds interpreter recursion.
    if module.funcs.len() > MAX_FUNCTIONS {
        let f = &module.funcs[MAX_FUNCTIONS];
        return Err(terr(
            f.line,
            format!("too many functions (max {MAX_FUNCTIONS} per module; spec §8.1)"),
        ));
    }

    let all_funcs: BTreeSet<String> = module.funcs.iter().map(|f| f.name.clone()).collect();
    let mut ck = Checker {
        module: &module,
        expr_ty: vec![None; module.node_count as usize],
        res: vec![None; module.node_count as usize],
        callable: BTreeMap::new(),
        all_funcs,
        const_types,
        string_bytes,
        str_vals: Vec::new(),
        fn_depth: Vec::new(),
        cur_max_callee: 0,
    };

    let mut fn_index = BTreeMap::new();
    let mut fn_nslots = Vec::new();
    for (fi, f) in module.funcs.iter().enumerate() {
        if is_reserved(&f.name) {
            return Err(terr(f.line, format!("'{}' is a reserved name", f.name)));
        }
        if fn_index.contains_key(&f.name) {
            return Err(terr(f.line, format!("duplicate function '{}'", f.name)));
        }
        if ck.const_types.contains_key(&f.name) {
            return Err(terr(
                f.line,
                format!("'{}' is already a module constant", f.name),
            ));
        }
        ck.cur_max_callee = 0;
        let nslots = ck.check_func(f)?;
        let depth = ck.cur_max_callee + 1;
        if depth > MAX_CALL_DEPTH {
            return Err(terr(
                f.line,
                format!(
                    "call chain too deep: '{}' reaches static call depth {depth} \
                     (max {MAX_CALL_DEPTH}; spec §8.1)",
                    f.name
                ),
            ));
        }
        ck.fn_depth.push(depth);
        ck.callable.insert(f.name.clone(), fi);
        fn_index.insert(f.name.clone(), fi);
        fn_nslots.push(nslots);
    }

    if ck.string_bytes > load_limits.memory_bytes {
        return Err(PsError::OutOfMemory);
    }

    let (expr_ty, res, str_vals) = (ck.expr_ty, ck.res, ck.str_vals);
    Ok(Program {
        module,
        source,
        expr_ty,
        res,
        fn_index,
        fn_nslots,
        const_vals,
        const_index,
        str_vals,
    })
}

impl<'a> Checker<'a> {
    fn check_func(&mut self, f: &FuncDef) -> Result<u32, PsError> {
        let mut scope = FnScope {
            slots: BTreeMap::new(),
            n_slots: 0,
        };
        let mut flow = Flow {
            visible: BTreeSet::new(),
            assigned: BTreeSet::new(),
        };
        let mut seen = BTreeSet::new();
        for (name, ty) in &f.params {
            if is_reserved(name) {
                return Err(terr(f.line, format!("'{name}' is a reserved name")));
            }
            if self.const_types.contains_key(name) || self.all_funcs.contains(name) {
                return Err(terr(
                    f.line,
                    format!("parameter '{name}' shadows a module-level name"),
                ));
            }
            if !seen.insert(name.clone()) {
                return Err(terr(f.line, format!("duplicate parameter '{name}'")));
            }
            let slot = scope.n_slots;
            scope.n_slots += 1;
            scope.slots.insert(name.clone(), (slot, ty.clone()));
            flow.visible.insert(name.clone());
            flow.assigned.insert(name.clone());
        }
        let returns = self.check_block(&f.body, &mut scope, &mut flow, &f.ret)?;
        if !returns {
            return Err(terr(
                f.line,
                format!(
                    "function '{}' does not return on every control path",
                    f.name
                ),
            ));
        }
        Ok(scope.n_slots)
    }

    /// Returns whether the block definitely returns.
    fn check_block(
        &mut self,
        stmts: &[StmtNode],
        scope: &mut FnScope,
        flow: &mut Flow,
        ret_ty: &Type,
    ) -> Result<bool, PsError> {
        let mut returns = false;
        for s in stmts {
            if self.check_stmt(s, scope, flow, ret_ty)? {
                returns = true;
            }
        }
        Ok(returns)
    }

    /// Declare-or-reuse a slot for `name` with type `ty`.
    fn bind(
        &mut self,
        scope: &mut FnScope,
        flow: &mut Flow,
        name: &str,
        ty: &Type,
        line: u32,
    ) -> Result<u32, PsError> {
        if is_reserved(name) {
            return Err(terr(line, format!("'{name}' is a reserved name")));
        }
        if self.const_types.contains_key(name) {
            return Err(terr(
                line,
                format!("cannot assign to module constant '{name}'"),
            ));
        }
        if self.all_funcs.contains(name) {
            return Err(terr(
                line,
                format!("cannot assign to function name '{name}'"),
            ));
        }
        let slot = match scope.slots.get(name) {
            Some((slot, prev)) => {
                if prev != ty {
                    return Err(terr(
                        line,
                        format!(
                            "'{name}' has type {} and cannot be re-bound to {}",
                            prev.render(),
                            ty.render()
                        ),
                    ));
                }
                *slot
            }
            None => {
                let slot = scope.n_slots;
                scope.n_slots += 1;
                scope.slots.insert(name.to_string(), (slot, ty.clone()));
                slot
            }
        };
        flow.visible.insert(name.to_string());
        flow.assigned.insert(name.to_string());
        Ok(slot)
    }

    fn check_stmt(
        &mut self,
        s: &StmtNode,
        scope: &mut FnScope,
        flow: &mut Flow,
        ret_ty: &Type,
    ) -> Result<bool, PsError> {
        match &s.kind {
            Stmt::Assign { name, ann, value } => {
                let vty = match (&value.kind, ann) {
                    // Empty list display: type comes from the annotation (spec §5.3).
                    (Expr::ListDisplay(items), Some(Type::List(elem))) if items.is_empty() => {
                        let t = Type::List(elem.clone());
                        self.expr_ty[value.id as usize] = Some(t.clone());
                        t
                    }
                    _ => self.check_expr(value, scope, flow)?,
                };
                if let Some(a) = ann {
                    if *a != vty {
                        return Err(terr(
                            s.line,
                            format!(
                                "annotation {} does not match value type {}",
                                a.render(),
                                vty.render()
                            ),
                        ));
                    }
                }
                let slot = self.bind(scope, flow, name, &vty, s.line)?;
                self.res[s.id as usize] = Some(Res::Local(slot));
                Ok(false)
            }
            Stmt::AugAssign { name, op, value } => {
                let (slot, tty) = self.read_var(name, scope, flow, s.line)?;
                let vty = self.check_expr(value, scope, flow)?;
                let out = self.binop_type(*op, &tty, &vty, s.line)?;
                if out != tty {
                    return Err(terr(s.line, "augmented assignment changes the type"));
                }
                self.res[s.id as usize] = Some(Res::Local(slot));
                Ok(false)
            }
            Stmt::Append { name, value } => {
                let (slot, tty) = self.read_var(name, scope, flow, s.line)?;
                let Type::List(elem) = &tty else {
                    return Err(terr(
                        s.line,
                        format!("'.append' target must be a list, got {}", tty.render()),
                    ));
                };
                let vty = self.check_expr(value, scope, flow)?;
                if vty != **elem {
                    return Err(terr(
                        s.line,
                        format!("cannot append {} to {}", vty.render(), tty.render()),
                    ));
                }
                self.res[s.id as usize] = Some(Res::Local(slot));
                Ok(false)
            }
            Stmt::Return(e) => {
                let ty = self.check_expr(e, scope, flow)?;
                if ty != *ret_ty {
                    return Err(terr(
                        s.line,
                        format!(
                            "return type {} does not match declared {}",
                            ty.render(),
                            ret_ty.render()
                        ),
                    ));
                }
                Ok(true)
            }
            Stmt::Pass => Ok(false),
            Stmt::If { arms, orelse } => {
                let mut all_return = true;
                let mut assigned_sets: Vec<BTreeSet<String>> = Vec::new();
                let mut visible_union = flow.visible.clone();
                for (cond, block) in arms {
                    let cty = self.check_expr(cond, scope, flow)?;
                    if cty != Type::Bool {
                        return Err(terr(
                            cond.line,
                            format!("condition must be bool, got {}", cty.render()),
                        ));
                    }
                    let mut branch = flow.clone();
                    let r = self.check_block(block, scope, &mut branch, ret_ty)?;
                    all_return &= r;
                    assigned_sets.push(branch.assigned);
                    visible_union.extend(branch.visible);
                }
                match orelse {
                    Some(block) => {
                        let mut branch = flow.clone();
                        let r = self.check_block(block, scope, &mut branch, ret_ty)?;
                        all_return &= r;
                        assigned_sets.push(branch.assigned);
                        visible_union.extend(branch.visible);
                    }
                    None => {
                        all_return = false;
                        assigned_sets.push(flow.assigned.clone());
                    }
                }
                // Definitely assigned after = intersection over all paths.
                let mut merged = assigned_sets.pop().unwrap();
                for s in &assigned_sets {
                    merged.retain(|n| s.contains(n));
                }
                flow.assigned = merged;
                flow.visible = visible_union;
                Ok(all_return)
            }
            Stmt::For { var, iter, body } => {
                let elem_ty = match iter {
                    ForIter::Range(a, b) => {
                        if let Some(a) = a {
                            let t = self.check_expr(a, scope, flow)?;
                            if t != Type::Int {
                                return Err(terr(s.line, "range bounds must be int"));
                            }
                        }
                        let t = self.check_expr(b, scope, flow)?;
                        if t != Type::Int {
                            return Err(terr(s.line, "range bounds must be int"));
                        }
                        Type::Int
                    }
                    ForIter::List(e) => {
                        let t = self.check_expr(e, scope, flow)?;
                        let Type::List(elem) = t else {
                            return Err(terr(
                                s.line,
                                format!("cannot iterate over {}", t.render()),
                            ));
                        };
                        *elem
                    }
                };
                if flow.visible.contains(var) {
                    return Err(terr(
                        s.line,
                        format!("loop variable '{var}' shadows an existing variable"),
                    ));
                }
                let mut body_flow = flow.clone();
                let slot = self.bind(scope, &mut body_flow, var, &elem_ty, s.line)?;
                self.res[s.id as usize] = Some(Res::Local(slot));
                let mut bf = body_flow;
                self.check_block(body, scope, &mut bf, ret_ty)?;
                // The body may run zero times: assignments inside don't become definite.
                // Loop variable goes out of scope (spec §5.3).
                let mut visible = bf.visible;
                visible.remove(var);
                flow.visible = visible;
                Ok(false)
            }
        }
    }

    fn read_var(
        &self,
        name: &str,
        scope: &FnScope,
        flow: &Flow,
        line: u32,
    ) -> Result<(u32, Type), PsError> {
        match scope.slots.get(name) {
            Some((slot, ty)) if flow.visible.contains(name) => {
                if !flow.assigned.contains(name) {
                    return Err(terr(
                        line,
                        format!("'{name}' may be unassigned on some control path"),
                    ));
                }
                Ok((*slot, ty.clone()))
            }
            _ => Err(terr(line, format!("undefined variable '{name}'"))),
        }
    }

    fn binop_type(&self, op: BinOp, l: &Type, r: &Type, line: u32) -> Result<Type, PsError> {
        let both = |t: &Type| l == t && r == t;
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => {
                if both(&Type::Int) {
                    Ok(Type::Int)
                } else if both(&Type::Float) {
                    Ok(Type::Float)
                } else {
                    Err(terr(
                        line,
                        format!(
                            "'{}' needs two ints or two floats, got {} and {} \
                             (no implicit conversion; use float()/int())",
                            op.token(),
                            l.render(),
                            r.render()
                        ),
                    ))
                }
            }
            BinOp::Div => {
                if both(&Type::Float) {
                    Ok(Type::Float)
                } else {
                    Err(terr(
                        line,
                        format!(
                            "'/' is float-only, got {} and {} (use '//' for ints)",
                            l.render(),
                            r.render()
                        ),
                    ))
                }
            }
            BinOp::FloorDiv | BinOp::Mod => {
                if both(&Type::Int) {
                    Ok(Type::Int)
                } else {
                    Err(terr(
                        line,
                        format!(
                            "'{}' is int-only, got {} and {}",
                            op.token(),
                            l.render(),
                            r.render()
                        ),
                    ))
                }
            }
        }
    }

    fn check_expr(
        &mut self,
        e: &ExprNode,
        scope: &mut FnScope,
        flow: &mut Flow,
    ) -> Result<Type, PsError> {
        let ty = match &e.kind {
            Expr::Int(_) => Type::Int,
            Expr::Float(_) => Type::Float,
            Expr::Bool(_) => Type::Bool,
            Expr::Str(s) => {
                // Intern at load (spec §8.2): charged to the load memory budget here;
                // evaluation clones the interned Rc — zero per-evaluation allocation.
                self.string_bytes += 32 + s.len() as u64;
                self.res[e.id as usize] = Some(Res::Str(self.str_vals.len() as u32));
                self.str_vals.push(Value::str(s));
                Type::Str
            }
            Expr::Name(name) => {
                if let Some((idx, ty)) = self.const_types.get(name) {
                    self.res[e.id as usize] = Some(Res::Const(*idx));
                    ty.clone()
                } else {
                    let (slot, ty) = self.read_var(name, scope, flow, e.line)?;
                    self.res[e.id as usize] = Some(Res::Local(slot));
                    ty
                }
            }
            Expr::Neg(inner) => {
                let t = self.check_expr(inner, scope, flow)?;
                if t != Type::Int && t != Type::Float {
                    return Err(terr(
                        e.line,
                        format!("unary '-' needs int or float, got {}", t.render()),
                    ));
                }
                t
            }
            Expr::Not(inner) => {
                let t = self.check_expr(inner, scope, flow)?;
                if t != Type::Bool {
                    return Err(terr(
                        e.line,
                        format!("'not' needs bool, got {}", t.render()),
                    ));
                }
                Type::Bool
            }
            Expr::Binary(op, l, r) => {
                let lt = self.check_expr(l, scope, flow)?;
                let rt = self.check_expr(r, scope, flow)?;
                self.binop_type(*op, &lt, &rt, e.line)?
            }
            Expr::Compare(op, l, r) => {
                let lt = self.check_expr(l, scope, flow)?;
                let rt = self.check_expr(r, scope, flow)?;
                if lt != rt {
                    return Err(terr(
                        e.line,
                        format!("cannot compare {} with {}", lt.render(), rt.render()),
                    ));
                }
                let ok = if op.is_ordering() {
                    matches!(lt, Type::Int | Type::Float)
                } else {
                    matches!(lt, Type::Int | Type::Float | Type::Bool | Type::Str)
                };
                if !ok {
                    return Err(terr(
                        e.line,
                        format!("comparison not defined on {}", lt.render()),
                    ));
                }
                Type::Bool
            }
            Expr::BoolChain(_, l, r) => {
                for side in [l, r] {
                    let t = self.check_expr(side, scope, flow)?;
                    if t != Type::Bool {
                        return Err(terr(
                            e.line,
                            format!("'and'/'or' need bool operands, got {}", t.render()),
                        ));
                    }
                }
                Type::Bool
            }
            Expr::Index(base, idx) => {
                let bt = self.check_expr(base, scope, flow)?;
                let it = self.check_expr(idx, scope, flow)?;
                let Type::List(elem) = bt else {
                    return Err(terr(e.line, format!("cannot index {}", bt.render())));
                };
                if it != Type::Int {
                    return Err(terr(e.line, "list index must be int"));
                }
                *elem
            }
            Expr::Field(base, field) => {
                let bt = self.check_expr(base, scope, flow)?;
                let Type::Record(rt) = bt else {
                    return Err(terr(e.line, format!("{} has no fields", bt.render())));
                };
                let Some((idx, fty)) = rt.field(field) else {
                    return Err(terr(
                        e.line,
                        format!("{} has no field '{field}'", rt.name()),
                    ));
                };
                self.res[e.id as usize] = Some(Res::Field(idx as u16));
                fty
            }
            Expr::Call(name, args) => self.check_call(e, name, args, scope, flow)?,
            Expr::ListDisplay(items) => {
                if items.is_empty() {
                    return Err(terr(
                        e.line,
                        "cannot infer the element type of an empty list \
                         (annotate the assignment: xs: list[T] = [])",
                    ));
                }
                let first = self.check_expr(&items[0], scope, flow)?;
                for item in &items[1..] {
                    let t = self.check_expr(item, scope, flow)?;
                    if t != first {
                        return Err(terr(
                            item.line,
                            format!(
                                "list elements must share one type: {} vs {}",
                                first.render(),
                                t.render()
                            ),
                        ));
                    }
                }
                Type::List(Box::new(first))
            }
        };
        self.expr_ty[e.id as usize] = Some(ty.clone());
        Ok(ty)
    }

    fn check_call(
        &mut self,
        e: &ExprNode,
        name: &str,
        args: &[ExprNode],
        scope: &mut FnScope,
        flow: &mut Flow,
    ) -> Result<Type, PsError> {
        let mut arg_tys = Vec::with_capacity(args.len());
        for a in args {
            arg_tys.push(self.check_expr(a, scope, flow)?);
        }
        let arity = |n: usize| -> Result<(), PsError> {
            if args.len() != n {
                Err(terr(
                    e.line,
                    format!("'{name}' expects {n} argument(s), got {}", args.len()),
                ))
            } else {
                Ok(())
            }
        };
        let builtin = |ck: &mut Self, b: Builtin, ty: Type| {
            ck.res[e.id as usize] = Some(Res::CallBuiltin(b));
            Ok(ty)
        };
        match name {
            "len" => {
                arity(1)?;
                if !matches!(arg_tys[0], Type::List(_)) {
                    return Err(terr(e.line, "len() needs a list"));
                }
                builtin(self, Builtin::Len, Type::Int)
            }
            "abs" => {
                arity(1)?;
                match arg_tys[0] {
                    Type::Int => builtin(self, Builtin::AbsInt, Type::Int),
                    Type::Float => builtin(self, Builtin::AbsFloat, Type::Float),
                    _ => Err(terr(e.line, "abs() needs int or float")),
                }
            }
            "min" | "max" => {
                arity(2)?;
                if arg_tys[0] != arg_tys[1] {
                    return Err(terr(
                        e.line,
                        format!("'{name}' needs two arguments of the same numeric type"),
                    ));
                }
                let b = match (name, &arg_tys[0]) {
                    ("min", Type::Int) => Builtin::MinInt,
                    ("min", Type::Float) => Builtin::MinFloat,
                    ("max", Type::Int) => Builtin::MaxInt,
                    ("max", Type::Float) => Builtin::MaxFloat,
                    _ => return Err(terr(e.line, format!("'{name}' needs int or float"))),
                };
                let t = arg_tys[0].clone();
                builtin(self, b, t)
            }
            "sqrt" => {
                arity(1)?;
                if arg_tys[0] != Type::Float {
                    return Err(terr(e.line, "sqrt() needs a float"));
                }
                builtin(self, Builtin::Sqrt, Type::Float)
            }
            "float" => {
                arity(1)?;
                if arg_tys[0] != Type::Int {
                    return Err(terr(e.line, "float() converts int to float"));
                }
                builtin(self, Builtin::CastFloat, Type::Float)
            }
            "int" => {
                arity(1)?;
                if arg_tys[0] != Type::Float {
                    return Err(terr(e.line, "int() converts float to int"));
                }
                builtin(self, Builtin::CastInt, Type::Int)
            }
            "move_pod" => {
                arity(2)?;
                if arg_tys[0] != Type::Int || arg_tys[1] != Type::Int {
                    return Err(terr(e.line, "move_pod(pod_id, dest) needs two ints"));
                }
                builtin(self, Builtin::MovePod, Type::Record(RecordType::Move))
            }
            "wake" => {
                arity(1)?;
                if arg_tys[0] != Type::Int {
                    return Err(terr(e.line, "wake(node_id) needs an int"));
                }
                builtin(self, Builtin::Wake, Type::Record(RecordType::Move))
            }
            _ => {
                if let Some(&fi) = self.callable.get(name) {
                    let f = &self.module.funcs[fi];
                    if args.len() != f.params.len() {
                        return Err(terr(
                            e.line,
                            format!(
                                "'{name}' expects {} argument(s), got {}",
                                f.params.len(),
                                args.len()
                            ),
                        ));
                    }
                    for (i, ((pname, pty), aty)) in f.params.iter().zip(arg_tys.iter()).enumerate()
                    {
                        if pty != aty {
                            return Err(terr(
                                e.line,
                                format!(
                                    "argument {} ('{pname}') of '{name}' must be {}, got {}",
                                    i + 1,
                                    pty.render(),
                                    aty.render()
                                ),
                            ));
                        }
                    }
                    let ret = f.ret.clone();
                    self.res[e.id as usize] = Some(Res::CallUser(fi as u32));
                    self.cur_max_callee = self.cur_max_callee.max(self.fn_depth[fi]);
                    Ok(ret)
                } else if self.all_funcs.contains(name) {
                    Err(terr(
                        e.line,
                        format!(
                            "cannot call '{name}' at or before its definition \
                             (recursion is not expressible)"
                        ),
                    ))
                } else {
                    Err(terr(e.line, format!("unknown function '{name}'")))
                }
            }
        }
    }
}

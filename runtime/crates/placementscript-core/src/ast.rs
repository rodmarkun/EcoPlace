//! Abstract syntax tree, types, and record schemas.
//!
//! Node ids are dense, assigned in parse (pre-)order over a single counter shared by
//! expressions and statements — the AST-level API (TODO 1.8) addresses subtrees by id.
//! Spans are byte offsets into the original source, which is what makes source-splicing
//! crossover and numeric-literal patching exact.

/// Dense node identifier (parse pre-order).
pub type NodeId = u32;

/// Byte-offset half-open span `[start, end)` into the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

/// Static types. PlacementScript has no `None`, unions, or user-defined types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// 64-bit signed integer with checked arithmetic.
    Int,
    /// IEEE-754 binary64 restricted to finite values.
    Float,
    /// Boolean (not an integer).
    Bool,
    /// Immutable string; equality only.
    Str,
    /// Homogeneous list, mutable only via `append`.
    List(Box<Type>),
    /// Host-defined record type.
    Record(RecordType),
}

impl Type {
    /// Render as PlacementScript surface syntax (e.g. `list[float]`).
    pub fn render(&self) -> String {
        match self {
            Type::Int => "int".into(),
            Type::Float => "float".into(),
            Type::Bool => "bool".into(),
            Type::Str => "str".into(),
            Type::List(t) => format!("list[{}]", t.render()),
            Type::Record(r) => r.name().into(),
        }
    }
}

/// The six record types of spec §4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    /// A pod (service replica) awaiting or holding placement.
    Pod,
    /// Per-node capacity/utilization/power state.
    NodeState,
    /// Cluster-wide call graph as a columnar edge list.
    CallGraphCtx,
    /// Scoring context: nodes, pods, graph, scenario clock.
    ClusterCtx,
    /// Proposer state: nodes and pods.
    ClusterState,
    /// A proposed action: move a pod or wake a node.
    Move,
}

impl RecordType {
    /// Surface-syntax name.
    pub fn name(&self) -> &'static str {
        match self {
            RecordType::Pod => "Pod",
            RecordType::NodeState => "NodeState",
            RecordType::CallGraphCtx => "CallGraphCtx",
            RecordType::ClusterCtx => "ClusterCtx",
            RecordType::ClusterState => "ClusterState",
            RecordType::Move => "Move",
        }
    }

    /// Look a record type up by surface name.
    pub fn by_name(name: &str) -> Option<RecordType> {
        Some(match name {
            "Pod" => RecordType::Pod,
            "NodeState" => RecordType::NodeState,
            "CallGraphCtx" => RecordType::CallGraphCtx,
            "ClusterCtx" => RecordType::ClusterCtx,
            "ClusterState" => RecordType::ClusterState,
            "Move" => RecordType::Move,
            _ => return None,
        })
    }

    /// Field schema in declaration order (spec §4). Field index = position here.
    pub fn fields(&self) -> Vec<(&'static str, Type)> {
        use RecordType as R;
        use Type as T;
        match self {
            R::Pod => vec![
                ("id", T::Int),
                ("service", T::Int),
                ("app", T::Int),
                ("cpu_request", T::Int),
                ("mem_request", T::Int),
                ("stateful", T::Bool),
                ("has_pdb", T::Bool),
                ("image_mb", T::Int),
                ("current_node", T::Int),
            ],
            R::NodeState => vec![
                ("id", T::Int),
                ("tier", T::Int),
                ("cpu_capacity", T::Int),
                ("cpu_allocated", T::Int),
                ("mem_capacity", T::Int),
                ("mem_allocated", T::Int),
                ("power_idle_w", T::Float),
                ("power_max_w", T::Float),
                ("suspended", T::Bool),
                ("suspendable", T::Bool),
                ("latency_to_ms", T::List(Box::new(T::Float))),
            ],
            R::CallGraphCtx => vec![
                ("src", T::List(Box::new(T::Int))),
                ("dst", T::List(Box::new(T::Int))),
                ("rate", T::List(Box::new(T::Float))),
                ("payload_kb", T::List(Box::new(T::Float))),
            ],
            R::ClusterCtx => vec![
                ("nodes", T::List(Box::new(T::Record(R::NodeState)))),
                ("pods", T::List(Box::new(T::Record(R::Pod)))),
                ("graph", T::Record(R::CallGraphCtx)),
                ("now_ms", T::Int),
            ],
            R::ClusterState => vec![
                ("nodes", T::List(Box::new(T::Record(R::NodeState)))),
                ("pods", T::List(Box::new(T::Record(R::Pod)))),
            ],
            R::Move => vec![("kind", T::Int), ("pod", T::Int), ("dest", T::Int)],
        }
    }

    /// Index and type of `field`, if it exists.
    pub fn field(&self, name: &str) -> Option<(usize, Type)> {
        self.fields()
            .into_iter()
            .enumerate()
            .find(|(_, (n, _))| *n == name)
            .map(|(i, (_, t))| (i, t))
    }
}

/// Binary arithmetic operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (float only)
    Div,
    /// `//` (int only, floor)
    FloorDiv,
    /// `%` (int only, Python sign)
    Mod,
}

impl BinOp {
    /// Surface token.
    pub fn token(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
        }
    }
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    Eq,
    /// `!=`
    Ne,
}

impl CmpOp {
    /// True for `< <= > >=` (ordering), false for `== !=` (equality).
    pub fn is_ordering(&self) -> bool {
        !matches!(self, CmpOp::Eq | CmpOp::Ne)
    }
}

/// Short-circuit boolean operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// `and`
    And,
    /// `or`
    Or,
}

/// Expression node with identity, location, and kind.
#[derive(Debug, Clone)]
pub struct ExprNode {
    /// Dense node id (parse pre-order).
    pub id: NodeId,
    /// Byte span in the source.
    pub span: Span,
    /// 1-based source line of the first token.
    pub line: u32,
    /// The expression itself.
    pub kind: Expr,
}

/// Expression kinds.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Integer literal (always non-negative; `-` is unary negation).
    Int(i64),
    /// Float literal (always finite and non-negative).
    Float(f64),
    /// String literal.
    Str(String),
    /// `True` / `False`.
    Bool(bool),
    /// Variable or module-constant reference.
    Name(String),
    /// Unary negation `-e`.
    Neg(Box<ExprNode>),
    /// Logical `not e`.
    Not(Box<ExprNode>),
    /// Binary arithmetic.
    Binary(BinOp, Box<ExprNode>, Box<ExprNode>),
    /// Comparison (never chained).
    Compare(CmpOp, Box<ExprNode>, Box<ExprNode>),
    /// Short-circuit `and`/`or`.
    BoolChain(BoolOp, Box<ExprNode>, Box<ExprNode>),
    /// Indexing `xs[i]`.
    Index(Box<ExprNode>, Box<ExprNode>),
    /// Field access `x.field`.
    Field(Box<ExprNode>, String),
    /// Call of a builtin or earlier-defined user function.
    Call(String, Vec<ExprNode>),
    /// List display `[e1, ..., en]`.
    ListDisplay(Vec<ExprNode>),
}

/// Loop iterable.
#[derive(Debug, Clone)]
pub enum ForIter {
    /// `range(b)` / `range(a, b)`.
    Range(Option<Box<ExprNode>>, Box<ExprNode>),
    /// A `list[T]` expression.
    List(Box<ExprNode>),
}

/// Statement node with identity and location.
#[derive(Debug, Clone)]
pub struct StmtNode {
    /// Dense node id (same counter as expressions).
    pub id: NodeId,
    /// Byte span in the source (simple statements: the statement text without newline).
    pub span: Span,
    /// 1-based source line.
    pub line: u32,
    /// The statement itself.
    pub kind: Stmt,
}

/// Statement kinds.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `name [: type] = expr`
    Assign {
        /// Target variable.
        name: String,
        /// Optional annotation (required for `[]`).
        ann: Option<Type>,
        /// Right-hand side.
        value: ExprNode,
    },
    /// `name op= expr`
    AugAssign {
        /// Target variable.
        name: String,
        /// The underlying arithmetic operator.
        op: BinOp,
        /// Right-hand side.
        value: ExprNode,
    },
    /// `name.append(expr)`
    Append {
        /// Target list variable.
        name: String,
        /// Element to append.
        value: ExprNode,
    },
    /// `return expr`
    Return(ExprNode),
    /// `pass`
    Pass,
    /// `if` / `elif` / `else`. Encoded as (cond, block) arms plus optional else block.
    If {
        /// `if` and `elif` arms, in order.
        arms: Vec<(ExprNode, Vec<StmtNode>)>,
        /// `else` block.
        orelse: Option<Vec<StmtNode>>,
    },
    /// `for var in iterable:`
    For {
        /// Loop variable (scoped to the body).
        var: String,
        /// What is iterated.
        iter: ForIter,
        /// Loop body.
        body: Vec<StmtNode>,
    },
}

/// Literal value of a module-level constant.
#[derive(Debug, Clone)]
pub enum ConstLit {
    /// Integer constant.
    Int(i64),
    /// Float constant (finite).
    Float(f64),
    /// String constant.
    Str(String),
    /// Boolean constant.
    Bool(bool),
}

/// Module-level constant definition.
#[derive(Debug, Clone)]
pub struct ConstDef {
    /// Constant name.
    pub name: String,
    /// Annotated type.
    pub ty: Type,
    /// Literal value.
    pub value: ConstLit,
    /// Byte span of the literal (including an optional leading `-`) — the numeric
    /// patching site for module constants (TODO 1.8).
    pub value_span: Span,
    /// 1-based line.
    pub line: u32,
}

/// Function definition.
#[derive(Debug, Clone)]
pub struct FuncDef {
    /// Function name.
    pub name: String,
    /// Parameters (name, type) in order.
    pub params: Vec<(String, Type)>,
    /// Declared return type.
    pub ret: Type,
    /// Body statements.
    pub body: Vec<StmtNode>,
    /// Byte span of the whole definition (`def` through the last body statement) —
    /// the function-level crossover site (TODO 1.8/C19).
    pub span: Span,
    /// 1-based line of the `def`.
    pub line: u32,
}

/// A parsed module: constants and functions in source order.
#[derive(Debug, Clone)]
pub struct Module {
    /// Module-level constants.
    pub consts: Vec<ConstDef>,
    /// Function definitions.
    pub funcs: Vec<FuncDef>,
    /// Total number of AST node ids allocated (ids are `0..node_count`).
    pub node_count: u32,
}

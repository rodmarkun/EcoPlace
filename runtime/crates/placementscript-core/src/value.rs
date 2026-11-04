//! Runtime values. Single-threaded by design (`Rc`): the interpreter is embedded
//! per-invocation in the simulator and per-instance in the wasm module.

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::ast::{RecordType, Type};

/// A runtime value.
#[derive(Debug, Clone)]
pub enum Value {
    /// 64-bit signed integer.
    Int(i64),
    /// Finite IEEE-754 binary64.
    Float(f64),
    /// Boolean.
    Bool(bool),
    /// Immutable string.
    Str(Rc<str>),
    /// Homogeneous list. Policy-created lists are append-mutable; host-provided lists
    /// are *frozen* (spec §5.2: host state is read-only — appending to one traps).
    List(Rc<ListVal>),
    /// Immutable record.
    Record(Rc<RecordVal>),
}

/// A list value: the elements plus the frozen (host-origin, immutable) flag.
#[derive(Debug)]
pub struct ListVal {
    frozen: bool,
    items: RefCell<Vec<Value>>,
}

impl ListVal {
    /// Is this a frozen (host-provided) list?
    pub fn frozen(&self) -> bool {
        self.frozen
    }

    /// Borrow the elements (read-only).
    pub fn borrow(&self) -> Ref<'_, Vec<Value>> {
        self.items.borrow()
    }

    /// Append an element. Caller (the interpreter) must have rejected frozen lists.
    pub(crate) fn push(&self, v: Value) {
        debug_assert!(!self.frozen, "push on a frozen list");
        self.items.borrow_mut().push(v);
    }
}

/// A record value: type tag + fields in schema order.
#[derive(Debug)]
pub struct RecordVal {
    /// Record type.
    pub rt: RecordType,
    /// Field values, in `RecordType::fields()` order.
    pub fields: Vec<Value>,
}

impl Value {
    /// Build a (policy-side, append-mutable) list value.
    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(ListVal {
            frozen: false,
            items: RefCell::new(items),
        }))
    }

    /// Build a *frozen* (host-origin, immutable) list value. Every host boundary
    /// (PyO3, wire deserialization, conformance args) uses this constructor so that
    /// host-provided state can never be mutated through an alias (spec §5.2).
    pub fn frozen_list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(ListVal {
            frozen: true,
            items: RefCell::new(items),
        }))
    }

    /// Build a string value.
    pub fn str(s: &str) -> Value {
        Value::Str(Rc::from(s))
    }

    /// Is every list reachable from this value frozen?
    pub fn fully_frozen(&self) -> bool {
        match self {
            Value::List(xs) => xs.frozen() && xs.borrow().iter().all(Value::fully_frozen),
            Value::Record(r) => r.fields.iter().all(Value::fully_frozen),
            _ => true,
        }
    }

    /// Return an equivalent value in which every reachable list is frozen (spec §5.2:
    /// host state is read-only). Already-frozen subtrees are shared, not copied.
    /// Recursion is bounded by the record schemas (host values are shallow).
    pub fn deep_frozen(&self) -> Value {
        if self.fully_frozen() {
            return self.clone();
        }
        match self {
            Value::List(xs) => {
                Value::frozen_list(xs.borrow().iter().map(Value::deep_frozen).collect())
            }
            Value::Record(r) => Value::Record(Rc::new(RecordVal {
                rt: r.rt,
                fields: r.fields.iter().map(Value::deep_frozen).collect(),
            })),
            v => v.clone(),
        }
    }

    /// Build and validate a record value against its schema (host construction path).
    /// Checks arity, field types, and float finiteness recursively. All contained
    /// lists are **deep-frozen** on construction: this constructor exists for host
    /// inputs, and host state must be immutable to the policy no matter how the host
    /// built the field values (the interpreter constructs its records directly, never
    /// through here).
    pub fn record(rt: RecordType, fields: Vec<Value>) -> Result<Value, String> {
        let fields: Vec<Value> = fields.iter().map(Value::deep_frozen).collect();
        let schema = rt.fields();
        if fields.len() != schema.len() {
            return Err(format!(
                "{} expects {} fields, got {}",
                rt.name(),
                schema.len(),
                fields.len()
            ));
        }
        for (v, (fname, fty)) in fields.iter().zip(schema.iter()) {
            if !v.conforms(fty) {
                return Err(format!(
                    "{}.{} must be {}, got {}",
                    rt.name(),
                    fname,
                    fty.render(),
                    v.type_name()
                ));
            }
        }
        Ok(Value::Record(Rc::new(RecordVal { rt, fields })))
    }

    /// Does this value conform to `ty` (recursively), with all floats finite?
    pub fn conforms(&self, ty: &Type) -> bool {
        match (self, ty) {
            (Value::Int(_), Type::Int) => true,
            (Value::Float(f), Type::Float) => f.is_finite(),
            (Value::Bool(_), Type::Bool) => true,
            (Value::Str(_), Type::Str) => true,
            (Value::List(xs), Type::List(t)) => xs.borrow().iter().all(|v| v.conforms(t)),
            (Value::Record(r), Type::Record(rt)) if r.rt == *rt => r
                .fields
                .iter()
                .zip(r.rt.fields().iter())
                .all(|(v, (_, ft))| v.conforms(ft)),
            _ => false,
        }
    }

    /// Short type description for error messages.
    pub fn type_name(&self) -> String {
        match self {
            Value::Int(_) => "int".into(),
            Value::Float(_) => "float".into(),
            Value::Bool(_) => "bool".into(),
            Value::Str(_) => "str".into(),
            Value::List(_) => "list".into(),
            Value::Record(r) => r.rt.name().into(),
        }
    }

    /// Bit-exact structural equality (floats compared by bit pattern; used by the
    /// conformance suite, where `-0.0 != 0.0` on purpose). The frozen flag is not
    /// part of value identity.
    pub fn bit_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => {
                let (a, b) = (a.borrow(), b.borrow());
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.bit_eq(y))
            }
            (Value::Record(a), Value::Record(b)) => {
                a.rt == b.rt
                    && a.fields
                        .iter()
                        .zip(b.fields.iter())
                        .all(|(x, y)| x.bit_eq(y))
            }
            _ => false,
        }
    }
}

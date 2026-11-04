//! Python bindings for `placementscript-core` (TODO 1.3, 1.8).
//!
//! The simulator-side embedding: `Policy` compiles + bundle-validates a PlacementScript
//! source and exposes the batch-evaluation API (`score_batch` scores one pod against N
//! nodes per boundary crossing — the Python↔Rust cost is paid per *batch*, per C18/1.3).
//! Records cross the boundary as plain dicts mirroring spec §4 (converted and validated
//! once per batch). The AST-level API (subtree enumeration, crossover, numeric patching)
//! is exposed for the same-DSL GP baseline (C19) and ShinkaEvolve numeric mutation.
//!
//! Error taxonomy (spec §9) maps to exception classes:
//! `ParseError`, `TypeCheckError`, `TrapError`, `FuelExhausted`, `OutOfMemory`
//! (all subclasses of `PlacementScriptError`), and `HostError` for host-boundary faults.

use placementscript_core as core;
use placementscript_core::ast::{RecordType, Type};
use placementscript_core::value::Value;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList};

create_exception!(placementscript, PlacementScriptError, PyException);
create_exception!(placementscript, ParseError, PlacementScriptError);
create_exception!(placementscript, TypeCheckError, PlacementScriptError);
create_exception!(placementscript, TrapError, PlacementScriptError);
create_exception!(placementscript, FuelExhausted, PlacementScriptError);
create_exception!(placementscript, OutOfMemory, PlacementScriptError);
create_exception!(placementscript, HostError, PyException);

fn ps_err(e: core::PsError) -> PyErr {
    let msg = e.to_string();
    match e {
        core::PsError::Parse { .. } => ParseError::new_err(msg),
        core::PsError::Type { .. } => TypeCheckError::new_err(msg),
        core::PsError::Trap { .. } => TrapError::new_err(msg),
        core::PsError::FuelExhausted => FuelExhausted::new_err(msg),
        core::PsError::OutOfMemory => OutOfMemory::new_err(msg),
    }
}

fn call_err(e: core::CallError) -> PyErr {
    match e {
        core::CallError::Ps(e) => ps_err(e),
        core::CallError::Host(e) => HostError::new_err(e.to_string()),
    }
}

fn py_type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into())
}

/// Type-directed Python → Value conversion (records are dicts keyed by field name).
///
/// Extraction is **exact** and every failure is a `placementscript.HostError` (spec §9:
/// malformed host input is a host-boundary fault, never the policy's): int fields
/// require a Python `int` in i64 range (`bool` rejected; overflow mapped, not leaked),
/// float fields require a Python `float` (no implicit int/bool coercion), bool fields
/// require a `bool`.
fn py_to_value(obj: &Bound<'_, PyAny>, ty: &Type) -> PyResult<Value> {
    match ty {
        Type::Int => {
            if obj.is_instance_of::<PyBool>() {
                return Err(HostError::new_err("expected int, got bool"));
            }
            let v = obj.extract::<i64>().map_err(|_| {
                HostError::new_err(format!(
                    "expected int in i64 range, got {}",
                    py_type_name(obj)
                ))
            })?;
            Ok(Value::Int(v))
        }
        Type::Float => {
            let f = obj.downcast::<PyFloat>().map_err(|_| {
                HostError::new_err(format!(
                    "expected float, got {} (no implicit conversion at the host boundary)",
                    py_type_name(obj)
                ))
            })?;
            let v = f.value();
            if !v.is_finite() {
                return Err(HostError::new_err("non-finite float in input state"));
            }
            Ok(Value::Float(v))
        }
        Type::Bool => {
            let b = obj.downcast::<PyBool>().map_err(|_| {
                HostError::new_err(format!("expected bool, got {}", py_type_name(obj)))
            })?;
            Ok(Value::Bool(b.is_true()))
        }
        Type::Str => {
            let s = obj.extract::<String>().map_err(|_| {
                HostError::new_err(format!("expected str, got {}", py_type_name(obj)))
            })?;
            Ok(Value::str(&s))
        }
        Type::List(elem) => {
            let list = obj
                .downcast::<PyList>()
                .map_err(|_| HostError::new_err(format!("expected list[{}]", elem.render())))?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(py_to_value(&item, elem)?);
            }
            // Host boundary: frozen (spec §5.2) — policies cannot mutate host state.
            Ok(Value::frozen_list(out))
        }
        Type::Record(rt) => {
            let dict = obj.downcast::<PyDict>().map_err(|_| {
                HostError::new_err(format!("expected a dict for record {}", rt.name()))
            })?;
            let mut fields = Vec::new();
            for (fname, fty) in rt.fields() {
                let Some(v) = dict.get_item(fname)? else {
                    return Err(HostError::new_err(format!(
                        "record {} is missing field '{fname}'",
                        rt.name()
                    )));
                };
                fields.push(py_to_value(&v, &fty)?);
            }
            Value::record(*rt, fields).map_err(HostError::new_err)
        }
    }
}

fn record_arg(obj: &Bound<'_, PyAny>, rt: RecordType) -> PyResult<Value> {
    py_to_value(obj, &Type::Record(rt))
}

/// Exact scalar extraction for non-record host parameters (`budget`, limit knobs).
/// PyO3's automatic extraction would leak raw `OverflowError`/`TypeError` for
/// out-of-range or wrong-typed values; spec §9 requires every host-boundary fault
/// to surface as `placementscript.HostError` (same contract as record fields).
fn host_i64(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<i64> {
    if obj.is_instance_of::<PyBool>() {
        return Err(HostError::new_err(format!(
            "{what}: expected int, got bool"
        )));
    }
    obj.extract::<i64>().map_err(|_| {
        HostError::new_err(format!(
            "{what}: expected int in i64 range, got {}",
            py_type_name(obj)
        ))
    })
}

fn host_u64_opt(obj: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Option<u64>> {
    obj.map(|o| {
        if o.is_instance_of::<PyBool>() {
            return Err(HostError::new_err(format!(
                "{what}: expected int, got bool"
            )));
        }
        o.extract::<u64>().map_err(|_| {
            HostError::new_err(format!(
                "{what}: expected non-negative int in u64 range, got {}",
                py_type_name(o)
            ))
        })
    })
    .transpose()
}

fn limits(
    fuel: Option<&Bound<'_, PyAny>>,
    memory: Option<&Bound<'_, PyAny>>,
    default: core::Limits,
) -> PyResult<core::Limits> {
    Ok(core::Limits {
        fuel: host_u64_opt(fuel, "fuel")?.unwrap_or(default.fuel),
        memory_bytes: host_u64_opt(memory, "memory")?.unwrap_or(default.memory_bytes),
    })
}

/// A compiled, bundle-validated PlacementScript policy.
#[pyclass(unsendable, module = "placementscript")]
struct Policy {
    prog: core::Program,
    meta: core::bundle::BundleMeta,
}

#[pymethods]
impl Policy {
    #[new]
    fn new(source: &str) -> PyResult<Self> {
        let prog = core::compile(source).map_err(ps_err)?;
        let meta = core::bundle::validate(&prog).map_err(ps_err)?;
        Ok(Policy { prog, meta })
    }

    /// `POLICY_NAME` metadata.
    #[getter]
    fn name(&self) -> String {
        self.meta.name.clone()
    }

    /// `POLICY_VERSION` metadata.
    #[getter]
    fn version(&self) -> String {
        self.meta.version.clone()
    }

    /// Original source text.
    #[getter]
    fn source(&self) -> String {
        self.prog.source().to_string()
    }

    /// Score one (pod, node) pair. Returns a finite float (`-0.0` normalized).
    #[pyo3(signature = (pod, node, ctx, *, fuel=None, memory=None))]
    fn score(
        &self,
        pod: &Bound<'_, PyAny>,
        node: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        fuel: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let pod = record_arg(pod, RecordType::Pod)?;
        let node = record_arg(node, RecordType::NodeState)?;
        let ctx = record_arg(ctx, RecordType::ClusterCtx)?;
        let lim = limits(fuel, memory, core::Limits::default_score())?;
        let (v, _) = core::bundle::score(&self.prog, &pod, &node, &ctx, lim).map_err(call_err)?;
        Ok(v)
    }

    /// Score one pod against N candidate nodes in one boundary crossing (TODO 1.3).
    /// `fuel`/`memory` are per-node budgets (spec §8). Returns a list of N floats.
    #[pyo3(signature = (pod, nodes, ctx, *, fuel=None, memory=None))]
    fn score_batch(
        &self,
        pod: &Bound<'_, PyAny>,
        nodes: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        fuel: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        let pod = record_arg(pod, RecordType::Pod)?;
        let ctx = record_arg(ctx, RecordType::ClusterCtx)?;
        let nodes_list = nodes
            .downcast::<PyList>()
            .map_err(|_| HostError::new_err("nodes must be a list of NodeState dicts"))?;
        let mut node_vals = Vec::with_capacity(nodes_list.len());
        for n in nodes_list.iter() {
            node_vals.push(record_arg(&n, RecordType::NodeState)?);
        }
        let lim = limits(fuel, memory, core::Limits::default_score())?;
        let (scores, _) = core::bundle::score_batch(&self.prog, &pod, &node_vals, &ctx, lim)
            .map_err(|(i, e)| {
                let base = call_err(e);
                Python::with_gil(|py| {
                    let msg = format!("node index {i}: {}", base.value(py));
                    PyErr::from_type(base.get_type(py), msg)
                })
            })?;
        Ok(scores)
    }

    /// Score a batch and expose deterministic interpreter cost for Phase-5 archive
    /// criteria. The cost is the same fuel/memory accounting already enforced by the
    /// normative interpreter; wall-clock timing is deliberately not returned here.
    #[pyo3(signature = (pod, nodes, ctx, *, fuel=None, memory=None))]
    fn score_batch_cost(
        &self,
        pod: &Bound<'_, PyAny>,
        nodes: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        fuel: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<f64>, u64, u64)> {
        let pod = record_arg(pod, RecordType::Pod)?;
        let ctx = record_arg(ctx, RecordType::ClusterCtx)?;
        let nodes_list = nodes
            .downcast::<PyList>()
            .map_err(|_| HostError::new_err("nodes must be a list of NodeState dicts"))?;
        let mut node_vals = Vec::with_capacity(nodes_list.len());
        for n in nodes_list.iter() {
            node_vals.push(record_arg(&n, RecordType::NodeState)?);
        }
        let lim = limits(fuel, memory, core::Limits::default_score())?;
        let (scores, cost) = core::bundle::score_batch(&self.prog, &pod, &node_vals, &ctx, lim)
            .map_err(|(i, e)| {
                let base = call_err(e);
                Python::with_gil(|py| {
                    let msg = format!("node index {i}: {}", base.value(py));
                    PyErr::from_type(base.get_type(py), msg)
                })
            })?;
        Ok((scores, cost.fuel_used, cost.mem_used))
    }

    /// Score a batch and apply the normative host tie-break (spec §5.5: highest score;
    /// bitwise ties → lowest node id) via the single shared core selection function.
    /// Returns `(best_node_id_or_None, scores)`.
    #[pyo3(signature = (pod, nodes, ctx, *, fuel=None, memory=None))]
    fn select_node(
        &self,
        pod: &Bound<'_, PyAny>,
        nodes: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        fuel: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Option<i64>, Vec<f64>)> {
        let pod_v = record_arg(pod, RecordType::Pod)?;
        let ctx_v = record_arg(ctx, RecordType::ClusterCtx)?;
        let nodes_list = nodes
            .downcast::<PyList>()
            .map_err(|_| HostError::new_err("nodes must be a list of NodeState dicts"))?;
        let mut node_vals = Vec::with_capacity(nodes_list.len());
        for n in nodes_list.iter() {
            node_vals.push(record_arg(&n, RecordType::NodeState)?);
        }
        let lim = limits(fuel, memory, core::Limits::default_score())?;
        let (scores, _) = core::bundle::score_batch(&self.prog, &pod_v, &node_vals, &ctx_v, lim)
            .map_err(|(i, e)| {
                let base = call_err(e);
                Python::with_gil(|py| {
                    let msg = format!("node index {i}: {}", base.value(py));
                    PyErr::from_type(base.get_type(py), msg)
                })
            })?;
        let best = core::bundle::select_node(&scores, &node_vals);
        Ok((best, scores))
    }

    /// Run `propose_moves`. Returns a list of `{"kind", "pod", "dest"}` dicts
    /// (kind 0 = move_pod, 1 = wake).
    #[pyo3(signature = (state, graph, budget, *, fuel=None, memory=None))]
    fn propose_moves(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        graph: &Bound<'_, PyAny>,
        budget: &Bound<'_, PyAny>,
        fuel: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let state = record_arg(state, RecordType::ClusterState)?;
        let graph = record_arg(graph, RecordType::CallGraphCtx)?;
        let budget = host_i64(budget, "budget")?;
        let lim = limits(fuel, memory, core::Limits::default_propose())?;
        let (moves, _) = core::bundle::propose_moves(&self.prog, &state, &graph, budget, lim)
            .map_err(call_err)?;
        let out = PyList::empty(py);
        for m in moves {
            let d = PyDict::new(py);
            d.set_item("kind", m.kind)?;
            d.set_item("pod", m.pod)?;
            d.set_item("dest", m.dest)?;
            out.append(d)?;
        }
        Ok(out.into())
    }
}

/// Parse + type-check only (no bundle requirement). Raises on failure.
#[pyfunction]
fn compile_check(source: &str) -> PyResult<()> {
    core::check_program(source).map_err(ps_err)
}

fn subtree_dict<'py>(py: Python<'py>, s: &core::astapi::Subtree) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", s.id)?;
    d.set_item("start", s.start)?;
    d.set_item("end", s.end)?;
    d.set_item("line", s.line)?;
    d.set_item("kind", s.kind)?;
    d.set_item("type", s.ty.clone())?;
    d.set_item("func", s.func.clone())?;
    d.set_item("is_expr", s.is_expr)?;
    d.set_item("parent", s.parent)?;
    d.set_item("depth", s.depth)?;
    Ok(d)
}

/// Enumerate crossover/mutation sites: typed expressions, simple and block statements,
/// and whole function definitions (with `parent`/`depth` tree paths).
#[pyfunction]
fn enumerate_subtrees(py: Python<'_>, source: &str) -> PyResult<Py<PyList>> {
    let sites = core::astapi::enumerate_subtrees(source).map_err(ps_err)?;
    let out = PyList::empty(py);
    for s in &sites {
        out.append(subtree_dict(py, s)?)?;
    }
    Ok(out.into())
}

/// Enumerate numeric literals for the numeric-patching operator.
#[pyfunction]
fn enumerate_numeric_literals(py: Python<'_>, source: &str) -> PyResult<Py<PyList>> {
    let lits = core::astapi::enumerate_numeric_literals(source).map_err(ps_err)?;
    let out = PyList::empty(py);
    for l in &lits {
        let d = PyDict::new(py);
        d.set_item("id", l.id)?;
        d.set_item("start", l.start)?;
        d.set_item("end", l.end)?;
        d.set_item("line", l.line)?;
        d.set_item("is_float", l.is_float)?;
        d.set_item("is_const", l.is_const)?;
        d.set_item("text", l.text.clone())?;
        out.append(d)?;
    }
    Ok(out.into())
}

/// Apply `{node_id: new_literal_text}` numeric patches; returns the new source.
#[pyfunction]
fn patch_numeric_literals(
    source: &str,
    patches: std::collections::BTreeMap<u32, String>,
) -> PyResult<String> {
    let patches: Vec<(u32, String)> = patches.into_iter().collect();
    core::astapi::patch_numeric_literals(source, &patches).map_err(ps_err)
}

/// Typed subtree crossover: replace subtree `id_a` of `src_a` with subtree `id_b` of
/// `src_b`; returns the offspring source (re-compiled) or raises (invalid offspring).
#[pyfunction]
fn crossover(src_a: &str, src_b: &str, id_a: u32, id_b: u32) -> PyResult<String> {
    core::astapi::crossover(src_a, src_b, id_a, id_b).map_err(ps_err)
}

/// Mutation hook: replace subtree `id` with `new_text` (re-compiled before returning).
#[pyfunction]
fn replace_subtree(source: &str, id: u32, new_text: &str) -> PyResult<String> {
    core::astapi::replace_subtree(source, id, new_text).map_err(ps_err)
}

#[pymodule]
fn placementscript(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Policy>()?;
    m.add_function(wrap_pyfunction!(compile_check, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_subtrees, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_numeric_literals, m)?)?;
    m.add_function(wrap_pyfunction!(patch_numeric_literals, m)?)?;
    m.add_function(wrap_pyfunction!(crossover, m)?)?;
    m.add_function(wrap_pyfunction!(replace_subtree, m)?)?;
    m.add(
        "PlacementScriptError",
        py.get_type::<PlacementScriptError>(),
    )?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("TypeCheckError", py.get_type::<TypeCheckError>())?;
    m.add("TrapError", py.get_type::<TrapError>())?;
    m.add("FuelExhausted", py.get_type::<FuelExhausted>())?;
    m.add("OutOfMemory", py.get_type::<OutOfMemory>())?;
    m.add("HostError", py.get_type::<HostError>())?;
    m.add("DEFAULT_SCORE_FUEL", core::Limits::default_score().fuel)?;
    m.add(
        "DEFAULT_SCORE_MEMORY",
        core::Limits::default_score().memory_bytes,
    )?;
    m.add("DEFAULT_PROPOSE_FUEL", core::Limits::default_propose().fuel)?;
    m.add(
        "DEFAULT_PROPOSE_MEMORY",
        core::Limits::default_propose().memory_bytes,
    )?;
    m.add("__doc__", "PlacementScript DSL: native embedding of the single Rust interpreter (C18). See docs/placementscript-spec.md.")?;
    Ok(())
}

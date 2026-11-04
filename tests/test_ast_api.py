"""TODO 1.8 verification from Python: the AST-level API that the same-DSL GP baseline
(C19) and ShinkaEvolve's numeric-patching operator (5.8) consume."""

import random
from pathlib import Path

import pytest

ps = pytest.importorskip("placementscript")

POLICIES = Path(__file__).resolve().parent.parent / "runtime" / "policies"


def src(name: str) -> str:
    return (POLICIES / f"{name}.ps").read_text()


class TestSubtreeEnumeration:
    def test_typed_and_deterministic(self):
        sites = ps.enumerate_subtrees(src("least_allocated"))
        assert sites == ps.enumerate_subtrees(src("least_allocated"))
        exprs = [s for s in sites if s["is_expr"]]
        stmts = [s for s in sites if not s["is_expr"]]
        assert exprs and stmts
        assert all(s["type"] is not None for s in exprs)
        assert all(s["type"] is None for s in stmts)
        # Every site addresses a real span of the source.
        text = src("least_allocated")
        for s in sites[:50]:
            assert 0 <= s["start"] < s["end"] <= len(text.encode())

    def test_invalid_source_raises_structured_error(self):
        with pytest.raises(ps.ParseError):
            ps.enumerate_subtrees("def broken(")


class TestGPCrossover:
    """The GP loop's shape: pick same-typed expression sites in two parents, splice,
    treat compile failures as invalid offspring."""

    def test_seeded_crossover_between_seed_policies(self):
        rng = random.Random(20260713)  # explicit seed per docs/SEEDS.md
        a, b = src("least_allocated"), src("energy_greedy")
        sites_a = [s for s in ps.enumerate_subtrees(a) if s["is_expr"] and s["func"] == "score"]
        sites_b = [s for s in ps.enumerate_subtrees(b) if s["is_expr"] and s["func"] == "score"]
        produced = 0
        invalid = 0
        for _ in range(60):
            sa = rng.choice(sites_a)
            compatible = [s for s in sites_b if s["type"] == sa["type"]]
            if not compatible:
                continue
            sb = rng.choice(compatible)
            try:
                child = ps.crossover(a, b, sa["id"], sb["id"])
            except ps.PlacementScriptError:
                invalid += 1  # e.g. donor references out-of-scope names
                continue
            produced += 1
            ps.Policy(child)  # offspring is a loadable bundle
        assert produced >= 5, (
            f"crossover produced only {produced} valid offspring ({invalid} invalid)"
        )

    def test_type_mismatch_rejected(self):
        a = src("least_allocated")
        sites = ps.enumerate_subtrees(a)
        int_site = next(s for s in sites if s["type"] == "int")
        float_site = next(s for s in sites if s["type"] == "float")
        with pytest.raises(ps.TypeCheckError, match="type mismatch"):
            ps.crossover(a, a, int_site["id"], float_site["id"])

    def test_point_mutation_hook(self):
        a = src("most_allocated")
        sites = ps.enumerate_subtrees(a)
        site = next(
            s for s in sites if s["type"] == "float" and s["func"] == "score" and s["is_expr"]
        )
        child = ps.replace_subtree(a, site["id"], "min(1.0, used_fraction_after(pod, node) * 2.0)")
        ps.Policy(child)
        with pytest.raises(ps.PlacementScriptError):
            ps.replace_subtree(a, site["id"], "unknown_helper(pod)")


class TestNumericPatching:
    def test_enumerate_and_patch(self):
        a = src("least_allocated")
        lits = ps.enumerate_numeric_literals(a)
        assert lits, "seed policy has numeric literals"
        floats = [x for x in lits if x["is_float"]]
        target = next(x for x in floats if x["text"] == "0.25")  # imbalance threshold
        patched = ps.patch_numeric_literals(a, {target["id"]: "0.4"})
        assert "0.4" in patched and "0.25" not in patched
        ps.Policy(patched)

    def test_multi_patch_shinka_style(self):
        """ShinkaEvolve numeric mutation: perturb several literals at once."""
        rng = random.Random(42)
        a = src("energy_greedy")
        lits = [x for x in ps.enumerate_numeric_literals(a) if x["is_float"]]
        patches = {}
        for lit in rng.sample(lits, min(3, len(lits))):
            factor = rng.choice([0.5, 0.9, 1.1, 2.0])
            patches[lit["id"]] = repr(float(lit["text"]) * factor)
        patched = ps.patch_numeric_literals(a, patches)
        ps.Policy(patched)
        # Numeric patching must never change program structure.
        assert len(ps.enumerate_subtrees(patched)) == len(ps.enumerate_subtrees(a))

    def test_kind_and_sign_guards(self):
        a = src("least_allocated")
        lits = ps.enumerate_numeric_literals(a)
        f = next(x for x in lits if x["is_float"])
        i = next(x for x in lits if not x["is_float"])
        with pytest.raises(ps.TypeCheckError):
            ps.patch_numeric_literals(a, {f["id"]: "3"})  # float slot, int text
        with pytest.raises(ps.TypeCheckError):
            ps.patch_numeric_literals(a, {i["id"]: "-2"})  # sign lives in the AST
        with pytest.raises(ps.TypeCheckError):
            ps.patch_numeric_literals(a, {f["id"]: "1e999"})  # non-finite


class TestFullGrammarSites:
    """Round-1 review R1.10: every grammar node is a GP site (C19)."""

    def test_block_and_function_sites_with_tree_paths(self):
        sites = ps.enumerate_subtrees(src("least_allocated"))
        kinds = {s["kind"] for s in sites}
        assert {"if", "for", "func"} <= kinds
        funcs = [s for s in sites if s["kind"] == "func"]
        assert all(s["parent"] is None and s["depth"] == 0 for s in funcs)
        by_id = {s["id"]: s for s in sites}
        for s in sites:
            if s["parent"] is not None:
                assert by_id[s["parent"]]["depth"] == s["depth"] - 1

    def test_block_statement_crossover(self):
        a, b = src("least_allocated"), src("most_allocated")
        a_if = next(s for s in ps.enumerate_subtrees(a) if s["kind"] == "if")
        sb = ps.enumerate_subtrees(b)
        b_stmt = next(s for s in sb if not s["is_expr"] and s["kind"] != "func")
        try:
            child = ps.crossover(a, b, a_if["id"], b_stmt["id"])
            ps.compile_check(child)  # valid offspring compiles
        except ps.PlacementScriptError:
            pass  # invalid offspring surfaces as a structured error, never broken source

    def test_function_crossover(self):
        a, b = src("least_allocated"), src("most_allocated")
        sa, sb = ps.enumerate_subtrees(a), ps.enumerate_subtrees(b)
        a_fn = next(s for s in sa if s["kind"] == "func" and s["func"] == "score")
        b_fn = next(s for s in sb if s["kind"] == "func" and s["func"] == "score")
        # Identity swap is always a valid offspring.
        ps.Policy(ps.crossover(a, a, a_fn["id"], a_fn["id"]))
        try:
            # Cross-policy swap may reference helpers absent in the target -> structured
            # error (invalid offspring), never silently broken source.
            ps.Policy(ps.crossover(a, b, a_fn["id"], b_fn["id"]))
        except ps.PlacementScriptError:
            pass
        # Function sites cross only with function sites.
        b_expr = next(s for s in sb if s["is_expr"])
        with pytest.raises(ps.PlacementScriptError):
            ps.crossover(a, b, a_fn["id"], b_expr["id"])

    def test_module_constant_literals_patchable(self):
        source = "GAIN: float = 0.75\ndef f(a: float) -> float:\n    return a * GAIN\n"
        lits = ps.enumerate_numeric_literals(source)
        const = next(x for x in lits if x["is_const"])
        patched = ps.patch_numeric_literals(source, {const["id"]: "-0.5"})
        assert "GAIN: float = -0.5" in patched
        ps.compile_check(patched)

    def test_underscore_literals_accepted(self):
        source = "def f() -> int:\n    return 1000\n"
        lits = ps.enumerate_numeric_literals(source)
        assert "1_000" in ps.patch_numeric_literals(source, {lits[0]["id"]: "1_000"})

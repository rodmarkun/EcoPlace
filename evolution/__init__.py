"""Evolution harness (Phase 5) and reusable components that predate it.

Includes the property-based risk-reduction gates from TODO 1.6 and the controlled
LLM-vs-GP placement-policy harness from Phase 5.
"""

from .gp_operator import GPOperatorConfig, PlacementScriptGPOperator

__all__ = ["GPOperatorConfig", "PlacementScriptGPOperator"]

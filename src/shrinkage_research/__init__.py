"""Independent numerical implementation; see docs/methodology.md."""

from .estimator import ShrinkageEstimate, ledoit_wolf_identity

__all__ = ["ShrinkageEstimate", "ledoit_wolf_identity"]
__version__ = "0.1.0"

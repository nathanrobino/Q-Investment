from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_module_path = Path(__file__).with_name("q-investment.py")
_spec = spec_from_file_location("Qmod.q_investment", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load module from {_module_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

Qmod = _module.Qmod

__all__ = ["Qmod"]


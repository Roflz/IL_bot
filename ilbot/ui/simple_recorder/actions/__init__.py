# bot/actions/__init__.py
from __future__ import annotations
import importlib, inspect, pkgutil as _pkg
from types import ModuleType
from typing import Callable

__all__: list[str] = []
__modules__: dict[str, ModuleType] = {}

def _should_export(name: str, obj) -> bool:
    # Export public functions only; tweak predicate as you like.
    return inspect.isfunction(obj) and not name.startswith("_")

# Import every submodule in this package once
for _m in _pkg.iter_modules(__path__):           # type: ignore[name-defined]
    if _m.name.startswith("_"):
        continue
    mod = importlib.import_module(f"{__name__}.{_m.name}")
    __modules__[_m.name] = mod
    for name, obj in vars(mod).items():
        if _should_export(name, obj):
            globals()[name] = obj
            __all__.append(name)

# Optional (PEP 562): dynamic lookup + nicer autocomplete
def __getattr__(name: str):
    for mod in __modules__.values():
        if hasattr(mod, name):
            obj = getattr(mod, name)
            if _should_export(name, obj):
                globals()[name] = obj
                if name not in __all__:
                    __all__.append(name)
                return obj
    raise AttributeError(name)

def __dir__():
    return sorted(set(list(globals().keys()) + __all__))

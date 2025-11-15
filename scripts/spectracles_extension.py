"""Stuff that I will eventually move to Spectracles properly, but that I want now."""

import keyword

import equinox as eqx


def is_valid_identifier(s: str) -> bool:
    """Check if a string is a valid Python identifier and not a keyword."""
    return s.isidentifier() and not keyword.iskeyword(s)


def dict_to_module(d: dict, module_name: str | None = None) -> eqx.Module:
    """
    Convert a dict to an equinox Module with attributes matching the keys.

    Args:
        d: Dictionary to convert. All keys must be valid Python identifiers
           (not keywords), and all values must be eqx.Module instances.

    Raises:
        TypeError: If any key is not a string or any value is not an eqx.Module
        ValueError: If any key is not a valid identifier or is a Python keyword
    """
    for key, value in d.items():
        # Check key is a string
        if not isinstance(key, str):
            raise TypeError(f"Dict key {key!r} is not a string (type: {type(key).__name__})")

        # Check key is a valid identifier
        if not is_valid_identifier(key):
            if keyword.iskeyword(key):
                raise ValueError(
                    f"Dict key {key!r} is a Python keyword and cannot be used as an attribute name"
                )
            else:
                raise ValueError(f"Dict key {key!r} is not a valid Python identifier")

        # Check value is an eqx.Module
        if not isinstance(value, eqx.Module):
            raise TypeError(
                f"Dict value for key {key!r} is not an eqx.Module (type: {type(value).__name__})"
            )

    # All checks passed, create the module with annotations
    annotations = {key: type(value) for key, value in d.items()}

    name = "DictModule" if module_name is None else module_name
    DictModule = type(name, (eqx.Module,), {"__annotations__": annotations})

    # Create instance and set attributes
    module = object.__new__(DictModule)
    for key, value in d.items():
        object.__setattr__(module, key, value)

    return module

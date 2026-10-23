"""Column specifications shared by the shear-wave and static operators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """Sizes for one linear 1D soil model.

    ``response_index`` selects state rows. Static stress is unscaled after the
    solve, so both of its state rows are reported.
    """

    name: str
    n_props: int
    state_dim: int
    n_bc: int
    response_index: tuple[int, ...]
    n_generators: int = 4
    use_frequency: bool = False
    hidden: int = 64
    n_latent: int = 8
    n_modes: int = 4


SH_SPEC = ColumnSpec(
    name="sh",
    n_props=1,
    state_dim=4,
    n_bc=1,
    response_index=(0, 1),
    use_frequency=True,
)

STATIC_SPEC = ColumnSpec(
    name="static",
    n_props=2,
    state_dim=2,
    n_bc=1,
    response_index=(0, 1),
    use_frequency=False,
)


def spec_by_name(name: str) -> ColumnSpec:
    if name == "sh":
        return SH_SPEC
    if name == "static":
        return STATIC_SPEC
    raise ValueError(f"unknown column model {name!r}")

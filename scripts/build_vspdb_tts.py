"""Convert VSPDB Vs profiles to raw two-way travel time (TTS).

Reads ``data/vspdb_vs_profiles.parquet`` (columns: velocity_metadata_id, depth,
vs_value) and writes ``data/vspdb_tts_profiles.parquet`` with an additional
``tts`` column in **seconds** (two-way: ``2 * Δz / Vs``), not log1p.

Breakpoint extraction for the flagship Transformer CFM model consumes this
raw-TTS parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soilgen_ai.tts_profiles.check import TTSProfileProcessor
from soilgen_ai.vs_profiles.vs_calculation import compute_tts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_tts_profiles(
    vs_path: Path,
    output_path: Path,
    validate: bool = True,
) -> pd.DataFrame:
    """Build a long-format TTS parquet from VSPDB Vs profiles.

    Args:
        vs_path: Path to ``vspdb_vs_profiles.parquet``.
        output_path: Destination parquet path.
        validate: If True, run ``TTSProfileProcessor`` as a sanity check.
            Invalid profiles are reported but still written (breakpoint
            extraction later drops degenerate sequences).

    Returns:
        The concatenated TTS DataFrame that was written.
    """
    vs_df = pd.read_parquet(vs_path)
    required = {"velocity_metadata_id", "depth", "vs_value"}
    missing = required - set(vs_df.columns)
    if missing:
        raise ValueError(f"{vs_path} missing columns: {sorted(missing)}")

    tts_profiles: dict = {}
    skipped = 0
    for metadata_id, group in vs_df.groupby("velocity_metadata_id"):
        profile = group[["depth", "vs_value"]].copy()
        try:
            tts_profile = compute_tts(profile, standardization=False)
        except ValueError as exc:
            print(f"Skipping profile {metadata_id}: {exc}")
            skipped += 1
            continue
        if tts_profile is None or tts_profile.empty:
            skipped += 1
            continue
        tts_profile = tts_profile.copy()
        tts_profile["velocity_metadata_id"] = metadata_id
        tts_profiles[metadata_id] = tts_profile

    if validate and tts_profiles:
        processor = TTSProfileProcessor(tts_profiles).process_profiles()
        n_invalid = len(processor.invalid_profiles)
        print(
            f"TTS validation: {len(tts_profiles) - n_invalid} valid, "
            f"{n_invalid} invalid"
        )
        if n_invalid:
            reasons: dict[str, int] = {}
            for reason in processor.invalid_profiles.values():
                reasons[reason] = reasons.get(reason, 0) + 1
            print(f"  Invalid reasons: {reasons}")

    if not tts_profiles:
        raise RuntimeError("No TTS profiles were produced.")

    out = pd.concat(tts_profiles.values(), ignore_index=True)
    out = out[["velocity_metadata_id", "depth", "vs_value", "tts"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    n_profiles = int(out["velocity_metadata_id"].nunique())
    tts = out["tts"]
    max_depth = float(out["depth"].max())
    print(f"Wrote {len(out)} rows / {n_profiles} profiles to {output_path}")
    print(f"  skipped profiles: {skipped}")
    print(f"  TTS range: [{float(tts.min()):.6f}, {float(tts.max()):.6f}] s")
    print(f"  TTS mean: {float(tts.mean()):.6f} s")
    print(f"  depth range: [0, {max_depth:.1f}] m")
    if float(tts.max()) > 50:
        print("  WARNING: TTS max > 50 s; values may not be in seconds.")
    if float(tts.max()) < 0.5 and max_depth > 100:
        print(
            "  WARNING: TTS max < 0.5 s at depths > 100 m; "
            "values may be log1p rather than raw seconds."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build raw two-way TTS parquet from VSPDB Vs profiles"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "vspdb_vs_profiles.parquet",
        help="Path to vspdb_vs_profiles.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "vspdb_tts_profiles.parquet",
        help="Destination parquet path",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip TTSProfileProcessor monotonicity/velocity sanity check",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Vs profiles not found: {args.input}. "
            "Run scripts/vspdb_request_data.py first."
        )
    build_tts_profiles(args.input, args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()

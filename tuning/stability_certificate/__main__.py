"""Command-line entry point for the automatic hierarchy certificate."""

from __future__ import annotations

import argparse
from pathlib import Path

from .hierarchy import (
    DEFAULT_CONFIG_FACTORY,
    DEFAULT_DELTAS,
    analyse_config,
    load_config_factory,
    write_json,
)
from .report import write_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-factory", default=DEFAULT_CONFIG_FACTORY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stability_certificate"),
    )
    parser.add_argument(
        "--delta",
        type=float,
        action="append",
        dest="deltas",
        help="Relative cached-gradient error bound; repeat for a sweep.",
    )
    args = parser.parse_args()
    deltas = tuple(args.deltas) if args.deltas is not None else DEFAULT_DELTAS
    config = load_config_factory(args.config_factory)
    certificate = analyse_config(
        config,
        config_factory=args.config_factory,
        deltas=deltas,
    )
    stem = args.config_factory.split(":", 1)[0].rsplit(".", 1)[-1]
    markdown_path = args.output_dir / f"{stem}_certificate.md"
    json_path = args.output_dir / f"{stem}_certificate.json"
    write_markdown(certificate, markdown_path)
    write_json(certificate, json_path)
    print(f"Certificate: {certificate.coupled_continuous.projected_full_state_iqc.status.value}")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

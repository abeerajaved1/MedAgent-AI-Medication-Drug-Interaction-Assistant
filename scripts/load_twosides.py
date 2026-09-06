#!/usr/bin/env python3
"""
Tier 3 #8: fetch a real TWOSIDES sample and load it into the interaction
graph, so real-world interaction coverage goes beyond the ~7 hand-seeded
rows in the demo database.

WHY THIS IS A SCRIPT YOU RUN, NOT SOMETHING THAT RUNS ITSELF
--------------------------------------------------------------
TWOSIDES (Tatonetti Lab) is free and public — no signup, no API key, no
cost — but it does not meet the bar of "something this delivery can just
do for you":
  - The official files are hosted on the Tatonetti Lab's own S3 bucket
    (https://nsides.io/, browseable at
    http://tatonettilab-resources.s3-website-us-west-1.amazonaws.com/?p=nsides/)
    and, depending on the current quarterly release, run from tens of MB
    up to several GB compressed.
  - That listing page is JavaScript-rendered, so its exact current
    filename can't be resolved by a static fetch — you'll need to open
    it in a real browser once to get the current direct link.
  - I could not download or execute any of this myself while building
    this delivery: my tool environment's network egress is restricted to
    a fixed allowlist (package registries, GitHub, etc.) that does not
    include nsides.io / tatonettilab.org / AWS S3, and the files are too
    large to reasonably fetch and commit into a delivery zip even if it
    did. This is an honest infrastructure limitation, not a cost one —
    the data itself is 100% free.

This script is a real, ready-to-run tool for the one step that has to
happen on your machine: given a TWOSIDES CSV (or .csv.gz — this script
handles either) you've downloaded, it loads it into MedAgent's existing
interaction graph loader (app/graph/interaction_graph.py's
`load_twosides_csv()`, which was already written but never actually
exercised against real data) and prints the before/after node and edge
counts, so you can directly confirm the graph's real-world coverage grew
beyond the seeded 9 nodes / 7 edges baseline.

USAGE
-----
    # 1. Get a TWOSIDES CSV yourself (one-time, needs a real internet
    #    connection this sandbox didn't have): visit
    #    http://tatonettilab-resources.s3-website-us-west-1.amazonaws.com/?p=nsides/
    #    in a browser, find the current TWOSIDES flat-file CSV (or
    #    .csv.gz), and download it.
    #
    # 2. Run this script against it:
    python scripts/load_twosides.py --file /path/to/TWOSIDES.csv.gz

    # Optional: cap how many rows to import (useful for a quick demo-sized
    # subset instead of the full multi-million-row file):
    python scripts/load_twosides.py --file /path/to/TWOSIDES.csv.gz --max-rows 5000

    # Optional: if your CSV's column names differ from the common export
    # format, override them (see load_twosides_csv()'s defaults):
    python scripts/load_twosides.py --file data.csv \\
        --drug1-col drug_1_concept_name --drug2-col drug_2_concept_name \\
        --event-col condition_meddra_name --prr-col PRR

Run this from the project root (same directory as requirements.txt) with
your normal virtualenv/dependencies active — it imports the app package
directly, the same way the FastAPI app does.
"""
import argparse
import gzip
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("load_twosides")


def _maybe_decompress(path: Path) -> Path:
    """Transparently handles a .csv.gz input by decompressing to a temp
    file first — the existing load_twosides_csv() loader expects a plain
    CSV path."""
    if path.suffix != ".gz":
        return path
    logger.info(f"Decompressing {path} ...")
    tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
    with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    logger.info(f"Decompressed to {tmp}")
    return tmp


def main():
    parser = argparse.ArgumentParser(description="Load a downloaded TWOSIDES CSV into MedAgent's interaction graph.")
    parser.add_argument("--file", required=True, help="Path to the TWOSIDES CSV or .csv.gz file you downloaded.")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Optional cap on rows imported (e.g. 5000 for a fast demo-sized subset).")
    parser.add_argument("--drug1-col", default="drug_1_concept_name")
    parser.add_argument("--drug2-col", default="drug_2_concept_name")
    parser.add_argument("--event-col", default="condition_meddra_name")
    parser.add_argument("--prr-col", default="PRR")
    args = parser.parse_args()

    src_path = Path(args.file)
    if not src_path.exists():
        logger.error(f"File not found: {src_path}")
        sys.exit(1)

    # Import the app package the same way FastAPI does — run this script
    # from the project root with your normal dependencies installed.
    try:
        from app.database import init_db
        from app.graph.interaction_graph import interaction_graph
    except ImportError as e:
        logger.error(
            f"Couldn't import the app package ({e}). Run this script from the project "
            f"root directory (the one containing requirements.txt and the app/ folder), "
            f"with your project's dependencies installed/activated."
        )
        sys.exit(1)

    init_db()
    interaction_graph._ensure_loaded()
    before_nodes = interaction_graph.graph.number_of_nodes()
    before_edges = interaction_graph.graph.number_of_edges()
    logger.info(f"Baseline (local seed data only): {before_nodes} nodes, {before_edges} edges.")

    csv_path = _maybe_decompress(src_path)
    added = interaction_graph.load_twosides_csv(
        str(csv_path),
        drug1_col=args.drug1_col, drug2_col=args.drug2_col,
        event_col=args.event_col, prr_col=args.prr_col,
        max_rows=args.max_rows,
    )

    after_nodes = interaction_graph.graph.number_of_nodes()
    after_edges = interaction_graph.graph.number_of_edges()

    print()
    print("=" * 60)
    print(f"TWOSIDES rows processed: {added}")
    print(f"Graph before: {before_nodes} nodes, {before_edges} edges")
    print(f"Graph after:  {after_nodes} nodes, {after_edges} edges")
    print("=" * 60)

    if after_nodes <= before_nodes and after_edges <= before_edges:
        logger.warning(
            "Graph size did not grow — double check the --drug1-col/--drug2-col/--event-col/"
            "--prr-col arguments match your CSV's actual column headers (TWOSIDES export "
            "formats have varied across releases)."
        )
    else:
        logger.info(
            "Success — the interaction graph now includes real TWOSIDES coverage beyond "
            "the local seed data for THIS SCRIPT RUN. To make the running app itself pick "
            "this up on every start (not just when you run this script), set the "
            "TWOSIDES_CSV_PATH environment variable to this file's path in your .env — "
            "app/main.py's startup hook will load it automatically every time the app starts."
        )


if __name__ == "__main__":
    main()

"""CLI entry point for step selection: train, evaluate, or grid search."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step selection: train, evaluate, or grid-search.",
        usage="python -m src.cli.select {train,evaluate,grid_search} ...",
    )
    parser.add_argument("command", choices=["train", "evaluate", "grid_search"])
    args, remaining = parser.parse_known_args(argv)

    if args.command == "train":
        from src.selection.train import main as train_main
        train_main(remaining)
    elif args.command == "evaluate":
        from src.selection.evaluate import main as eval_main
        eval_main(remaining)
    elif args.command == "grid_search":
        from src.selection.grid_search import main as grid_main
        grid_main(remaining)


if __name__ == "__main__":
    main()

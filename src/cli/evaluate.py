"""CLI entry point for Stage 7: Evaluate UQ features."""

from __future__ import annotations

import argparse

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UQ features with AUROC/AUPRC/PRR.")
    parser.add_argument("--features_path", required=True, help="Path to uq_features.jsonl")
    parser.add_argument("--output_metrics_json", help="Output JSON path")
    parser.add_argument("--output_metrics_csv", help="Output CSV path")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    from src.evaluate.runner import evaluate_run

    results = evaluate_run(
        args.features_path,
        output_json=args.output_metrics_json,
        output_csv=args.output_metrics_csv,
        bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    n = results["n_examples"]
    correct = results["n_correct"]
    wrong = results["n_wrong"]
    print(f"[evaluate] {n} examples: {correct} correct, {wrong} wrong")
    for name, metrics in results.get("features", {}).items():
        auroc = metrics.get("auroc")
        if auroc is not None:
            parts = [f"AUROC={auroc:.4f}"]
            for key in ("auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"):
                val = metrics.get(key)
                if val is not None:
                    parts.append(f"{key}={val:.4f}")
            print(f"  {name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()

"""Command line interface for the safety evaluation harness."""

from __future__ import annotations

import argparse
import sys

from safety_eval import database
from safety_eval.report import generate_html_report
from safety_eval.runner import run_evaluation, summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safety_eval.cli",
        description="Run lightweight AI safety evaluations over JSON or CSV prompt datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a dataset against a model.")
    run_parser.add_argument("--dataset", required=True, help="Path to JSON or CSV prompts.")
    run_parser.add_argument(
        "--model",
        default="mock",
        choices=["mock", "claude"],
        help="Model backend to use. Mock works locally; Claude requires env vars.",
    )
    run_parser.add_argument(
        "--db",
        default=database.DEFAULT_DB_PATH,
        help=f"SQLite database path. Default: {database.DEFAULT_DB_PATH}",
    )
    run_parser.add_argument("--notes", default="", help="Optional note stored with each row.")
    run_parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path for exporting this run to CSV immediately.",
    )
    run_parser.add_argument(
        "--html-report",
        default="",
        help="Optional path for generating an HTML report for this run immediately.",
    )

    export_parser = subparsers.add_parser("export", help="Export logged runs.")
    export_parser.add_argument(
        "--format",
        default="csv",
        choices=["csv"],
        help="Export format. Currently only csv is supported.",
    )
    export_parser.add_argument(
        "--db",
        default=database.DEFAULT_DB_PATH,
        help=f"SQLite database path. Default: {database.DEFAULT_DB_PATH}",
    )
    export_parser.add_argument(
        "--output",
        default="exports/eval_runs.csv",
        help="Output path. Default: exports/eval_runs.csv",
    )
    export_parser.add_argument("--run-id", default="", help="Optional run id filter.")

    report_parser = subparsers.add_parser("report", help="Generate a static HTML report.")
    report_parser.add_argument(
        "--db",
        default=database.DEFAULT_DB_PATH,
        help=f"SQLite database path. Default: {database.DEFAULT_DB_PATH}",
    )
    report_parser.add_argument(
        "--output",
        default="reports/eval_report.html",
        help="Output path. Default: reports/eval_report.html",
    )
    report_parser.add_argument("--run-id", default="", help="Optional run id filter.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            results = run_evaluation(
                dataset_path=args.dataset,
                model_name=args.model,
                db_path=args.db,
                notes=args.notes,
            )
            summary = summarize_results(results)
            print_run_summary(summary, results[0].run_id if results else "", args.db)

            rows = database.fetch_runs(args.db, run_id=results[0].run_id if results else None)
            if args.export_csv:
                path = database.export_runs_csv(rows, args.export_csv)
                print(f"CSV exported: {path}")
            if args.html_report:
                path = generate_html_report(rows, args.html_report)
                print(f"HTML report written: {path}")
            return 0

        if args.command == "export":
            rows = database.fetch_runs(args.db, run_id=args.run_id or None)
            path = database.export_runs_csv(rows, args.output)
            print(f"Exported {len(rows)} rows to {path}")
            return 0

        if args.command == "report":
            rows = database.fetch_runs(args.db, run_id=args.run_id or None)
            path = generate_html_report(rows, args.output)
            print(f"HTML report written: {path}")
            return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def print_run_summary(summary: dict[str, object], run_id: str, db_path: str) -> None:
    print("Run complete")
    print(f"Run id: {run_id}")
    print(f"SQLite log: {db_path}")
    print(f"Total prompts: {summary['total']}")
    print(f"Refusal rate: {summary['refusal_rate']:.1%}")
    print(f"Harmful compliance count: {summary['harmful_compliance_count']}")
    print(f"Jailbreak indicator count: {summary['jailbreak_count']}")
    print(f"Hallucination-risk language count: {summary['hallucination_risk_count']}")
    print(f"Uncertainty language count: {summary['uncertainty_count']}")
    print(f"Average risk score: {summary['average_risk_score']:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())


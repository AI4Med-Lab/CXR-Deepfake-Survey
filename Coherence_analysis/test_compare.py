"""
Compute matching accuracy from per-round CSVs (match_results_round{i}.csv)
over a specified range of round numbers, instead of relying on the
combined match_results_all_rounds.csv.

Usage:
    python compute_accuracy.py --dir /path/to/output_dir --rounds 0-49
    python compute_accuracy.py --dir /path/to/output_dir --rounds 0,1,2,5,10
    python compute_accuracy.py --dir /path/to/output_dir --rounds 0-9,20-29
    python compute_accuracy.py --dir /path/to/output_dir --rounds all
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_rounds(rounds_str: str) -> list[int]:
    """Parse a rounds spec like '0-49', '0,1,2,5', or '0-9,20-29' into a
    sorted list of unique round indices.

    Callers should check for the special value "all" (case-insensitive)
    before calling this function -- "all" means "read
    match_results_all_rounds.csv directly" rather than a set of round
    indices, so it's handled separately in compute_accuracy.
    """
    rounds = set()
    for part in rounds_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{part}': end < start")
            rounds.update(range(start, end + 1))
        else:
            rounds.add(int(part))
    return sorted(rounds)


def load_combined(output_dir: Path) -> pd.DataFrame:
    """Load match_results_all_rounds.csv directly, for --rounds all."""
    csv_path = output_dir / "match_results_all_rounds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    return pd.read_csv(csv_path)


def load_from_round_files(output_dir: Path, round_indices: list[int]) -> pd.DataFrame | None:
    """Load and concatenate match_results_round{i}.csv for each requested
    round index, warning about (and skipping) any that don't exist.
    """
    frames = []
    missing = []

    for round_idx in round_indices:
        csv_path = output_dir / f"match_results_round{round_idx}.csv"
        if not csv_path.exists():
            missing.append(round_idx)
            continue
        df = pd.read_csv(csv_path)
        frames.append(df)

    if missing:
        print(f"Warning: missing CSVs for round(s): {missing}")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def report_accuracy(combined: pd.DataFrame, label: str) -> None:
    # Two known schemas produce match_results_all_rounds.csv / per-round
    # CSVs:
    #   1. MedGemma-style (match_reports_v3.py): one row PER IMAGE, with a
    #      per-row boolean "correct" column.
    #   2. MedSigLIP-style (match_reports_siglip_nxn.py): one row PER
    #      ROUND, with aggregate columns "hungarian_correct_count" and
    #      "n" (samples in that round) instead of a per-row "correct".
    #      There's no per-image boolean column at all here -- accuracy
    #      has to be computed as sum(hungarian_correct_count) / sum(n).
    # Detect which schema we're looking at and report accordingly, rather
    # than assuming "correct" always exists.
    if "correct" in combined.columns:
        _report_per_row_accuracy(combined, label)
    elif "hungarian_correct_count" in combined.columns and "n" in combined.columns:
        _report_per_round_aggregate_accuracy(combined, label)
    else:
        raise ValueError(
            "Neither a 'correct' column nor 'hungarian_correct_count'/'n' columns "
            "found in the round CSVs -- unrecognized schema."
        )


def _report_per_row_accuracy(combined: pd.DataFrame, label: str) -> None:
    n_total = len(combined)
    n_correct = (combined["correct"] == True).sum()  # noqa: E712
    accuracy = n_correct / n_total if n_total else 0.0

    print(f"Rounds included: {label}")
    print(f"Total samples: {n_total}")
    print(f"Correct: {n_correct}")
    print(f"Accuracy: {accuracy:.2%}")

    # Per-round breakdown, useful for spotting rounds that dragged the
    # average down (e.g. because of truncated/failed generations).
    if "round" in combined.columns:
        print("\nPer-round breakdown:")
        per_round = combined.groupby("round")["correct"].agg(["sum", "count"])
        per_round["accuracy"] = per_round["sum"] / per_round["count"]
        for r, row in per_round.iterrows():
            print(f"  Round {r}: {row['accuracy']:.2%} ({int(row['sum'])}/{int(row['count'])})")


def _report_per_round_aggregate_accuracy(combined: pd.DataFrame, label: str) -> None:
    n_total = int(combined["n"].sum())
    n_correct = int(combined["hungarian_correct_count"].sum())
    accuracy = n_correct / n_total if n_total else 0.0

    print(f"Rounds included: {label}")
    print(f"Total samples: {n_total}")
    print(f"Correct (Hungarian): {n_correct}")
    print(f"Accuracy (Hungarian): {accuracy:.2%}")

    # Greedy (independent-argmax) numbers are saved alongside Hungarian
    # in this schema -- surface them too since they're right there and
    # useful for the same diagnostic purpose (comparing optimal vs. naive
    # assignment quality).
    if "greedy_correct_count" in combined.columns:
        n_greedy_correct = int(combined["greedy_correct_count"].sum())
        greedy_accuracy = n_greedy_correct / n_total if n_total else 0.0
        print(f"Correct (greedy): {n_greedy_correct}")
        print(f"Accuracy (greedy): {greedy_accuracy:.2%}")

    if "greedy_had_conflict" in combined.columns:
        conflict_rate = combined["greedy_had_conflict"].mean()
        print(f"Rounds with greedy conflict: {conflict_rate:.2%}")

    if "hungarian_vs_greedy_differ" in combined.columns:
        disagreement_rate = combined["hungarian_vs_greedy_differ"].mean()
        print(f"Rounds where Hungarian and greedy differ: {disagreement_rate:.2%}")

    # Per-round breakdown: here each row already IS a round, so no
    # groupby is needed -- just print each row's own accuracy.
    if "round" in combined.columns:
        print("\nPer-round breakdown:")
        for _, row in combined.iterrows():
            r = row["round"]
            correct = int(row["hungarian_correct_count"])
            n = int(row["n"])
            round_acc = correct / n if n else 0.0
            print(f"  Round {r}: {round_acc:.2%} ({correct}/{n})")


def compute_accuracy(output_dir: Path, rounds_str: str) -> None:
    # "all" bypasses round-index parsing entirely and reads the combined
    # file the main script already writes out, rather than trying to
    # infer which round indices exist on disk.
    if rounds_str.strip().lower() == "all":
        combined = load_combined(output_dir)
        report_accuracy(combined, label="all (match_results_all_rounds.csv)")
        return

    round_indices = parse_rounds(rounds_str)
    combined = load_from_round_files(output_dir, round_indices)
    if combined is None:
        print("No round CSVs found -- nothing to compute.")
        return
    report_accuracy(combined, label=str(round_indices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute accuracy across a range of rounds")
    parser.add_argument(
        "--dir",
        type=Path,
        required=False,
        default=Path(""),
        help="Directory containing match_results_round{i}.csv files "
        "(and/or match_results_all_rounds.csv)",
    )
    parser.add_argument(
        "--rounds",
        type=str,
        required=False,
        default="all",
        help="Round numbers/ranges, e.g. '0-49', '0,1,2,5', '0-9,20-29', "
        "or 'all' to read match_results_all_rounds.csv directly",
    )
    args = parser.parse_args()

    compute_accuracy(args.dir, args.rounds)
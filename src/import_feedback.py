"""Turn user feedback (collected on the map) into labeled training rows.

Joins Feedback (ground truth: did it actually flood, was the score
accurate/over/under) with the RiskScore.features snapshot from the same
(location, scoredFor) generation -- the exact input the model was scored
on -- and writes data/feedback.csv. src/train.py concatenates this onto
train.csv at training time, so this is the data flywheel: map feedback
-> labeled rows -> next retrain.

This is a manual/CI-triggered job (run before `python -m src.train`), not
a best-effort serving-time concern like src/logstore.py -- it fails
loudly if DATABASE_URL isn't set or the query fails, rather than no-op.

Relabeling is a first-pass heuristic (see derive_target below), not a
calibrated remapping. A natural follow-up: fit an isotonic regression
from (predicted, feedback) pairs instead of fixed multipliers, once
there's enough feedback volume to do that meaningfully.
"""
import json
import os
from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path("data/feedback.csv")


def derive_target(predicted_score: float, actual_flooded: bool | None, accuracy: str | None) -> float | None:
    """Relabel a prediction using user feedback. Returns None if neither
    `accuracy` nor `actual_flooded` carries any signal (caller skips the row)."""
    if accuracy == "accurate":
        return round(predicted_score, 4)
    if accuracy == "overestimated":
        return round(predicted_score * 0.5, 4)
    if accuracy == "underestimated":
        return round(min(1.0, predicted_score * 1.5 + 0.1), 4)
    if actual_flooded is not None:
        # No accuracy given, but the outcome is known -- anchor toward it.
        return 0.85 if actual_flooded else 0.15
    return None


def fetch_feedback_rows() -> list[dict]:
    database_url = os.environ["DATABASE_URL"]  # required: fail loudly if missing
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    f.id AS feedback_id,
                    f."actualFlooded" AS actual_flooded,
                    f.accuracy,
                    f.comment,
                    rs.features,
                    rs.score AS predicted_score,
                    l.name AS place_name
                FROM "Feedback" f
                JOIN "RiskScore" rs
                    ON rs."locationId" = f."locationId" AND rs."scoredFor" = f."scoredFor"
                JOIN "Location" l ON l.id = f."locationId"
                WHERE f."actualFlooded" IS NOT NULL OR f.accuracy IS NOT NULL
                """
            )
            return cur.fetchall()
    finally:
        conn.close()


def build_feedback_dataset() -> pd.DataFrame:
    rows = fetch_feedback_rows()
    out_rows = []
    skipped_no_features = 0
    skipped_no_signal = 0
    for r in rows:
        if not r["features"]:
            skipped_no_features += 1
            continue
        target = derive_target(r["predicted_score"], r["actual_flooded"], r["accuracy"])
        if target is None:
            skipped_no_signal += 1
            continue

        features = r["features"] if isinstance(r["features"], dict) else json.loads(r["features"])
        row = dict(features)
        row["flood_risk_score"] = target
        row["record_id"] = f"feedback-{r['feedback_id']}"
        row["place_name"] = r["place_name"]
        row["reason_not_good_to_live"] = r["comment"] or ""
        row["is_synthetic"] = True  # marks this row as feedback-derived, not original data
        out_rows.append(row)

    if skipped_no_features:
        print(f"Skipped {skipped_no_features} feedback row(s) with no matching RiskScore.features.")
    if skipped_no_signal:
        print(f"Skipped {skipped_no_signal} feedback row(s) with neither accuracy nor actualFlooded set.")

    return pd.DataFrame(out_rows)


def main():
    df = build_feedback_dataset()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} feedback-derived row(s) to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()

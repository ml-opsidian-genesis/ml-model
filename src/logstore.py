import os
import uuid

DATABASE_URL = os.getenv("DATABASE_URL")


def log_prediction(
    source: str,
    model_version: str,
    payload: dict,
    score: float,
    risk_level: str,
    confidence: str,
    location_id: str | None = None,
) -> None:
    """Insert one row into PredictionLog. No-op if DATABASE_URL isn't
    configured. Never raises."""
    if not DATABASE_URL:
        return
    try:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "PredictionLog"
                        ("id", "source", "modelVersion", "locationId", "input", "score", "riskLevel", "confidence")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()), source, model_version, location_id,
                        psycopg2.extras.Json(payload), score, risk_level, confidence,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[logstore] failed to log prediction: {e}")


def log_predictions_batch(entries: list[dict]) -> None:
    """Insert multiple PredictionLog rows in one connection/transaction
    (used by /predict/batch so one batch call doesn't open N connections).
    Each entry: {payload, score, risk_level, confidence, location_id}.
    No-op if DATABASE_URL isn't configured. Never raises."""
    if not DATABASE_URL or not entries:
        return
    try:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn.cursor() as cur:
                rows = [
                    (
                        str(uuid.uuid4()), "predict/batch", e["model_version"], e.get("location_id"),
                        psycopg2.extras.Json(e["payload"]), e["score"], e["risk_level"], e["confidence"],
                    )
                    for e in entries
                ]
                cur.executemany(
                    """
                    INSERT INTO "PredictionLog"
                        ("id", "source", "modelVersion", "locationId", "input", "score", "riskLevel", "confidence")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[logstore] failed to log prediction batch: {e}")


def fetch_metrics(limit_recent: int = 10) -> dict | None:
    """Query aggregate stats from PredictionLog. Returns None if
    DATABASE_URL isn't configured or the query fails, so callers can fall
    back to a local/file-based view."""
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute('SELECT COUNT(*) AS total, AVG("score") AS avg_score FROM "PredictionLog"')
                totals = cur.fetchone()

                cur.execute('SELECT "riskLevel", COUNT(*) AS n FROM "PredictionLog" GROUP BY "riskLevel"')
                dist = {row["riskLevel"]: row["n"] for row in cur.fetchall()}

                cur.execute(
                    'SELECT "input", "score", "riskLevel", "createdAt" FROM "PredictionLog" '
                    'ORDER BY "createdAt" DESC LIMIT %s',
                    (limit_recent,),
                )
                recent = [
                    {
                        "input": row["input"],
                        "score": row["score"],
                        "risk_level": row["riskLevel"],
                        "timestamp": row["createdAt"].isoformat(),
                    }
                    for row in cur.fetchall()
                ]
            return {
                "total_predictions": totals["total"] or 0,
                "avg_risk_score": round(float(totals["avg_score"]), 4) if totals["avg_score"] is not None else 0,
                "risk_distribution": dist,
                "recent": recent,
            }
        finally:
            conn.close()
    except Exception as e:
        print(f"[logstore] failed to fetch metrics: {e}")
        return None

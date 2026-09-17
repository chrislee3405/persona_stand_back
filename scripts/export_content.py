import json, os, pathlib
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])

TABLES = {
    "site_content":   "section, created_at",
    "site_media":     "section, description, created_at",
    "site_project":   "project_id, created_at",
    "site_journey":   "journey_id, created_at",
    "consent_policy": "id",
    "doc_reference":         "document_topic",
    "scenario_reference":    "scenario_topic",
    "question_bank":         "id",
    "personality_reference": "id",
}

out = pathlib.Path("/app/audit/db")
out.mkdir(parents=True, exist_ok=True)

with engine.connect() as c:
    for t, order in TABLES.items():
        try:
            rows = [dict(r) for r in c.execute(text(f"SELECT * FROM {t} ORDER BY {order}")).mappings()]
            (out / f"{t}.json").write_text(
                json.dumps(rows, indent=2, default=str, ensure_ascii=False)
            )
            print(f"{t}: {len(rows)} rows")
        except Exception as e:
            print(f"{t}: SKIP ({e})")
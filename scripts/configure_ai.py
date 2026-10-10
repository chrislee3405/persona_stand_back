"""Validate a complete provider configuration with live probes, then activate atomically.

Run from the backend root: python -m scripts.configure_ai settings.json [--check-only]
The probe calls incur provider usage. No prompts contain conversation or owner data.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from sqlalchemy import select, text, update, func

from app.database import SessionLocal, engine
from app.models.model_configuration import ModelConfiguration
from app.services.ai.configuration import AIConfiguration
from app.services.ai.gemini_service import GeminiService
from app.services.ai.provider_factory import get_ai_service


async def activate(configuration, check_only=False):
    ai = get_ai_service(GeminiService())
    ai.configure(configuration)
    provider = ai.providers[configuration.provider]
    # Validate every distinct configured model before opening the write transaction.
    # This verifies access and basic text/structured capabilities, not task quality.
    models = {value for key, value in asdict(configuration).items() if key != "provider" and value}
    for model in sorted(models):
        await provider.call_model(model, "Reply with OK.")
        result = await provider.call_model_structured(model, "Return OK.", "Follow the response schema.",
                                                      {"type": "STRING", "enum": ["OK"]})
        if result != "OK":
            raise ValueError("Configured model failed the structured-output probe")
    if not check_only:
        async with SessionLocal.begin() as db:
            await db.execute(text("LOCK TABLE model_configuration IN SHARE ROW EXCLUSIVE MODE"))
            row = await db.scalar(select(ModelConfiguration).where(ModelConfiguration.provider == configuration.provider))
            enabled_priorities = set((await db.scalars(select(ModelConfiguration.use_priority).where(
                ModelConfiguration.is_active.is_(True), ModelConfiguration.provider != configuration.provider))).all())
            priority = row.use_priority if row is not None else 1
            if priority in enabled_priorities:
                priority = max(enabled_priorities) + 1
            if row is None:
                next_id = (await db.scalar(select(func.max(ModelConfiguration.id))) or 0) + 1
                db.add(ModelConfiguration(id=next_id, is_active=True, use_priority=priority, **asdict(configuration)))
            else:
                await db.execute(update(ModelConfiguration).where(ModelConfiguration.id == row.id)
                                 .values(**asdict(configuration), is_active=True, use_priority=priority, updated_at=func.now()))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.file.read_text(encoding="utf-8"))
    configuration = AIConfiguration(**payload)
    async def run():
        try:
            await activate(configuration, args.check_only)
        finally:
            await engine.dispose()
    asyncio.run(run())
    print("Configuration validated" if args.check_only else "Configuration validated and activated")


if __name__ == "__main__":
    main()

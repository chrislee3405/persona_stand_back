"""Opt-in live checks of the deployed prompts using only fictional fixtures.

Run inside the backend environment:
  python scripts/check_conversation_prompts.py --run-live --repeat 3
Uses paid model calls with the current stage configuration. Reads only model
configuration from the database; never reads or writes conversation/profile data.
Failures remain failures: this script does not retry until a model passes.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal, engine
from app.services.ai.configuration import load_configuration
from app.services.ai.gemini_service import GeminiService
from app.services.ai.provider_factory import get_ai_service
from app.services.model_collaborate.evidence_contract import validate_grounding
from app.services.model_collaborate.grounding_service import _GROUND_SYSTEM_PROMPT, _GROUND_RESPONSE_SCHEMA
from app.services.model_collaborate.response_parser import (
    _SPLIT_RESPONSE_SYSTEM_PROMPT_TEMPLATE, _SPLIT_RESPONSE_USER_PROMPT_TEMPLATE,
)

FIXTURES = Path(__file__).with_name('fixtures') / 'conversation_prompt_regressions.json'


def check(fixture, value):
    expected = fixture['expect']
    if fixture['stage'] == 'split':
        if not isinstance(value, list) or not value or any(not isinstance(s, str) or not s.strip() for s in value):
            return ['invalid_shape']
        failures = []
        if ''.join(value) != expected['text']:
            failures.append('changed_text_or_whitespace')
        if not expected['min_natural'] <= len(value) <= expected['max_natural']:
            failures.append('natural_grouping')
        if len(value) > expected['max_turns']:
            failures.append('turn_ceiling')
        # Counts alone cannot distinguish a complete thought from a dangling setup.
        if 'allowed_boundaries' in expected and ''.join(value) == expected['text']:
            offset = 0
            for turn in value[:-1]:
                offset += len(turn)
                if offset not in expected['allowed_boundaries']:
                    failures.append('incomplete_thought_boundary')
                    break
        return failures
    try:
        value = validate_grounding(value)
    except (ValueError, TypeError):
        return ['invalid_contract']
    failures = []
    if {part['action'] for part in value['response_parts']} != set(expected['actions']):
        failures.append('actions')
    if value['question_type'] not in expected['kind']:
        failures.append('question_type')
    if value['coverage'] not in expected['coverage']:
        failures.append('coverage')
    if any(not any(claim['source_quote'] in source['content'] for source in fixture['sources'])
           for claim in value['claims']):
        failures.append('nonverbatim_source')
    if expected['actions'] == ['answer_behavioural'] and value['missing'].strip():
        failures.append('behavioural_gap')
    return failures


async def run(args):
    fixtures = json.loads(FIXTURES.read_text(encoding='utf-8'))['fixtures']
    ai = get_ai_service(GeminiService())
    try:
        async with SessionLocal() as db:
            configuration = await load_configuration(db)
        ai.configure(configuration)
        rows = []
        semaphore = asyncio.Semaphore(3)

        async def one(fixture, repeat):
            async with semaphore:
                stage = fixture['stage']
                if stage == 'split':
                    system = _SPLIT_RESPONSE_SYSTEM_PROMPT_TEMPLATE.format(max_turns=fixture['expect']['max_turns'])
                    user = _SPLIT_RESPONSE_USER_PROMPT_TEMPLATE.format(response_text=fixture['expect']['text'])
                    schema = {'type': 'ARRAY', 'items': {'type': 'STRING'}}
                else:
                    system, user, schema = _GROUND_SYSTEM_PROMPT, fixture['user'], _GROUND_RESPONSE_SCHEMA
                value, error = None, None
                try:
                    value = await ai.call_model_structured(stage=stage, user_prompt=user,
                                                          system_prompt=system, schema=schema)
                    failures = check(fixture, value)
                except Exception as exc:
                    error, failures = type(exc).__name__, ['call_failed']
                row = {'stage': stage, 'case': fixture['case'], 'repeat': repeat,
                       'model': configuration.selection_for(stage), 'failures': failures,
                       'error': error, 'response': value}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'response'}), flush=True)

        await asyncio.gather(*(one(f, repeat) for repeat in range(args.repeat) for f in fixtures))
        if args.output:
            Path(args.output).write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
        failures = sum(bool(row['failures']) for row in rows)
        print(json.dumps({'trials': len(rows), 'failed': failures}), flush=True)
        return 1 if failures else 0
    finally:
        await engine.dispose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', help='Required: permits paid calls with fictional fixtures.')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', help='Optional local JSON result file.')
    args = parser.parse_args()
    if not args.run_live or args.repeat < 1:
        parser.error('--run-live and a positive --repeat are required')
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(asyncio.run(run(args)))

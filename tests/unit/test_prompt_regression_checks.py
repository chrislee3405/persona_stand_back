"""Judge completed meaning at boundaries, not just the number of bubbles."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prompt_checks', ROOT / 'scripts/check_conversation_prompts.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
FIXTURES = json.loads(checks.FIXTURES.read_text(encoding='utf-8'))['fixtures']


def fixture(case):
    return next(f for f in FIXTURES if f['case'] == case)


def test_unchanged_single_turn_is_not_enough_for_independent_parts():
    f = fixture('rights_availability')
    assert checks.check(f, [f['expect']['text']]) == ['natural_grouping']


def test_related_complete_thoughts_can_pause_or_stay_together():
    f = fixture('holdout_effect')
    first, second = f['expect']['text'].split('\n\n')
    assert checks.check(f, [first + '\n\n', second]) == []
    assert checks.check(f, [f['expect']['text']]) == []


def test_paragraph_break_does_not_complete_a_vague_setup():
    f = fixture('unfinished_setup')
    first, second = f['expect']['text'].split('\n\n')
    assert 'incomplete_thought_boundary' in checks.check(f, [first + '\n\n', second])
    assert checks.check(f, [f['expect']['text']]) == []


def test_correct_count_can_still_stop_at_an_unfinished_thought():
    f = fixture('setup_then_next_beat')
    text = f['expect']['text']
    wrong_stop = text.index('. ') + 2
    assert checks.check(f, [text[:wrong_stop], text[wrong_stop:]]) == ['incomplete_thought_boundary']
    right_stop = f['expect']['allowed_boundaries'][0]
    assert checks.check(f, [text[:right_stop], text[right_stop:]]) == []


def test_correct_boundaries_must_also_preserve_exact_text():
    f = fixture('rights_availability')
    first, second = f['expect']['text'].split('. ', 1)
    assert checks.check(f, [first + '. ', second]) == []
    assert checks.check(f, [first + '.', second]) == ['changed_text_or_whitespace']


def test_empty_bubbles_are_not_valid_splits():
    f = fixture('rights_availability')
    assert checks.check(f, [f['expect']['text'], '']) == ['invalid_shape']

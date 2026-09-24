from unittest.mock import AsyncMock

import pytest

from app.services.model_collaborate.response_parser import ResponseParser

RESPONSE = "The first sentence describes the project. The second sentence explains its tests."


async def test_short_reply_needs_no_model_call():
    gemini = AsyncMock()
    assert await ResponseParser(gemini).parse("A short reply.") == ["A short reply."]
    gemini.call_model_structured.assert_not_called()


@pytest.mark.parametrize("result", [None, {}, [], [""], [42]])
async def test_malformed_split_keeps_the_approved_reply(result):
    gemini = AsyncMock()
    gemini.call_model_structured.return_value = result
    assert await ResponseParser(gemini).parse(RESPONSE) == [RESPONSE]


async def test_failed_split_keeps_the_approved_reply():
    gemini = AsyncMock()
    gemini.call_model_structured.side_effect = TimeoutError("simulated")
    assert await ResponseParser(gemini).parse(RESPONSE) == [RESPONSE]


async def test_excess_bubbles_merge_without_losing_tail():
    gemini = AsyncMock()
    gemini.call_model_structured.return_value = ["The first sentence", "describes the project.", "The second sentence", "explains its tests."]
    result = await ResponseParser(gemini, min_chars_per_turn=40).parse(RESPONSE)
    assert result == ["The first sentence", "describes the project. The second sentence explains its tests."]


@pytest.mark.parametrize('split', [
    ['An invented qualification.'],
    ['The first sentence describes the project.'],
    [RESPONSE, RESPONSE],
    ['The second sentence explains its tests.', 'The first sentence describes the project.'],
])
async def test_split_cannot_rewrite_approved_content(split):
    gemini = AsyncMock()
    gemini.call_model_structured.return_value = split
    assert await ResponseParser(gemini).parse(RESPONSE) == [RESPONSE]


async def test_exact_partition_may_normalize_whitespace():
    gemini = AsyncMock()
    split = ['The first sentence describes the project.', 'The second sentence explains its tests.']
    gemini.call_model_structured.return_value = split
    assert await ResponseParser(gemini).parse(RESPONSE) == split

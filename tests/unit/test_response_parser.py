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
    gemini.call_model_structured.return_value = ["one", "two", "three", "four"]
    result = await ResponseParser(gemini, min_chars_per_turn=40).parse(RESPONSE)
    assert result == ["one", "two three four"]

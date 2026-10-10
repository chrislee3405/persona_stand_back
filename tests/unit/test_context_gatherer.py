import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_collaborate.context_gatherer import ContextGatherer
from app.services.model_collaborate.stage_errors import ModelStageError

# The explicit stage argument identifies phase-2 calls independently of wording.

DOC_TOPICS = [("study", "what they studied")]
SCENARIO_TOPICS = [("greeting", "how they open")]
CANDIDATES = [{"question": "tell me about yourself", "answer": "I studied IT.", "score": 1.0}]


@pytest.mark.parametrize("question,selected,expected", [
    ("Which parts of your experience are relevant to this position?",
     ["deployment", "release testing"], ["deployment", "release testing"]),
    ("What is the weather tomorrow?", [], []),
])
async def test_retrieval_prompt_defers_fact_approval_and_preserves_topic_boundaries(question, selected, expected):
    # Mock output verifies prompt delivery and validation, not live relevance decisions.
    from app.services.model_collaborate.role_context import role_context_section
    client = SimpleNamespace(call_model_structured=AsyncMock(return_value={
        "document_topics": selected, "scenario_topics": [],
    }))
    gatherer, _ = build(gemini=client)
    topics = [("deployment", "Docker and AWS deployment contributions"),
              ("release testing", "CI/CD release verification")]
    result = await gatherer._select_relevant_topics(
        topics, [("teamwork", "collaboration")],
        role_context_section("Role: Junior cloud engineer"), question)
    assert result == (expected, [])
    call = client.call_model_structured.await_args.kwargs
    assert question in call["user_prompt"]
    assert "Junior cloud engineer" in call["user_prompt"]
    prompts = call["system_prompt"] + call["user_prompt"]
    assert "Retrieve plausible supporting material" in prompts
    assert "grounding will decide" in prompts
    assert "Exclude unrelated or redundant topics" in prompts
    assert "highly confident" not in prompts
    assert "When in doubt, leave it out" not in prompts
    assert call["schema"]["properties"]["document_topics"]["items"]["enum"] == [t[0] for t in topics]


@pytest.mark.parametrize("role", [None, "", "   ", "Role: Junior AI developer", "Role: Junior cloud engineer"])
async def test_offered_role_reaches_topic_selection_and_example_reranking(role):
    gemini = FakeGemini()
    question = "Which experience is relevant to this position?"
    gatherer, _ = build(gemini=gemini, conversations=FakeConversations(
        pending=[message(0, "user", question)]))
    await gatherer.gather(question, "conv", job_context=role)
    for stage in ("topics", "example"):
        prompt = gemini.prompts[stage]
        assert ("Role context (quoted data" in prompt) == bool(role and role.strip())
        if role and role.strip():
            assert role in prompt
            assert "never claims that the candidate has its required skills" in prompt


class RecordingSession:
    """
    An AsyncSession stand-in that fails the test if two reads overlap on it.

    The real session is a single connection: concurrent use raises
    InterfaceError at runtime. Here it is turned into an assertion, so a
    future change that moves a query into the concurrent phase is caught by
    the suite rather than in production.
    """

    def __init__(self, rows_by_table=None):
        self.new = self.dirty = self.deleted = set()
        self.commits = 0
        self.in_flight = 0
        self.overlapped = False
        self.rows_by_table = rows_by_table or {}
        self.queries = 0

    async def commit(self):
        self.commits += 1

    async def execute(self, statement):
        self.queries += 1
        self.in_flight += 1
        if self.in_flight > 1:
            self.overlapped = True
        try:
            # Yield to the loop: a genuinely concurrent caller gets its turn
            # here, which is what makes the overlap detectable at all.
            await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        froms = statement.get_final_froms()
        table = froms[0].name if froms else ""
        if table == "model_configuration":
            from app.models.model_configuration import MODEL_COLUMNS
            return _Result([{ "id": 1, "provider": "gemini", "default_model": "test-model",
                              "use_priority": 1, **dict.fromkeys(MODEL_COLUMNS)}])
        return _Result(self.rows_by_table.get(table, []))


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def mappings(self):
        return self

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class FakeGemini:
    """Records when each model call starts and finishes, so overlap is visible."""

    def __init__(self, delay=0.05, topic_error=None, example_error=None,
                 readiness_error=None, decision="respond"):
        self.delay = delay
        self.topic_error = topic_error
        self.example_error = example_error
        self.readiness_error = readiness_error
        self.decision = decision
        self.events = []
        self.prompts = {}
        self.cancelled = []

    def configure(self, configuration):
        self.configuration = configuration

    async def call_model_structured(self, stage, user_prompt, system_prompt, schema):
        assert stage in {"topics", "readiness", "example"}
        which = stage
        self.events.append(("start", which))
        self.prompts[which] = user_prompt
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.append(which)
            raise
        self.events.append(("end", which))
        if which == "topics":
            if self.topic_error:
                raise self.topic_error
            return {"document_topics": ["study"], "scenario_topics": ["greeting"]}
        if which == "readiness":
            if self.readiness_error:
                raise self.readiness_error
            return {"decision": self.decision, "reason": "because"}
        if self.example_error:
            raise self.example_error
        return "0"


class FakeBM25:
    def __init__(self, candidates=CANDIDATES):
        self.candidates = candidates

    async def find_similar_questions(self, user_message, top_k=3):
        return list(self.candidates)


def message(order_index, sender, text):
    return SimpleNamespace(order_index=order_index, sender=sender, text=text)


class FakeConversations:
    def __init__(self, summary="a summary", messages=None, pending=None):
        self.summary = summary
        self.messages = messages if messages is not None else []
        self.pending = pending if pending is not None else [message(0, "user", "how are you")]

    async def get_conversation_unlocked(self, conversation_id):
        return SimpleNamespace(summary=self.summary)

    async def get_recent_messages(self, conversation_id, exclude_last=True):
        messages = list(self.messages)
        return messages[:-1] if exclude_last and messages else messages

    async def get_recent_reply_openings(self, conversation_id):
        return ["I'd start by looking at the logs."]

    async def get_pending_user_messages(self, conversation_id):
        return list(self.pending)


def build(gemini=None, session=None, bm25=None, conversations=None):
    if gemini is not None and not hasattr(gemini, "configure"):
        gemini.configure = lambda configuration: None
    personality = SimpleNamespace(
        legal_name="A Candidate", prefer_name="Chris",
        culture_background="somewhere", core_personality="be yourself",
    )
    session = session or RecordingSession({
        "doc_reference": [("study", "what they studied")],
        "scenario_reference": [("greeting", "how they open")],
        "personality_reference": [personality],
    })
    return ContextGatherer(
        db=session,
        ai_service=gemini or FakeGemini(),
        bm25_service=bm25 or FakeBM25(),
        conversation_service=conversations or FakeConversations(),
    ), session


async def test_the_three_model_calls_overlap():
    gemini = FakeGemini(delay=0.05)
    gatherer, _ = build(gemini)

    await gatherer.gather("how are you", "conv-1")

    # All three start before any ends: the whole point of the phase.
    assert [e[1] for e in gemini.events[:3]] == ["readiness", "topics", "example"]
    assert {e[1] for e in gemini.events if e[0] == "end"} == {"readiness", "topics", "example"}


async def test_concurrent_phase_is_faster_than_three_round_trips():
    delay = 0.1
    gemini = FakeGemini(delay=delay)
    gatherer, _ = build(gemini)

    loop = asyncio.get_running_loop()
    start = loop.time()
    await gatherer.gather("how are you", "conv-1")
    elapsed = loop.time() - start

    # Sequential would be 3 * delay. Allow generous slack for a slow runner
    # while still failing if the calls were serialised.
    assert elapsed < delay * 2.5


async def test_no_database_reads_overlap():
    session = RecordingSession({
        "doc_reference": [("study", "what they studied")],
        "scenario_reference": [("greeting", "how they open")],
        "personality_reference": [SimpleNamespace(
            legal_name="A Candidate", prefer_name="Chris",
            culture_background="somewhere", core_personality="be yourself",
        )],
    })
    gatherer, _ = build(session=session)

    await gatherer.gather("how are you", "conv-1")

    assert session.overlapped is False
    assert session.queries > 0


async def test_topic_call_failure_stops_before_reading_selected_references():
    gemini = FakeGemini(topic_error=TimeoutError("simulated"))
    gatherer, session = build(gemini)
    with pytest.raises(ModelStageError) as error:
        await gatherer.gather("how are you", "conv-1")
    assert error.value.stage == "topics"
    assert error.value.reason == "call_failed"
    assert session.queries == 4  # Configuration plus the three reference reads.


async def test_example_call_failure_still_returns_the_topics():
    gemini = FakeGemini(example_error=TimeoutError("simulated"))
    gatherer, _ = build(gemini)

    context = await gatherer.gather("how are you", "conv-1")

    assert context["similar_examples"] == []
    assert context["example_failure"] == "call_failed"
    assert context["doc_topic_list"] == ["study"]
    assert context["scenario_topic_list"] == ["greeting"]


async def test_both_searches_failing_does_not_produce_a_knowledge_gap():
    gemini = FakeGemini(topic_error=RuntimeError("a"), example_error=RuntimeError("b"))
    gatherer, _ = build(gemini)

    with pytest.raises(ModelStageError, match="stage=topics"):
        await gatherer.gather("how are you", "conv-1")


async def test_whole_turn_timeout_cancels_both_model_calls():
    gemini = FakeGemini(delay=5)
    gatherer, _ = build(gemini)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gatherer.gather("how are you", "conv-1"), timeout=0.05)

    await asyncio.sleep(0)
    assert sorted(gemini.cancelled) == ["example", "readiness", "topics"]


async def test_no_bm25_candidates_skips_the_example_call():
    gemini = FakeGemini()
    gatherer, _ = build(gemini, bm25=FakeBM25(candidates=[]))

    context = await gatherer.gather("how are you", "conv-1")

    assert context["similar_examples"] == []
    assert [e for e in gemini.events if e[1] == "example"] == []


# --- Pending-message grouping ------------------------------------------------
# A message the gate decided to wait on stays pending, so the NEXT turn has to
# answer both together: the group is the question, and it must not also appear
# in the history as though it had already been dealt with.


async def test_pending_group_becomes_the_message_the_turn_answers():
    conversations = FakeConversations(
        messages=[
            message(0, "user", "hi"),
            message(1, "backend", "hey"),
            message(2, "user", "two questions, first"),
            message(3, "user", "what did you study"),
        ],
        pending=[
            message(2, "user", "two questions, first"),
            message(3, "user", "what did you study"),
        ],
    )
    gatherer, _ = build(conversations=conversations)

    context = await gatherer.gather("what did you study", "conv-1")

    assert context["effective_user_message"] == "two questions, first\nwhat did you study"
    assert context["evaluated_through"] == 3
    # History stops before the pending group.
    assert [m.order_index for m in context["recent_messages"]] == [0, 1]
    # And the group is what the other two calls were asked about.
    assert "what did you study" in gemini_prompt(gatherer, "readiness")


def gemini_prompt(gatherer, which):
    return gatherer.ai_service.prompts[which]


async def test_single_pending_message_evaluates_only_itself():
    conversations = FakeConversations(
        messages=[message(0, "user", "how are you")],
        pending=[message(0, "user", "how are you")],
    )
    gatherer, _ = build(conversations=conversations)

    context = await gatherer.gather("how are you", "conv-1")

    assert context["effective_user_message"] == "how are you"
    assert context["evaluated_through"] == 0
    assert context["recent_messages"] == []


async def test_readiness_failure_falls_back_to_responding():
    gemini = FakeGemini(readiness_error=TimeoutError("simulated"))
    gatherer, _ = build(gemini)

    context = await gatherer.gather("how are you", "conv-1")

    # Failing towards silence would leave the visitor with nothing at all.
    assert context["readiness"]["decision"] == "respond"


@pytest.mark.parametrize("decision", ["wait", "no_reply"])
async def test_readiness_decision_is_carried_into_the_context(decision):
    gemini = FakeGemini(decision=decision)
    gatherer, _ = build(gemini)

    context = await gatherer.gather("ok thanks", "conv-1")

    assert context["readiness"]["decision"] == decision


# --- A held turn must be SHORT ----------------------------------------------
# The turn returns when gather returns, so anything gather waits for is time
# the visitor spends watching a typing indicator for a reply that is not
# coming. A non-respond verdict therefore ends phase 2 immediately.


class SlowRetrieval(FakeGemini):
    """Readiness answers quickly; the retrieval calls do not. The real shape:
    topic selection carries every topic description and is the slow one."""

    async def call_model_structured(self, stage, user_prompt, system_prompt, schema):
        if stage != "readiness":
            self.delay = 0.4
        return await super().call_model_structured(stage, user_prompt, system_prompt, schema)


@pytest.mark.parametrize("decision", ["wait", "no_reply"])
async def test_a_held_turn_skips_topic_selection_and_reranking(decision):
    gemini = SlowRetrieval(decision=decision, delay=0.02)
    gatherer, session = build(gemini)

    context = await gatherer.gather("two questions, first", "conv-1")

    # All three were started together, but only readiness was waited on --
    # the other two were cancelled mid-flight the moment the verdict landed.
    assert [e[1] for e in gemini.events if e[0] == "start"] == ["readiness", "topics", "example"]
    assert [e[1] for e in gemini.events if e[0] == "end"] == ["readiness"]
    assert context["doc_topic_list"] == []
    assert context["similar_examples"] == []
    assert context["doc_reference_section"] == "No document reference available."


@pytest.mark.parametrize("decision", ["wait", "no_reply"])
async def test_a_held_turn_costs_one_round_trip_not_two(decision):
    # The readiness call is deliberately the FASTEST here, which is the case
    # that used to cost the most: the verdict was in early and the turn waited
    # for topic selection anyway.
    gemini = SlowRetrieval(decision=decision, delay=0.05)
    gatherer, _ = build(gemini)

    loop = asyncio.get_running_loop()
    start = loop.time()
    await gatherer.gather("ok thanks", "conv-1")
    elapsed = loop.time() - start

    # Waiting for the slow call too would put this at 0.4s+.
    assert elapsed < 0.25


async def test_a_replying_turn_still_uses_both_other_calls():
    gemini = FakeGemini(decision="respond")
    gatherer, _ = build(gemini)

    context = await gatherer.gather("what did you study", "conv-1")

    assert sorted(e[1] for e in gemini.events if e[0] == "end") == ["example", "readiness", "topics"]
    assert context["doc_topic_list"] == ["study"]
    assert context["similar_examples"] == [CANDIDATES[0]]


# --- Continue turns ------------------------------------------------------------
# The visitor went quiet after a "wait". The held group is answered as it
# stands: the gate is not asked again, because it would only say "wait" again.


async def test_continue_answers_the_held_group_without_asking_the_gate():
    conversations = FakeConversations(
        messages=[message(0, "user", "two questions, first")],
        pending=[message(0, "user", "two questions, first")],
    )
    gemini = FakeGemini(decision="wait")
    gatherer, _ = build(gemini, conversations=conversations)

    context = await gatherer.gather("", "conv-1", skip_readiness=True)

    assert context["readiness"]["decision"] == "respond"
    assert context["effective_user_message"] == "two questions, first"
    assert context["evaluated_through"] == 0
    assert "readiness" not in {e[1] for e in gemini.events}
    # Retrieval runs as for any replying turn.
    assert context["doc_topic_list"] == ["study"]


async def test_continue_with_nothing_held_makes_no_model_call():
    gemini = FakeGemini()
    gatherer, _ = build(gemini, conversations=FakeConversations(pending=[]))

    context = await gatherer.gather("", "conv-1", skip_readiness=True)

    assert context["readiness"]["decision"] == "no_reply"
    assert context["evaluated_through"] == -1
    assert gemini.events == []

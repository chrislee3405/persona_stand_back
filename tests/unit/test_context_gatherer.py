import asyncio
from types import SimpleNamespace

import pytest

from app.services.model_collaborate.context_gatherer import ContextGatherer

# What the three phase-2 model calls are: ContextGatherer makes exactly three
# Gemini calls per turn, told apart by which system prompt they carry.
TOPIC_CALL = "identify which topics"
EXAMPLE_CALL = "picking which stored interview question"
READINESS_CALL = "should reply YET"

DOC_TOPICS = [("study", "what they studied")]
SCENARIO_TOPICS = [("greeting", "how they open")]
CANDIDATES = [{"question": "tell me about yourself", "answer": "I studied IT.", "score": 1.0}]


class RecordingSession:
    """
    An AsyncSession stand-in that fails the test if two reads overlap on it.

    The real session is a single connection: concurrent use raises
    InterfaceError at runtime. Here it is turned into an assertion, so a
    future change that moves a query into the concurrent phase is caught by
    the suite rather than in production.
    """

    def __init__(self, rows_by_table=None):
        self.in_flight = 0
        self.overlapped = False
        self.rows_by_table = rows_by_table or {}
        self.queries = 0

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
        return _Result(self.rows_by_table.get(table, []))


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

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

    async def call_model_structured(self, model_name, user_prompt, system_prompt, schema):
        if TOPIC_CALL in system_prompt:
            which = "topics"
        elif READINESS_CALL in system_prompt:
            which = "readiness"
        else:
            which = "example"
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

    async def get_pending_user_messages(self, conversation_id):
        return list(self.pending)


def build(gemini=None, session=None, bm25=None, conversations=None):
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
        gemini_service=gemini or FakeGemini(),
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


async def test_topic_call_failure_still_returns_the_example():
    gemini = FakeGemini(topic_error=TimeoutError("simulated"))
    gatherer, _ = build(gemini)

    context = await gatherer.gather("how are you", "conv-1")

    assert context["doc_topic_list"] == []
    assert context["scenario_topic_list"] == []
    assert context["doc_reference_section"] == "No document reference available."
    assert context["similar_examples"] == [CANDIDATES[0]]


async def test_example_call_failure_still_returns_the_topics():
    gemini = FakeGemini(example_error=TimeoutError("simulated"))
    gatherer, _ = build(gemini)

    context = await gatherer.gather("how are you", "conv-1")

    assert context["similar_examples"] == []
    assert context["doc_topic_list"] == ["study"]
    assert context["scenario_topic_list"] == ["greeting"]


async def test_both_calls_failing_still_produces_a_usable_context():
    gemini = FakeGemini(topic_error=RuntimeError("a"), example_error=RuntimeError("b"))
    gatherer, _ = build(gemini)

    context = await gatherer.gather("how are you", "conv-1")

    assert context["similar_examples"] == []
    assert context["doc_reference_section"] == "No document reference available."
    assert context["scenario_reference_section"] == "No scenario reference available."
    # The parts that do not depend on either call survive untouched.
    assert context["prefer_name"] == "Chris"
    assert context["summary"] == "a summary"


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
    return gatherer.gemini_service.prompts[which]


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

    async def call_model_structured(self, model_name, user_prompt, system_prompt, schema):
        if READINESS_CALL not in system_prompt:
            self.delay = 0.4
        return await super().call_model_structured(model_name, user_prompt, system_prompt, schema)


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

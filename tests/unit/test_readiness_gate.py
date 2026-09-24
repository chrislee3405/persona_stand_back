"""
The readiness gate as the turn sees it: what runs, what is persisted, and how
the pending cursor moves.

These exercise ModelCollaborateService and ChatService against fakes rather
than the database, so they cover the decision logic and the ordering rules --
the parts a wrong change breaks silently. The SQL behind
get_pending_user_messages / mark_handled_up_to is covered by the integration
suite, which has a real Postgres.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.constants import Sender
from app.services.chat_service import ChatService
from app.services.model_collaborate_service import ModelCollaborateService, TurnOutcome


@pytest.fixture(autouse=True)
def fake_session_work(monkeypatch):
    @asynccontextmanager
    async def unlocked(session_id):
        yield
    monkeypatch.setattr('app.services.chat_service.session_work', unlocked)


def message(order_index, sender="user", text="hello"):
    return SimpleNamespace(order_index=order_index, sender=sender, text=text)


class FakeGatherer:
    """Returns a ready-made context, and records what it was asked for."""

    def __init__(self, decision="respond", evaluated_through=0, pending=None):
        self.decision = decision
        self.evaluated_through = evaluated_through
        self.pending = pending or [message(0)]
        self.calls = 0
        self.skip_readiness = None

    async def gather(self, user_message, conversation_id, skip_readiness=False):
        self.calls += 1
        self.skip_readiness = skip_readiness
        return {
            "readiness": {"decision": self.decision, "reason": "because"},
            "effective_user_message": "\n".join(m.text for m in self.pending),
            "pending_messages": self.pending,
            "evaluated_through": self.evaluated_through,
            "similar_examples": [],
            "doc_reference_section": "",
            "scenario_reference_section": "",
            "doc_topic_list": ["study"],
            "scenario_topic_list": [],
            "candidate_identity": "",
            "core_personality": "",
            "prefer_name": "Chris",
            "recent_messages": [],
            "summary": None,
        }


class CountingStage:
    """Stands in for any downstream stage, counting how often it ran."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.result


def build_model_service(decision="respond", evaluated_through=0, pending=None):
    service = ModelCollaborateService.__new__(ModelCollaborateService)
    service.context_gatherer = FakeGatherer(decision, evaluated_through, pending)
    service.grounding_service = SimpleNamespace(
        ground=CountingStage({"question_type": "factual", "coverage": "full", "facts": ["f"], "missing": ""})
    )
    service.prompt_builder = SimpleNamespace(build_reply=lambda *a, **k: ("sys", "user"))
    service._generate_reply = CountingStage("a reply")
    service.response_gate = SimpleNamespace(check=CountingStage("a reply"))
    service.response_parser = SimpleNamespace(parse=CountingStage(["a reply"]))
    return service


async def test_respond_runs_the_whole_pipeline():
    service = build_model_service(decision="respond", evaluated_through=3)

    outcome = await service.model_orchestration("hi", "conv-1", "sess-1", "guest")

    assert outcome.decision == "respond"
    assert outcome.reply_turns == ["a reply"]
    assert outcome.evaluated_through == 3
    assert service.grounding_service.ground.calls == 1
    assert service._generate_reply.calls == 1
    assert service.response_gate.check.calls == 1
    assert service.response_parser.parse.calls == 1


@pytest.mark.parametrize("decision", ["wait", "no_reply"])
async def test_non_respond_skips_grounding_generation_verification_and_splitting(decision):
    service = build_model_service(decision=decision, evaluated_through=2)

    outcome = await service.model_orchestration("hi", "conv-1", "sess-1", "guest")

    assert outcome.decision == decision
    assert outcome.reply_text is None
    assert outcome.reply_turns == []
    # The whole point of the gate: none of these cost anything on this turn.
    assert service.grounding_service.ground.calls == 0
    assert service._generate_reply.calls == 0
    assert service.response_gate.check.calls == 0
    assert service.response_parser.parse.calls == 0
    # The cursor still knows how far the run looked.
    assert outcome.evaluated_through == 2


async def test_the_pending_group_is_what_gets_grounded():
    service = build_model_service(pending=[message(2, text="first part"), message(3, text="and the rest")])
    seen = {}

    async def record(effective_message, context):
        seen["message"] = effective_message
        return {"question_type": "factual", "coverage": "full", "facts": [], "missing": ""}

    service.grounding_service.ground = record

    await service.model_orchestration("and the rest", "conv-1", "sess-1", "guest")

    assert seen["message"] == "first part\nand the rest"


# --- ChatService: persistence, the cursor, and stale runs --------------------


class FakeConversationService:
    def __init__(self, latest_user_index=0):
        self.latest_user_index = latest_user_index
        self.appended = []
        self.handled_up_to = None
        self.handled_calls = 0

    async def publish_turn(self, **values):
        if self.latest_user_index > values['evaluated_through']:
            return 'superseded'
        decision = values['decision']
        if decision == 'wait':
            return decision
        # Model the publication boundary, not independent partial commits.
        await self.mark_handled_up_to(values['conversation_id'], values['evaluated_through'])
        if decision == 'respond':
            await self.append_message(values['conversation_id'], values['code'], values['session_id'],
                                      values['reply_sender'], values['reply_text'])
            if values['reply_sender'] == Sender.SYSTEM:
                for pending in getattr(self, 'pending', []):
                    await self.retag_message_sender(pending, Sender.UNANSWERED_USER)
        return decision

    async def discard_pending_group(self, conversation_id, session_id):
        if not hasattr(self, 'pending'):
            return
        self.assert_ownership(await self.get_conversation_unlocked(conversation_id), session_id)
        if self.pending:
            for pending in self.pending:
                await self.retag_message_sender(pending, Sender.UNANSWERED_USER)
            await self.mark_handled_up_to(conversation_id, self.pending[-1].order_index)

    async def append_message(self, conversation_id, code, session_id, sender, text, **kwargs):
        self.appended.append((sender, text))
        return message(0, sender, text), conversation_id or "conv-1"

    async def get_latest_user_order_index(self, conversation_id):
        return self.latest_user_index

    async def mark_handled_up_to(self, conversation_id, order_index):
        self.handled_calls += 1
        self.handled_up_to = order_index

    async def retag_message_sender(self, message, sender):
        pass


class FakeRateControl:
    def __init__(self, continue_error=None):
        self.released = 0
        self.daily_units = 0
        self.continue_slots = 0
        self.continue_error = continue_error

    async def reserve_slot(self, db, session_id, tier):
        self.daily_units += 1

    def reserve_continue_slot(self, session_id, tier):
        if self.continue_error:
            raise self.continue_error
        self.continue_slots += 1

    async def reserve_ip_slot(self, db, client_ip):
        pass

    def release_slot(self, session_id):
        self.released += 1

    def release_ip_slot(self, client_ip):
        pass

    @asynccontextmanager
    async def turn(self, session_id, tier):
        yield


class FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, **kwargs):
        self.tasks.append((func, kwargs))


def build_chat_service(outcome, conversations=None):
    service = ChatService.__new__(ChatService)
    service.db = SimpleNamespace(rollback=_noop)
    service.conversation_service = conversations or FakeConversationService()
    service.model_service = SimpleNamespace(model_orchestration=_returning(outcome))
    service.summarization_service = SimpleNamespace(summarize_conversation_if_needed=None)
    service.privacy_gate_service = SimpleNamespace(check=lambda text: None)
    service.rate_control_service = FakeRateControl()
    service.consent_service = SimpleNamespace(check=_noop_arg)
    return service


async def _noop():
    pass


async def _noop_arg(*args, **kwargs):
    pass


def _returning(outcome):
    async def run(*args, **kwargs):
        return outcome
    return run


async def run_turn(service, text="hello"):
    return await service.handle_chat_turn(
        session_id="sess-1", code=None, conversation_id="conv-1", user_text=text,
        background_tasks=FakeBackgroundTasks(), client_ip="127.0.0.1",
    )


async def test_wait_keeps_the_messages_pending_and_persists_no_reply():
    conversations = FakeConversationService()
    service = build_chat_service(TurnOutcome(decision="wait", evaluated_through=3), conversations)

    result = await run_turn(service)

    assert result["status"] == "wait"
    assert result["turns"] == []
    assert result["userMessageKept"] is True
    # Only the visitor's own message was written, and the cursor did not move:
    # the next message is judged together with this one.
    assert [sender for sender, _ in conversations.appended] == [Sender.USER]
    assert conversations.handled_calls == 0


async def test_no_reply_marks_the_messages_handled_without_replying():
    conversations = FakeConversationService()
    service = build_chat_service(TurnOutcome(decision="no_reply", evaluated_through=3), conversations)

    result = await run_turn(service, "ok thanks")

    assert result["status"] == "no_reply"
    assert result["turns"] == []
    assert [sender for sender, _ in conversations.appended] == [Sender.USER]
    assert conversations.handled_up_to == 3


async def test_respond_persists_the_reply_then_advances_the_cursor():
    conversations = FakeConversationService(latest_user_index=3)
    service = build_chat_service(
        TurnOutcome(
            decision="respond", evaluated_through=3,
            reply_text="here you go", reply_turns=["here you go"], doc_topics=["study"],
        ),
        conversations,
    )

    result = await run_turn(service)

    assert result["status"] == "respond"
    assert result["turns"] == ["here you go"]
    assert [sender for sender, _ in conversations.appended] == [Sender.USER, Sender.BACKEND]
    assert conversations.handled_up_to == 3


async def test_a_reply_overtaken_by_a_newer_message_is_not_published():
    # The run evaluated up to order_index 3; message 4 arrived while it was
    # generating. Publishing now would answer half the question.
    conversations = FakeConversationService(latest_user_index=4)
    service = build_chat_service(
        TurnOutcome(
            decision="respond", evaluated_through=3,
            reply_text="stale answer", reply_turns=["stale answer"],
        ),
        conversations,
    )

    result = await run_turn(service)

    assert result["status"] == "superseded"
    assert result["turns"] == []
    # No persona row, and the cursor stays put so the newer turn re-reads the
    # whole group and answers all of it.
    assert [sender for sender, _ in conversations.appended] == [Sender.USER]
    assert conversations.handled_calls == 0


async def test_a_failed_publication_does_not_return_or_append_a_reply():
    class FailingCursor(FakeConversationService):
        async def mark_handled_up_to(self, conversation_id, order_index):
            raise RuntimeError("database gone")

    conversations = FailingCursor(latest_user_index=3)
    service = build_chat_service(
        TurnOutcome(
            decision="respond", evaluated_through=3,
            reply_text="here you go", reply_turns=["here you go"],
        ),
        conversations,
    )

    with pytest.raises(RuntimeError, match="database gone"):
        await run_turn(service)
    assert [sender for sender, _ in conversations.appended] == [Sender.USER]


async def test_a_cancelled_turn_is_reported_as_a_failure_not_a_decision():
    # The whole-turn deadline in handle_chat_turn cancels orchestration. That
    # must stay a failed turn (retag, no persona row), not be mistaken for one
    # of the gate's own no-reply decisions.
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    conversations = FakeConversationService()
    service = build_chat_service(TurnOutcome(decision="respond", evaluated_through=0), conversations)
    service.model_service = SimpleNamespace(model_orchestration=hang)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.services.chat_service.TURN_DEADLINE_SECONDS", 0.05)
        result = await run_turn(service)

    # "respond": there IS something to show (the apology), which is what the
    # frontend reads the status for. A gate decision would mean silence.
    assert result["status"] == "respond"
    assert result["turns"] == ["Sorry, that took too long to answer. Please try again."]
    assert result["userMessageKept"] is False
    assert conversations.handled_calls == 0


# --- A rejected message ends a wait -----------------------------------------
# The frontend marks a held bubble "withheld" -- stored, but no longer part of
# the conversation -- when the message that should have completed it is
# rejected. That is only true if the server stops waiting on it too.


class PendingConversationService(FakeConversationService):
    def __init__(self, pending=None, owner="sess-1", latest_user_index=0):
        super().__init__(latest_user_index=latest_user_index)
        self.pending = pending if pending is not None else [message(2), message(3)]
        self.owner = owner
        self.retagged = []

    async def retag_message_sender(self, message, sender):
        self.retagged.append((message.order_index, sender))
        message.sender = sender

    async def get_conversation_unlocked(self, conversation_id):
        return SimpleNamespace(conversation_id=conversation_id, owner_session_id=self.owner)

    def assert_ownership(self, conversation, session_id):
        if conversation.owner_session_id != session_id:
            raise PermissionError("not yours")

    async def get_pending_user_messages(self, conversation_id):
        return list(self.pending)


async def reject_a_turn(conversations, text="x" * 5000, session_id="sess-1"):
    service = build_chat_service(TurnOutcome(decision="respond", evaluated_through=0), conversations)
    with pytest.raises(Exception):
        await service.handle_chat_turn(
            session_id=session_id, code=None, conversation_id="conv-1", user_text=text,
            background_tasks=FakeBackgroundTasks(), client_ip="127.0.0.1",
        )


async def test_a_rejected_message_releases_the_held_group():
    conversations = PendingConversationService()

    await reject_a_turn(conversations)

    # Held through order_index 3, so nothing below it stays pending: the
    # thought those messages were waiting to complete never arrived.
    assert conversations.handled_up_to == 3


async def test_released_messages_leave_the_prompt_history():
    # The frontend marks these bubbles "Not answered". If the rows stayed
    # Sender.USER they would keep steering every later reply -- a red bubble
    # the persona is still reading. Retagging is what makes the mark true.
    conversations = PendingConversationService()

    await reject_a_turn(conversations)

    assert conversations.retagged == [(2, Sender.UNANSWERED_USER), (3, Sender.UNANSWERED_USER)]


async def test_a_rejected_message_releases_nothing_when_no_group_is_held():
    # This is the no_reply case: that turn already advanced the cursor, so
    # there is no pending group and the rejection has nothing to do with it.
    conversations = PendingConversationService(pending=[])

    await reject_a_turn(conversations)

    assert conversations.handled_calls == 0


async def test_a_rejected_message_cannot_release_another_session_group():
    conversations = PendingConversationService(owner="someone-else")

    await reject_a_turn(conversations, session_id="sess-1")

    assert conversations.handled_calls == 0


async def test_a_failure_to_release_does_not_replace_the_rejection():
    class FailingRelease(PendingConversationService):
        async def mark_handled_up_to(self, conversation_id, order_index):
            raise RuntimeError("database gone")

    service = build_chat_service(
        TurnOutcome(decision="respond", evaluated_through=0), FailingRelease()
    )

    # The visitor must still get the 413 that started all this, not a 500
    # from the tidy-up behind it.
    from app.services.chat_service import MessageTooLongError

    with pytest.raises(MessageTooLongError):
        await service.handle_chat_turn(
            session_id="sess-1", code=None, conversation_id="conv-1", user_text="x" * 5000,
            background_tasks=FakeBackgroundTasks(), client_ip="127.0.0.1",
        )


async def test_a_rejected_first_message_has_no_conversation_to_release():
    conversations = PendingConversationService()
    service = build_chat_service(TurnOutcome(decision="respond", evaluated_through=0), conversations)

    with pytest.raises(Exception):
        await service.handle_chat_turn(
            session_id="sess-1", code=None, conversation_id=None, user_text="x" * 5000,
            background_tasks=FakeBackgroundTasks(), client_ip="127.0.0.1",
        )

    assert conversations.handled_calls == 0


async def test_a_withheld_reply_retags_every_message_it_answered():
    # The gate fallback means the persona never really answered. The question
    # it stood in for is the whole held group, so all of it leaves the
    # conversation -- ConversationService._drop_withheld_turns only ever
    # paired off the single row before the notice, which left the earlier
    # held messages steering later prompts while their bubbles read
    # "Not answered".
    conversations = PendingConversationService(latest_user_index=3)
    group = [message(2, text="what about"), message(3, text="and the rest")]
    service = build_chat_service(
        TurnOutcome(
            decision="respond", evaluated_through=3, pending_messages=group,
            reply_text="Sorry, I couldn't put together a suitable reply.",
            reply_turns=["Sorry, I couldn't put together a suitable reply."],
        ),
        conversations,
    )

    result = await run_turn(service)

    assert result["userMessageKept"] is False
    assert conversations.retagged == [(2, Sender.UNANSWERED_USER), (3, Sender.UNANSWERED_USER)]


async def test_a_normal_reply_leaves_the_messages_it_answered_in_the_conversation():
    conversations = PendingConversationService(latest_user_index=3)
    group = [message(2, text="what about"), message(3, text="the projects")]
    service = build_chat_service(
        TurnOutcome(
            decision="respond", evaluated_through=3, pending_messages=group,
            reply_text="here you go", reply_turns=["here you go"],
        ),
        conversations,
    )

    result = await run_turn(service)

    assert result["userMessageKept"] is True
    assert conversations.retagged == []


async def test_a_failed_turn_releases_the_whole_held_group():
    # The visitor is told every bubble in the group was not sent, and the
    # thought has to be sent again. An earlier held message left pending
    # would be answered later anyway, alongside whatever came next.
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    conversations = PendingConversationService(pending=[message(2), message(3)])
    service = build_chat_service(TurnOutcome(decision="respond", evaluated_through=3), conversations)
    service.model_service = SimpleNamespace(model_orchestration=hang)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.services.chat_service.TURN_DEADLINE_SECONDS", 0.05)
        result = await run_turn(service)

    assert result["userMessageKept"] is False
    assert [order for order, _ in conversations.retagged[:2]] == [2, 3]
    assert conversations.handled_up_to == 3


# --- Continue: the visitor went quiet after a "wait" --------------------------
# The frontend asks for this once the input has sat empty and untouched after a
# "wait". The held group is answered as it stands, without a new message.


class RecordingOrchestration:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def __call__(self, user_message, conversation_id, session_id, tier, **kwargs):
        self.calls.append((user_message, kwargs))
        return self.outcome


def build_continue_service(outcome, conversations, rate_control=None):
    service = build_chat_service(outcome, conversations)
    service.model_service = SimpleNamespace(model_orchestration=RecordingOrchestration(outcome))
    service.rate_control_service = rate_control or FakeRateControl()
    return service


async def run_continue(service, session_id="sess-1"):
    return await service.handle_continue_turn(
        session_id=session_id, code=None, conversation_id="conv-1",
        background_tasks=FakeBackgroundTasks(),
    )


async def test_continue_answers_the_held_group_without_a_new_message():
    group = [message(2, text="two questions, first")]
    conversations = PendingConversationService(pending=group, latest_user_index=2)
    service = build_continue_service(
        TurnOutcome(
            decision="respond", evaluated_through=2, pending_messages=group,
            reply_text="here you go", reply_turns=["here you go"],
        ),
        conversations,
    )

    result = await run_continue(service)

    assert result["status"] == "respond"
    assert result["turns"] == ["here you go"]
    assert result["userMessageKept"] is True
    # No new text: only the persona's reply is written, and the gate is skipped.
    assert [sender for sender, _ in conversations.appended] == [Sender.BACKEND]
    assert service.model_service.model_orchestration.calls == [("", {"skip_readiness": True})]
    assert conversations.handled_up_to == 2
    # An in-flight slot, but no daily unit: the held message already paid.
    assert service.rate_control_service.continue_slots == 1
    assert service.rate_control_service.daily_units == 0
    assert service.rate_control_service.released == 1


async def test_continue_with_nothing_held_calls_no_model():
    conversations = PendingConversationService(pending=[])
    service = build_continue_service(TurnOutcome(decision="respond", evaluated_through=0), conversations)

    result = await run_continue(service)

    assert result["status"] == "no_reply"
    assert result["turns"] == []
    assert service.model_service.model_orchestration.calls == []
    assert conversations.appended == []


async def test_continue_refuses_another_sessions_conversation():
    conversations = PendingConversationService(owner="someone-else")
    service = build_continue_service(TurnOutcome(decision="respond", evaluated_through=3), conversations)

    with pytest.raises(PermissionError):
        await run_continue(service, session_id="sess-1")

    assert service.model_service.model_orchestration.calls == []
    assert conversations.retagged == []
    assert service.rate_control_service.released == 1


async def test_a_rejected_continue_leaves_the_held_group_pending():
    # Unlike a rejected MESSAGE, a continue is not the continuation the group
    # was waiting for -- failing to run it breaks no thought, so the group is
    # answered with the visitor's next message instead.
    from app.services.rate_control_service import TooManyPendingMessagesError

    conversations = PendingConversationService()
    service = build_continue_service(
        TurnOutcome(decision="respond", evaluated_through=3),
        conversations,
        FakeRateControl(continue_error=TooManyPendingMessagesError("sess-1", "guest")),
    )

    with pytest.raises(TooManyPendingMessagesError):
        await run_continue(service)

    assert conversations.handled_calls == 0
    assert conversations.retagged == []


async def test_a_failed_continue_releases_the_held_group():
    # Generation itself failed: the visitor is told those messages were not
    # answered, so the server must stop holding them -- as on any failed turn.
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    conversations = PendingConversationService(pending=[message(2), message(3)])
    service = build_continue_service(TurnOutcome(decision="respond", evaluated_through=3), conversations)
    service.model_service = SimpleNamespace(model_orchestration=hang)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.services.chat_service.TURN_DEADLINE_SECONDS", 0.05)
        result = await run_continue(service)

    assert result["userMessageKept"] is False
    assert conversations.retagged == [(2, Sender.UNANSWERED_USER), (3, Sender.UNANSWERED_USER)]
    assert conversations.handled_up_to == 3
    # Only the review row: a continue has no visitor row of its own to write.
    assert [sender for sender, _ in conversations.appended] == [Sender.ERROR]


async def test_model_orchestration_passes_skip_readiness_to_the_gatherer():
    service = build_model_service(decision="respond", evaluated_through=2)

    await service.model_orchestration("", "conv-1", "sess-1", "guest", skip_readiness=True)

    assert service.context_gatherer.skip_readiness is True

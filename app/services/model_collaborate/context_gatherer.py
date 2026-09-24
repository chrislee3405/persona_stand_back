import asyncio
import logging
import time

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_trace import trace
from app.constants import DEFAULT_MODEL
from app.database import get_db
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate import turn_metrics
from app.services.model_collaborate.prepare_history import prepare_history
from app.services.model_collaborate.readiness_service import (
    CONTINUE_NOTHING_HELD,
    CONTINUE_VERDICT,
    RESPOND,
    ReadinessService,
)
from app.models.prompt_reference import DocReference, PersonalityReference, ScenarioReference

logger = logging.getLogger(__name__)

# How many BM25 hits to hand the model to re-rank. BM25 is keyword-overlap
# only, so its #1 hit is often a generic "tell me about yourself" row that
# shares words but not meaning; a handful of candidates gives the model
# room to pick the real match (or reject them all). More candidates = a
# slightly better chance the right row is in the set, at more prompt
# tokens per turn.
_EXAMPLE_CANDIDATE_COUNT = 5

# Placeholders for "retrieval matched nothing". Named rather than inlined
# because the retrieval helpers below return them from two places each.
_NO_DOC_REFERENCE = "No document reference available."
_NO_SCENARIO_REFERENCE = "No scenario reference available."


async def _continue_verdict() -> dict:
    """
    Stands in for the readiness call on a continue turn that has something held.

    Parameters:
    - none

    Returns:
    - dict: CONTINUE_VERDICT ("respond") -- goes to gather in the readiness task's place, so the rest of phase 2 runs unchanged
    """
    return dict(CONTINUE_VERDICT)


# ONE call selects from BOTH topic lists. It used to be two structured calls
# per turn -- same template, same history, same message, differing only in the
# topic list and a label -- which sent the conversation history twice and cost
# an extra round trip on every message. The response schema gives each list its
# own enum, so the model still cannot put a scenario topic in the document list.
_FIND_TOPIC_SYSTEM_PROMPT = (
    "Your task is to identify which topics are relevant to the user's current "
    "message. There are two independent lists:\n"
    "- DOCUMENT topics: factual reference material about the candidate.\n"
    "- SCENARIO topics: guidance on how the candidate behaves or approaches a "
    "kind of question.\n"
    "Judge the two lists separately. A message may need topics from both, from "
    "one, or from neither.\n\n"
    "Be conservative: only select a topic if you are highly confident it is "
    "directly relevant to answering the current message -- a loose, "
    "tangential, or merely thematically-similar connection is not enough. "
    "If no topic in a list clears that bar, return an empty array for that "
    "list rather than guessing or including a weak match.\n\n"
    "Return only exact topic strings, each from the list it belongs to -- "
    "never invent a topic, and never return a topic under the other list."
)

_FIND_TOPIC_USER_PROMPT_TEMPLATE = (
    "DOCUMENT topics -- factual reference about the candidate:\n"
    "{doc_topic_descriptions}\n\n"
    "SCENARIO topics -- how the candidate behaves:\n"
    "{scenario_topic_descriptions}\n\n"
    "{history_context}"
    "Current user message: {user_message}\n\n"
    "Using the conversation history strictly as context to understand "
    "references (not as a source of topics on its own), select only the "
    "topic(s) you are highly confident are directly relevant to answering "
    "the current message. When in doubt, leave it out."
)

_SELECT_EXAMPLE_SYSTEM_PROMPT = (
    "You are picking which stored interview question -- if any -- genuinely "
    "matches what the user is asking right now, so its stored answer can "
    "guide the reply.\n\n"
    "The candidates were retrieved by keyword overlap, which is easily "
    "misled by shared generic words such as \"tell\", \"yourself\", "
    "\"interest\", or \"company\". Judge by MEANING, not shared words: keep "
    "a candidate only if it asks for essentially the same thing as the "
    "user's current message. If none of them do, choose \"none\" -- a wrong "
    "match is worse than no match.\n\n"
    "Use the conversation history only to understand what the current "
    "message refers to, not as a question in its own right."
)

_SELECT_EXAMPLE_USER_PROMPT_TEMPLATE = (
    "Candidate stored questions:\n{candidates}\n\n"
    "{history_context}"
    "User's current message: {user_message}\n\n"
    "Reply with the index of the one candidate that matches the current "
    "message in meaning, or \"none\" if none is a real match."
)


def _format_topic_descriptions(topics_data: list[tuple[str, str]]) -> str:
    """
    Renders one topic list into the labelled block the topic-selection prompt reads.

    Parameters:
    - topics_data (list[tuple[str, str]]): (topic, description) pairs — comes from ContextGatherer._select_relevant_topics

    Returns:
    - str: one "Topic: ... / Description: ..." entry per topic, blank-line separated, or a placeholder when the list is empty. Each pair is unambiguously labelled and separated rather than written as a terser "- topic: description" one-liner, because a one-liner would rely on the model parsing colon placement correctly with two lists of many topics in the same prompt.
    """
    if not topics_data:
        return "None available."
    return "\n\n".join(f"Topic: {t[0]}\nDescription: {t[1]}" for t in topics_data)


def _topic_history_context(recent_messages: list | None) -> str:
    """
    Renders the recent-history block the topic-selection prompt reads.

    Parameters:
    - recent_messages (list | None): recent conversation history — comes from ContextGatherer.gather or _find_topic

    Returns:
    - str: up to the last 8 messages (4 exchanges) under a labelled header, or "" when there is no history — goes to _select_relevant_topics. Module level rather than inline in _find_topic because gather now calls _select_relevant_topics directly, and both paths must produce the identical block.
    """
    if not recent_messages:
        return ""
    history_str = prepare_history(recent_messages[-8:], None)
    return f"Recent conversation history for context:\n{history_str}\n\n"



class ContextGatherer:
    def __init__(self, db: AsyncSession = Depends(get_db), gemini_service: GeminiService = Depends(), bm25_service: BM25Service = Depends(), conversation_service: ConversationService = Depends(), readiness_service: ReadinessService | None = None):
        """
        Stores the injected database session and service instances.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db, or passed explicitly by ModelCollaborateService
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly
        - bm25_service (BM25Service): retrieves similar past questions — injected by FastAPI, or passed explicitly
        - conversation_service (ConversationService): reads conversation state — injected by FastAPI, or passed explicitly
        - readiness_service (ReadinessService | None): decides whether the pending messages should be answered yet — defaults to one built on the same Gemini service, so the existing four-argument construction (ModelCollaborateService, the probe scripts) keeps working

        Returns:
        - None: sets self.db, self.gemini_service, self.bm25_service, self.conversation_service, self.readiness_service
        """
        self.db = db
        self.gemini_service = gemini_service
        self.bm25_service = bm25_service
        self.conversation_service = conversation_service
        self.readiness_service = readiness_service or ReadinessService(gemini_service)

    async def gather(self, user_message: str, conversation_id: str, skip_readiness: bool = False) -> dict:
        """
        Collects DB references, similar past questions, and conversation history needed to build a prompt.

        Parameters:
        - user_message (str): the user's current message — comes from ModelCollaborateService.model_orchestration
        - conversation_id (str): the conversation being replied to — comes from ModelCollaborateService.model_orchestration
        - skip_readiness (bool): True on a continue turn — the readiness call is replaced by _continue_verdict, which answers whatever is held without asking the model again

        Returns:
        - dict: readiness, effective_user_message, pending_messages, evaluated_through, similar_examples, doc/scenario reference sections, doc/scenario topic lists, candidate_identity, core_personality, prefer_name, recent_messages, summary — goes to PromptBuilder.build and ModelCollaborateService.model_orchestration

        Runs in three phases: every database read the model calls need, then
        the readiness check, topic selection and example re-ranking together,
        then the reference text for whichever topics were selected. A turn
        the gate holds or ignores stops at the readiness result and skips the
        rest of phase 2 and all of phase 3, so it costs one round trip
        instead of a whole retrieval pass. Phases 1
        and 3 are sequential because they share one AsyncSession; phase 2 is
        concurrent because it touches no session at all. The three model calls
        are independent, so the readiness gate costs a call but not a round
        trip -- it finishes inside the time the other two were taking anyway.

        THE TURN ANSWERS THE PENDING GROUP, not just `user_message`. A message
        the gate previously decided to wait on is still pending, so the next
        message is gathered, grounded and answered together with it; see
        effective_user_message below.
        """
        started = time.perf_counter()

        # ---- Phase 1: every database read this turn needs up front ---------
        #
        # STRICTLY SEQUENTIAL, and not an oversight. All of these run on the
        # one AsyncSession this service was constructed with, and a session is
        # not safe to use from two coroutines at once -- overlapping reads on
        # one connection raise InterfaceError ("another operation is in
        # progress") or, worse, interleave inside the turn's transaction.
        # Nothing here is parallelised without giving each branch its own
        # session first.
        conversation = await self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None

        # Everything the visitor has sent that no reply has dealt with yet --
        # this turn's message, plus anything an earlier turn decided to wait
        # on. The gate judges the group, and a reply answers the group.
        pending_messages = await self.conversation_service.get_pending_user_messages(conversation_id)
        pending_texts = [m.text for m in pending_messages]
        evaluated_through = pending_messages[-1].order_index if pending_messages else -1

        # The message the rest of the pipeline treats as "what was asked".
        # Falls back to the caller's text if the cursor and the rows disagree,
        # so a turn still answers something rather than nothing.
        effective_user_message = "\n".join(pending_texts) if pending_texts else user_message

        # History first: all three model calls in phase 2 read it, to resolve
        # what the message refers to, to disambiguate the re-rank, and to tell
        # a mid-thought fragment from a complete one.
        recent_messages = await self.conversation_service.get_recent_messages(
            conversation_id, exclude_last=not pending_messages
        )
        if pending_messages:
            # The pending group is the QUESTION, so it must not also appear in
            # the history as though it had already been dealt with. Everything
            # strictly before the group is history; the group itself is not.
            first_pending_index = pending_messages[0].order_index
            recent_messages = [m for m in recent_messages if m.order_index < first_pending_index]

        doc_topics_data = await self._get_doc_topics()
        scenario_topics_data = await self._get_scenario_topics()

        # BM25 ranks by keyword overlap, so its top hit is often a generic
        # row that shares words but not meaning. Pull a few candidates and
        # let the model keep the one that genuinely matches, or none.
        bm25_candidates = await self.bm25_service.find_similar_questions(effective_user_message, top_k=_EXAMPLE_CANDIDATE_COUNT)

        candidate_identity, core_personality, prefer_name = await self._get_personality_profile()
        db_elapsed = time.perf_counter() - started

        # ---- Phase 2: the three model calls, concurrently ------------------
        #
        # Readiness, topic selection and example re-ranking all read the
        # history and the pending group, and none reads another's result, so
        # the turn spends one Gemini round trip on all three rather than
        # three. None of them touches the database: phase 1 already fetched
        # everything they read, which is what makes overlapping them safe.
        #
        # A NON-RESPOND VERDICT ENDS THE PHASE EARLY. This used to wait for
        # all three and throw the other two results away, on the reasoning
        # that they were already in flight so cancelling saved tokens but not
        # time. That was wrong, and the chatroom showed it: the turn returns
        # when GATHER returns, so a held turn was still paying for the
        # slowest of the three -- topic selection, which carries every topic
        # description in its prompt and is reliably the slowest. A visitor
        # whose message was merely held still waited a full reply's worth of
        # time, watching the typing indicator, for no reply at all.
        #
        # Awaiting readiness first costs nothing: all three start together,
        # so if readiness is the slow one the others have long since
        # finished. If it comes back "wait" or "no_reply", the other two are
        # cancelled unread and the turn is over in one round trip.
        #
        # A CONTINUE WITH NOTHING HELD stops before any call is started --
        # not merely before one is awaited. A started call is an HTTP request
        # already on its way to Vertex, and cancelling it does not unsend it.
        if skip_readiness and not pending_messages:
            return self._unanswered_context(
                dict(CONTINUE_NOTHING_HELD), effective_user_message, pending_messages, evaluated_through,
                candidate_identity, core_personality, prefer_name, recent_messages, summary,
            )
        readiness_task = asyncio.ensure_future(
            _continue_verdict() if skip_readiness
            else self.readiness_service.check(pending_texts, recent_messages)
        )
        topics_task = asyncio.ensure_future(
            self._select_relevant_topics(
                doc_topics_data, scenario_topics_data,
                _topic_history_context(recent_messages), effective_user_message,
            )
        )
        example_task = asyncio.ensure_future(
            self._select_relevant_example(bm25_candidates, recent_messages, effective_user_message)
        )
        tasks = (readiness_task, topics_task, example_task)
        try:
            readiness = await readiness_task
            if readiness["decision"] != RESPOND:
                # Cancelled, not awaited: these two are HTTP calls that
                # degrade internally rather than raising, so there is no
                # result to collect and no exception to retrieve. Waiting for
                # the cancellation to land would reintroduce the delay this
                # branch exists to remove.
                topics_task.cancel()
                example_task.cancel()
                logger.debug(
                    "Readiness returned %s -- skipping topic selection and example re-ranking.",
                    readiness["decision"],
                )
                return self._unanswered_context(
                    readiness, effective_user_message, pending_messages, evaluated_through,
                    candidate_identity, core_personality, prefer_name, recent_messages, summary,
                )

            (doc_topic_list, scenario_topic_list), similar_examples = await asyncio.gather(
                topics_task, example_task
            )
        except BaseException:
            # A cancelled gather already cancels its children, and all three
            # calls degrade rather than raise -- but if one ever does raise,
            # the others must not be left running past the turn that owns
            # them. CancelledError is included deliberately: the whole-turn
            # deadline in ChatService cancels this coroutine, and the model
            # calls have to stop with it.
            for task in tasks:
                task.cancel()
            raise
        model_elapsed = time.perf_counter() - started - db_elapsed

        # ---- Phase 3: the reads that needed the selection ------------------
        # Sequential for the same session reason as phase 1.
        doc_reference_section = await self._get_doc_references(doc_topic_list)
        scenario_reference_section = await self._get_scenario_references(scenario_topic_list)

        logger.debug(
            "Context gathered in %.3fs (phase 1 db %.3fs, phase 2 models %.3fs, phase 3 db %.3fs)",
            time.perf_counter() - started, db_elapsed, model_elapsed,
            time.perf_counter() - started - db_elapsed - model_elapsed,
        )

        context = {
            "readiness": readiness,
            "effective_user_message": effective_user_message,
            "pending_messages": pending_messages,
            "evaluated_through": evaluated_through,
            "similar_examples": similar_examples,
            "doc_reference_section": doc_reference_section,
            "scenario_reference_section": scenario_reference_section,
            "doc_topic_list": doc_topic_list,
            "scenario_topic_list": scenario_topic_list,
            "candidate_identity": candidate_identity,
            "core_personality": core_personality,
            "prefer_name": prefer_name,
            "recent_messages": recent_messages,
            "summary": summary
        }

        return context

    @staticmethod
    def _unanswered_context(readiness, effective_user_message, pending_messages, evaluated_through,
                            candidate_identity, core_personality, prefer_name, recent_messages, summary) -> dict:
        """
        Builds the context for a turn that is not going to reply.

        Parameters:
        - readiness (dict): the gate's verdict — comes from gather
        - effective_user_message (str), pending_messages (list), evaluated_through (int): what was judged — come from gather
        - candidate_identity (str), core_personality (str), prefer_name (str), recent_messages (list), summary (str | None): everything phase 1 already read — come from gather

        Returns:
        - dict: the same keys gather always returns, with the retrieval-dependent ones empty — goes to ModelCollaborateService.model_orchestration, which reads only `readiness` and `evaluated_through` before returning

        Every key is present even though this caller reads two of them. A
        context that is missing keys depending on how the turn went is a
        KeyError waiting for the first piece of code that logs or inspects
        one, and the empty values are the honest answer here: no topics were
        selected because nothing asked for any.
        """
        return {
            "readiness": readiness,
            "effective_user_message": effective_user_message,
            "pending_messages": pending_messages,
            "evaluated_through": evaluated_through,
            "similar_examples": [],
            "doc_reference_section": _NO_DOC_REFERENCE,
            "scenario_reference_section": _NO_SCENARIO_REFERENCE,
            "doc_topic_list": [],
            "scenario_topic_list": [],
            "candidate_identity": candidate_identity,
            "core_personality": core_personality,
            "prefer_name": prefer_name,
            "recent_messages": recent_messages,
            "summary": summary,
        }

    # --- DB Helper Method for Personality Profile ---

    async def _get_personality_profile(self) -> tuple[str, str, str]:
        """
        Fetches the single personality_reference row and builds the candidate identity block plus the core personality text.

        Parameters:
        - none

        Returns:
        - tuple[str, str, str]: (candidate_identity, core_personality, prefer_name) built from personality_reference's single/first row (the table has no topic to select between), or fallback strings if the table is empty — goes to gather. prefer_name is returned on its own as well as inside candidate_identity because PromptBuilder has to interpolate the bare name into the decline instruction; see _grounding_section.
        """
        result = await self.db.execute(select(PersonalityReference).limit(1))
        row = result.scalar_one_or_none()
        if row is None:
            logger.warning("personality_reference table is empty — using fallback identity and personality text.")
            return (
                "Name: <not configured -- add a row to personality_reference>",
                "No core personality defined.",
                "the candidate"
            )

        candidate_identity = (
            f"Legal name: {row.legal_name}\n"
            f"Preferred name: {row.prefer_name}\n"
            f"Cultural background: {row.culture_background}"
        )
        return candidate_identity, row.core_personality, row.prefer_name

    # --- DB Helper Methods for Doc References ---

    async def _get_doc_topics(self) -> list[tuple[str, str]]:
        """
        Fetches every document reference topic and its description.

        Parameters:
        - none

        Returns:
        - list[tuple[str, str]]: (document_topic, topic_description) pairs — goes to _find_topic
        """
        result = await self.db.execute(
            select(DocReference.document_topic, DocReference.topic_description)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def _get_doc_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted document reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): document topics to include — comes from gather (via _find_topic)

        Returns:
        - str: the concatenated document reference text, or a placeholder if none found — goes to gather

        One query for all topics rather than one per topic (the previous
        _get_doc_from_db helper, now gone). `topics` comes from the model's
        selection and its ORDER is meaningful, so the rows are indexed by
        topic and then walked in the caller's order rather than in whatever
        order the database returns them.
        """
        if not topics:
            return _NO_DOC_REFERENCE

        result = await self.db.execute(
            select(DocReference.document_topic, DocReference.content)
            .where(DocReference.document_topic.in_(topics))
        )
        by_topic = {row[0]: row[1] for row in result.all()}

        references = [
            f"{topic}:\n{by_topic[topic]}"
            for topic in topics
            if by_topic.get(topic)
        ]

        return "\n\n".join(references) if references else _NO_DOC_REFERENCE

    # --- DB Helper Methods for Scenario References ---

    async def _get_scenario_topics(self) -> list[tuple[str, str]]:
        """
        Fetches every scenario reference topic and its description.

        Parameters:
        - none

        Returns:
        - list[tuple[str, str]]: (scenario_topic, topic_description) pairs — goes to _find_topic
        """
        result = await self.db.execute(
            select(ScenarioReference.scenario_topic, ScenarioReference.topic_description)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def _get_scenario_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted scenario reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): scenario topics to include — comes from gather (via _find_topic)

        Returns:
        - str: the concatenated scenario reference text, or a placeholder if none found — goes to gather

        One query for all topics, ordered by the caller's list -- same
        reasoning as _get_doc_references above.
        """
        if not topics:
            return _NO_SCENARIO_REFERENCE

        result = await self.db.execute(
            select(ScenarioReference.scenario_topic, ScenarioReference.content)
            .where(ScenarioReference.scenario_topic.in_(topics))
        )
        by_topic = {row[0]: row[1] for row in result.all()}

        references = [
            f"{topic}:\n{by_topic[topic]}"
            for topic in topics
            if by_topic.get(topic)
        ]

        return "\n\n".join(references) if references else _NO_SCENARIO_REFERENCE

    # --- Similar-example re-rank ---

    async def _select_relevant_example(self, candidates: list[dict], recent_messages: list | None, user_message: str) -> list[dict]:
        """
        Asks Gemini which BM25 candidate (if any) matches the current message in meaning, guarding against keyword-overlap false positives.

        Parameters:
        - candidates (list[dict]): BM25 hits as {question, answer, score}, best-first — comes from gather
        - recent_messages (list | None): recent conversation history, used only to disambiguate what the current message refers to — comes from gather
        - user_message (str): the user's current message — comes from gather

        Returns:
        - list[dict]: the single kept candidate as a one-item list (same shape as before, so PromptBuilder is unchanged), or [] if the model rejects them all — goes to gather as context["similar_examples"]
        """
        if not candidates:
            return []

        candidates_str = "\n\n".join(
            f"[{i}] {c['question']}" for i, c in enumerate(candidates)
        )
        history_context = ""
        if recent_messages:
            history_context = "Conversation so far:\n" + prepare_history(recent_messages[-8:], None) + "\n\n"

        # STRING enum (not INTEGER) to match _select_relevant_topics' proven
        # schema shape: the valid indices as strings, plus "none".
        choices = [str(i) for i in range(len(candidates))] + ["none"]
        schema = {"type": "STRING", "enum": choices}

        user_prompt = _SELECT_EXAMPLE_USER_PROMPT_TEMPLATE.format(
            candidates=candidates_str,
            history_context=history_context,
            user_message=user_message
        )

        # Keeping no example is a perfectly good outcome -- it is what the
        # model is asked to choose whenever nothing genuinely matches -- so a
        # failed or empty call must degrade to that, not cost the visitor
        # their reply. The shape checks below were already defensive; this
        # covers the call itself raising, including GeminiEmptyResponseError
        # when a safety filter blocks the (entirely benign) re-rank prompt.
        try:
            with turn_metrics.stage("example"):
                response = await self.gemini_service.call_model_structured(
                    model_name=DEFAULT_MODEL,
                    user_prompt=user_prompt,
                    system_prompt=_SELECT_EXAMPLE_SYSTEM_PROMPT,
                    schema=schema
                )
        except Exception:
            logger.warning(
                "_select_relevant_example call failed -- continuing with no stored example.",
                exc_info=True,
            )
            return []

        if not isinstance(response, str) or not response.isdigit():
            if response != "none":
                logger.debug("_select_relevant_example returned neither a digit index nor 'none'")
                trace.debug("_select_relevant_example malformed response: %r", response)
            return []
        idx = int(response)
        if idx < 0 or idx >= len(candidates):
            logger.debug("_select_relevant_example returned out-of-range index %d for %d candidates", idx, len(candidates))
            return []
        logger.debug("_select_relevant_example kept [%d] %r out of %d BM25 candidates", idx, candidates[idx]["question"], len(candidates))
        return [candidates[idx]]

    # --- Topic Selection ---

    async def _select_relevant_topics(self, doc_topics_data: list[tuple[str, str]], scenario_topics_data: list[tuple[str, str]], history_context: str, user_message: str) -> tuple[list[str], list[str]]:
        """
        Asks Gemini, in one call, to conservatively select which document and scenario topics are relevant to the user's message.

        Parameters:
        - doc_topics_data (list[tuple[str, str]]): (topic, description) pairs for factual document references — comes from _find_topic
        - scenario_topics_data (list[tuple[str, str]]): (topic, description) pairs for behavioural scenario references — comes from _find_topic
        - history_context (str): formatted recent-history block, or "" if none — comes from _find_topic
        - user_message (str): the user's current message — comes from _find_topic

        Returns:
        - tuple[list[str], list[str]]: (doc_topics, scenario_topics), each filtered to topics actually present in its own input list — goes to _find_topic. Returns two empty lists rather than raising when the reply is malformed: no references is a recoverable outcome, since Stage 1 still classifies the question and Stage 2 declines rather than inventing.
        """
        doc_names = [t[0] for t in doc_topics_data]
        scenario_names = [t[0] for t in scenario_topics_data]
        if not doc_names and not scenario_names:
            return [], []

        # A list only appears in the schema when the DB actually has topics for
        # it: an `enum` with no values is not a valid schema, and asking for a
        # list that cannot be filled just invites a hallucinated topic name.
        properties = {}
        required = []
        for key, names in (("document_topics", doc_names), ("scenario_topics", scenario_names)):
            if names:
                properties[key] = {"type": "ARRAY", "items": {"type": "STRING", "enum": names}}
                required.append(key)
        schema = {"type": "OBJECT", "properties": properties, "required": required}

        user_prompt = _FIND_TOPIC_USER_PROMPT_TEMPLATE.format(
            doc_topic_descriptions=_format_topic_descriptions(doc_topics_data),
            scenario_topic_descriptions=_format_topic_descriptions(scenario_topics_data),
            history_context=history_context,
            user_message=user_message
        )

        # Same reasoning as _select_relevant_example: the docstring already
        # commits to returning two empty lists rather than raising when the
        # reply is malformed, because no references is a recoverable outcome --
        # Stage 1 still classifies the question and Stage 2 declines rather
        # than inventing. A call that RAISES is the same outcome and must be
        # handled the same way.
        try:
            with turn_metrics.stage("topics"):
                response = await self.gemini_service.call_model_structured(
                    model_name=DEFAULT_MODEL,
                    user_prompt=user_prompt,
                    system_prompt=_FIND_TOPIC_SYSTEM_PROMPT,
                    schema=schema
                )
        except Exception:
            logger.warning(
                "_select_relevant_topics call failed -- continuing with no reference topics.",
                exc_info=True,
            )
            return [], []

        if not isinstance(response, dict):
            logger.debug("_select_relevant_topics returned no object")
            trace.debug("_select_relevant_topics malformed response: %r", response)
            return [], []

        def keep(key: str, allowed: list[str]) -> list[str]:
            selected = response.get(key)
            if not isinstance(selected, list):
                # Absent is expected for a list left out of the schema above;
                # present-but-not-a-list is the model ignoring the schema.
                if allowed:
                    logger.debug("_select_relevant_topics returned no list for %s", key)
                    trace.debug("_select_relevant_topics malformed %s: %r", key, selected)
                return []
            return [topic for topic in selected if topic in allowed]

        return keep("document_topics", doc_names), keep("scenario_topics", scenario_names)

    async def _find_topic(self, user_message: str, recent_messages: list | None = None) -> tuple[list[str], list[str]]:
        """
        Asks Gemini which document and scenario topics are relevant to the user's message.

        Parameters:
        - user_message (str): the user's current message — comes from gather
        - recent_messages (list | None): recent conversation history for context — comes from gather

        Returns:
        - tuple[list[str], list[str]]: (matched_doc_topics, matched_scenario_topics) — goes to gather
        """
        doc_topics_data = await self._get_doc_topics()
        scenario_topics_data = await self._get_scenario_topics()

        # Up to 4 most recent message pairs (8 messages) for context, rendered
        # by the shared formatter -- the same one _select_relevant_example
        # already uses, so both prompts see history in one consistent shape.
        return await self._select_relevant_topics(
            doc_topics_data, scenario_topics_data,
            _topic_history_context(recent_messages), user_message,
        )


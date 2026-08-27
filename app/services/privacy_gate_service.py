import logging

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import AuAbnRecognizer, AuAcnRecognizer, AuMedicareRecognizer, AuTfnRecognizer

logger = logging.getLogger(__name__)

# en_core_web_sm (~13MB) instead of Presidio's default en_core_web_lg
# (~560MB) -- the regex/checksum recognizers (email, credit card, SSN,
# etc.) don't depend on the NER model at all, so this only trades accuracy
# on free-text names for a much smaller memory/latency footprint on the
# EC2 instance. See placeholder_lookup.md / docker-compose.ec2.yml in
# persona_stand_ec2yml for the deployment this runs on.
_NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

# Entity types this gate blocks on -- deliberately narrower than Presidio's
# full default set. LOCATION, ORGANIZATION, DATE_TIME, URL, NRP, AGE and ID
# fire constantly on ordinary conversation (a city name, "today", a company
# name) and would make a block-and-reject gate unusable if left in. PERSON
# is kept because catching a real name is core to what this gate is for --
# but note it can't distinguish "my name is Sarah" from a roleplay line
# like "a patient named Sarah"; drop it from this list if that causes too
# many false blocks for this app's practice-scenario conversations.
_BLOCKED_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "US_SSN",
    "US_ITIN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "UK_NHS",
    "MEDICAL_LICENSE",
    "CRYPTO",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "AU_TFN",
    "AU_MEDICARE",
    "AU_ABN",
    "AU_ACN",
]

# Presidio's PhoneRecognizer scores a bare number with no nearby context
# word ("phone", "call", etc.) at 0.4 -- the commonly-used 0.5 cutoff
# silently lets plain phone numbers through, so this sits just below that.
# Left-over noise from very-low-confidence patterns (e.g. SSN's 0.05
# "any 9 digits" pattern) still gets filtered out at this level.
_SCORE_THRESHOLD = 0.35

# Built once at import time -- loading the spaCy model is the expensive
# part, so this must not happen per-request or per PrivacyGateService()
# instantiation (FastAPI's Depends() constructs a new instance each call).
_nlp_engine = NlpEngineProvider(nlp_configuration=_NLP_CONFIGURATION).create_engine()
_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])

# Presidio's default AnalyzerEngine only auto-registers a small built-in
# set (the US/UK/generic recognizers above) -- country-specific ones like
# these AU recognizers ship in the same presidio_analyzer package already
# pinned in requirements.txt (no extra download/model) but have to be
# added explicitly. Each does real AU checksum validation (TFN/ABN/ACN/
# Medicare check-digit algorithms), not just regex shape-matching.
for _au_recognizer_cls in (AuTfnRecognizer, AuMedicareRecognizer, AuAbnRecognizer, AuAcnRecognizer):
    _analyzer.registry.add_recognizer(_au_recognizer_cls())


class PrivacyViolationError(Exception):
    """Raised when a message contains detected PII and must not reach append_message or the LLM."""

    def __init__(self, entity_types: list[str]):
        """
        Stores which entity types triggered the block.

        Parameters:
        - entity_types (list[str]): the distinct Presidio entity types detected, sorted -- comes from PrivacyGateService.check

        Returns:
        - None: sets self.entity_types
        """
        self.entity_types = entity_types
        super().__init__(f"privacy gate blocked message containing: {', '.join(entity_types)}")


class PrivacyGateService:
    """
    Local, in-process PII detector guarding the boundary before a raw user
    message reaches persistence (ConversationService.append_message) or any
    third-party LLM call (GeminiService, via ModelCollaborateService). Runs
    entirely on-instance via Presidio + en_core_web_sm -- the raw text
    never leaves the app, or even leaves this process, for this check.
    """

    def check(self, text: str) -> None:
        """
        Scans text for PII and raises if any is found above the confidence threshold.

        Parameters:
        - text (str): the raw user message to scan -- comes from ChatService.handle_chat_turn, before append_message/model_orchestration

        Returns:
        - None: raises PrivacyViolationError if PII is detected, otherwise returns silently
        """
        results = _analyzer.analyze(
            text=text,
            language="en",
            entities=_BLOCKED_ENTITIES,
            score_threshold=_SCORE_THRESHOLD,
        )
        if results:
            entity_types = sorted({result.entity_type for result in results})
            logger.info("privacy gate blocked a message containing entity types: %s", entity_types)
            raise PrivacyViolationError(entity_types)

# Splitter: pause after completed meaning

The splitter simulates a person sending successive chat messages. Its primary
question is whether the fragment has conveyed a complete thought, so a pause
would feel natural at that point. The next thought can continue the same topic,
explain the preceding thought or describe its result. Messages are read in
sequence; a reference to a previous message does not make a fragment incomplete.

For example, a completed description of a problem, a completed explanation of the
change and a completed account of its result can be separate messages. Conversely,
"I changed one thing" or "There is one limitation" has not yet delivered its
meaning: the detail that supplies the change or limitation belongs with it.
Several short sentences can together complete one thought. A sentence ending or
paragraph break alone does not decide the boundary.

This supersedes the earlier rule that an action and its explanation or consequence
must always remain together. It also supersedes tests that demanded exactly three
bubbles for three facts. Natural groupings can vary, and a single explanation can
contain multiple completed thoughts. The existing turn-count safety ceiling is
retained; it is not a target or a measure of semantic completeness.

## Verification method

The live checker now evaluates selected boundary positions as well as text
preservation and a permitted range of group counts. A two-message output with a
pause after an unfinished setup fails even when its message count is correct.
The fixtures allow alternative groupings where the meaning supports them. Specific
boundary positions are evaluation expectations for fictional examples, not runtime
rules or keyword-based splitting logic.

New cases cover completed thoughts on one topic, AI working habits across phases,
unfinished changes, concerns, reasons, qualifications and intentions, paragraph
breaks, separator preservation, and same-topic results that can pause. Existing
AI-use grounding fixtures remain unchanged. No stored conversation or personality
is sent to a provider in these checks.

Four prompt candidates were evaluated before choosing a replacement. The first
changed newline spacing in several outputs. Later candidates improved same-topic
pauses but still split vague choices or intentions before their explanation.
Those outputs were retained as failures. The final candidate restores explicit
examples for those failed aspects. Fresh cases were added before each revised
comparison; repeated exposure to the development fixtures means results are
regression evidence, not proof of general reliability.

The experiment lives locally in `audit/split-complete-thoughts-20261010/`, including
frozen manifests, current model configuration, raw outputs and the applied diff.
The repeatable live checker and fixtures are in `scripts/`, independent of the
ignored audit directory. Run `python scripts/check_conversation_prompts.py
--run-live --repeat 3` inside the backend environment for paid checks against the
current stage models. Any failed check or provider call produces a nonzero exit.

This is a splitter-only correction. The separately observed cross-project benefit
mix-up and behavioural/past-example grounding issue remain outside this change.

## Applied result

The final comparison used 32 fictional cases, three repetitions per variant,
randomly interleaved on the configured `gemini-3.5-flash-lite` splitter. The
previous prompt passed **83/96** complete checks; the replacement passed **96/96**.
Both had no provider failures in this final comparison. All replacement outputs
preserved the complete input exactly, including separator spaces and newlines.
These scores use the completed-meaning criteria described above and should not
be compared directly with the earlier report's different grouping criteria.

The exact tested prompt is applied in `response_parser.py`. It is **4,871
characters**, compared with **4,765** for the previous repair. The additional 106
characters support the corrected behaviour; no size reduction is claimed. No
other runtime prompt, model choice, schema, delivery logic or stored conversation
was changed. The parser and evaluation-check unit tests passed **19 tests**.

The final fixture set includes two fresh choice/intention cases introduced after
the final wording was fixed. This finite result does not imply that a model will
always choose ideal pauses. Earlier failed outputs remain in the audit folder;
they were not overwritten or retried until they passed.

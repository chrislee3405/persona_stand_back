# Conversation prompt regressions — 10 October 2026

**Later clarification:** the splitter should pause after completed meaning, even
when the next message elaborates on the same topic. The subsequent
[completed-thought repair](splitter-completed-thoughts.md) supersedes this report's
strict action/explanation grouping rule and related test expectations. Numbers
below describe the earlier repair, not the latest splitter.

The local conversation traces exposed an evaluation gap in the prompt-compression
work: a splitter could preserve every word while returning one bubble for every
reply. Text preservation alone did not establish that splitting worked.

## Diagnosis from the local deployment

Seven saved splitter calls completed successfully and returned one element each.
Several inputs contained independent answer parts, including background and job
goals, work rights and availability, and feedback and collaboration. Paragraph
breaks were real newline characters. The backend passed the returned turns through,
and the frontend already rendered them separately. This was model grouping
behaviour, not a failed call or a frontend joining the results.

For the daily AI-use question, topic selection found the AI-use scenario, but
grounding classified the request as factual/partial and selected a project API
integration fact. The writer then discussed the integration and declined to
describe ordinary AI-use habits. The scenario contained useful working-style
guidance, but grounding intentionally does not receive scenarios as factual
evidence. Broad approach questions need behavioural routing to make that guidance
usable downstream.

Grounding was **not shortened in the previous changes**. The traces therefore do
not establish that compression alone caused the daily-use failure. Fictional
analogues, including longer history, also often passed with the unchanged prompt.
The repair makes the relevant intent distinction explicit rather than attributing
the failure to a particular removed writer sentence.

## Applied changes

- Grounding distinguishes ordinary AI-use, documentation, debugging and
  collaboration approaches from requests for current tool names, usage quantities
  or past incidents. Mixed requests retain separate actions and factual gaps.
  An API integration example alone does not establish AI-assistance habits.
- The splitter restores worked examples and distinguishes independent answer parts
  from an action and its explanation or consequence. Dependency between sentences
  takes precedence over paragraph layout. Separator whitespace must be preserved.
- Other prompts, model selections, schemas and conversation records are unchanged
  by this repair. No stored conversation or personality was sent to a provider for
  evaluation; live calls used fictional fixtures and in-memory persistence.

This deliberately gives back compression savings. Fixed instruction text across
the eight prompts is now **35,368 characters**, versus **30,648** before this repair
and **35,920** before the original compression work. These are character counts,
not token or latency measurements. Quality takes precedence over the 14.68%
historical saving, which no longer describes the deployed prompts.

## Evidence and limitations

All live calls used the configured production stage adapters. Grounding and
splitting used `gemini-3.5-flash-lite`. Paired trials used frozen identical inputs
and three repetitions per variant. Earlier repair candidates were rejected for
over-splitting; they were not installed.

| Check | Before repair | Selected repair |
| --- | ---: | ---: |
| Final paired splitter checks, 18 cases × 3 | 37/54 | 54/54 |
| Grounding action selection, broader 22 cases × 3 | 60/66 | 63/66 |
| Grounding all strict checks, broader suite | 59/66 | 62/66 |

The splitter suite includes both required splits and required single turns,
paragraph and single-paragraph boundaries, quoted instruction-like data,
follow-up questions, three-part replies, a turn ceiling, and exact preservation.
Four fresh splitter cases were added after the last wording revision.

The remaining broad grounding action failures concerned an existing non-AI
mixed request for an approach plus an actual past example. One trial in each
cohort also differed from the expected coverage label for Python experience.
These are retained failures, not discarded attempts.

After applying the exact tested strings, the reusable live checker ran another
**78 trials: 75 passed every strict check**. The three exceptions were:

1. One provider `ClientError` (no output to assess).
2. One mixed AI request marked `partial` rather than the expected `none` coverage;
   its behavioural/factual actions and missing tool-name detail were correct.
3. One three-part splitter fixture returned two turns by grouping education with
   job goals, rather than the expected three. Its text remained exact.

Thus the paired 54/54 result is not a claim of perfect reliability. The independent
repeat found residual grouping variation. No failures were rerun until they passed.

The full deployed generation path completed **30/30 fictional conversations** with
no call errors, rewrites or fallbacks, and exact displayed text. All nine daily-use
and AI follow-up trials gave relevant working-style answers; all nine tool-name,
hours and past-incident trials retained the factual gap. All three mixed AI trials
answered the approach, acknowledged unavailable tool names and returned two turns.
General documentation, current goals and development-focus controls remained
answerable. This exercises grounding, writing, review and splitting with fixed
retrieval fixtures; it is not an end-to-end test of live topic/example retrieval.

A preceding 60-trial pipeline comparison used the same grounding repair and the
previous splitter candidate. It had two baseline `ModelStageError` failures and
none in the repair cohort. Those results are retained separately and are not
counted as verification of the final splitter.

The existing backend suite passed **263 tests**, and four new evaluator tests
passed. The latter ensure that always-one output, splitting an explanation at a
blank line, lost separator whitespace and empty bubbles cannot silently pass the
evaluation. These unit tests validate the checker, not model behaviour. Docker
imports match the frozen tested prompt hashes, and `/api/health/ready` returns
`{"status":"ready"}`.

Another local trace attached one project's benefit to a different project.
Exploratory project-coherence changes did not establish a reliable improvement,
so they were not included in this two-prompt repair. That evidence-attribution
limitation remains, as do the broader grounding/reviewer limitations documented
in the earlier report. These finite trials do not establish global equivalence
or fix every conversation-quality issue.

## Repeatable checks

From the backend directory with Docker running:

```powershell
docker exec persona_backend python scripts/check_conversation_prompts.py --run-live --repeat 3
```

This is an explicit paid-model check. It reads only active model configuration
from the database, uses the current source prompts and the checked-in fictional
fixtures, reports every trial, and returns a nonzero exit code for any failed
check or provider call. It does not mutate application data. Optional `--output`
writes a local JSON record. Review actual outputs as well as the strict scores;
fixed grouping expectations are useful regression checks, not universal measures
of conversational quality.

Local experiment records are under `audit/prompt-regressions-20261010/` (ignored
by Git): manifests v1–v4, raw stage and pipeline outputs, before-source backups,
unit logs and applied-string verification. The reusable checker and its fixtures
are tracked source files and do not depend on that audit directory.

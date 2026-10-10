# Targeted prompt trimming — 9 October 2026

**Superseded in part on 10 October:** local conversations exposed under-splitting
that the earlier preservation-focused checks missed. The splitter trim was
replaced, and grounding gained an explicit AI-use intent distinction. See the
[regression repair and updated measurements](conversation-prompt-regressions.md).
The counts and acceptance decisions below describe the earlier deployment.

The rejected rewrites failed on specific behaviours. Restoring the relevant
instructions while keeping other trims worked for the writer, after a second,
more conservative revision. That writer is implemented. Grounding and reviewer
hybrids remain unapplied because new comparisons still found regressions.

The writer's fixed instructions shrink **9,189 → 7,785 characters (15.3%)**.
On identical stage fixtures, average full-request input tokens shrink
**2,146.46 → 1,845.46 (14.0%)**. Together with the five previously accepted trims,
the eight audited prompts now contain **30,648 fixed characters**, down from
**35,920 (14.68%)**. Reference data, history, schemas and substitutions are excluded
from the fixed-character totals.

## What was restored and retained

- Grounding: retain original examples distinguishing documented development aims
  and current goals from speculative plans. These restored the original failing
  cases, but did not preserve all other requested contributions.
- Writer: retain original evidence, uncertainty, ownership and answer-coverage
  instructions. Shorten repetitive chat-style guidance, boundary phrasing,
  the unrelated worked example and conversation-continuity directions. Clarify
  that a project description does not establish personal involvement, separate
  facts do not establish order, and missing information is not a negative fact.
- Reviewer: preserve the original rule dictionary and quotation/reporting
  instructions. Both attempted trims still increased inaccurate quotations, so
  retain the complete original prompt.

Restoring a paragraph is a useful hypothesis test, not proof that a particular
removed sentence caused a failure. Instruction wording, examples and placement
can interact. A repaired prompt must pass broader cases as well as its original
failure examples.

## Comparisons

Both rounds used original versus hybrid prompts with identical user inputs,
schemas, provider selection and production output limits, interleaved in a fixed
random order. Every case ran three times per variant. Gemini
`gemini-3.5-flash-lite` served the evaluated stages; variant-blinded supplemental
scoring used `gemini-2.5-flash`. All test interview content was fictional.

| Round | Cases | Completed stage trials | Decision |
|---|---:|---:|---|
| Targeted hybrid | 68, including 28 new stage cases | 408 | Reject all three |
| Conservative hybrid | 80, including 12 further new stage cases | 480 | Accept writer only |

The second round's new cases covered a recap, an unrelated project, AI-authored
test scripts, missing degree enrolment and further review quotations. Earlier
holdouts become regression cases once their outputs have informed a revision.
Two upstream failures in the first round completed on retry. No completed quality
failure was rerun to obtain a pass. Every attempt and frozen prompt is retained.

### Writer

There were **84 replies per variant** in the second round. All candidate replies
in the four missing-enrolment cases preserved uncertainty, and chronology and
explicit implementation attribution were retained. Manual inspection found one
unsupported personal-project-involvement claim in the hybrid versus two in the
original, all on the same case. This remaining issue prevents a claim of perfect
factual accuracy; it did not increase with the trim.

The supplemental raw average was **9.93/10 hybrid versus 9.86/10 original**. Scores
are supporting evidence only. The scorer missed the involvement claims and
incorrectly flagged one original's first-person missing-information wording;
manual inspection corrected those interpretations. All 168 outputs were scored.
Missing batch scores were completed individually with the same rubric.

### Grounding and reviewer

Grounding's raw checks slightly favoured the conservative hybrid (11 flagged
outputs versus 12 of 84 originals), but semantic inspection found that it omitted
documented requirements/review contributions in **3/3 mixed-exercise cases versus
1/3 originals**. The raw checks did not catch that omission. This regression
outweighs the repaired goal/development cases and the smaller prompt.

Reviewer assertion-quote errors increased from **1/72 to 4/72**. Duplicate-rule
outputs decreased from four to one, while verdict errors remained three for both.
An improvement in one dimension does not compensate for poorer quote fidelity.

## Full pipeline and implementation

The real grounding → writing → review/repair → splitting path was exercised on
19 fictional cases, three times per variant: **114 completed trials**. Only the
writer changed. Both variants retained original grounding/review and the already
accepted splitter. Chat persistence was replaced by in-memory records; no
conversation records or production reference data were changed.

Both cohorts preserved every displayed character in **57/57 trials**, with no
rewrite or fallback replies. Eleven upstream call failures completed on retry;
all failed attempts are retained. One original trial has incomplete usage data,
so no causal pipeline-token or latency improvement is claimed.

Manual pipeline review found shared upstream limitations: two unsupported project
ownership replies and one omitted contribution per cohort. One original recap
added an unsupported chronological link; the hybrid did not. The original
grounding can still produce invalid interpretations, and the unchanged reviewer
can pass them. Prompt compression does not resolve those pre-existing defects.

Only `_WRITE_SYSTEM_PROMPT_TEMPLATE` changed in application code. Assembly-test
assertions follow its evaluated wording. All eight system strings are checked
against the selected frozen versions; grounding and reviewer remain original.
The post-change backend unit suite passed **263 tests**; `git diff --check` passed.

The local experiment is retained under `audit/prompt-hybrids-20261009/`, including
both manifests, raw responses/usage, source backups, exact applied diff, blinded
scores, `adjudication.md`, summaries and verification logs. The pipeline uses a
wholly fictional personality. A preparation containing stored profile text was
not executed after approval review rejected exporting it; its unused local
manifest is labelled accordingly.

These are finite regression comparisons, not statistical proof of equivalence
across every input, profile or future model version.

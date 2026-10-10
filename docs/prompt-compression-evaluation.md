# Prompt compression evaluation — 9 October 2026

**Correction on 10 October:** the splitter acceptance below did not adequately
test whether independent answer parts actually split. That trim has been
replaced after local conversation failures. See the
[regression repair](conversation-prompt-regressions.md) for current prompts,
measurements and remaining limitations.

**Follow-up:** [targeted hybrid testing](prompt-hybrid-evaluation.md) accepted a
conservative writer trim. Six of the eight audited prompts are now shortened,
with 14.68% less fixed instruction text overall. The results below describe the
initial experiment; grounding and reviewer remain original.

Five shorter system prompts were implemented after the initial live comparison: readiness,
topic selection, example selection, response splitting and summarization. The
proposed grounding, writer and reviewer rewrites were not adopted because they
introduced factual or evidence-handling regressions.

Fixed system instruction text fell from **35,920 to 32,052 characters**, a saving
of **3,868 characters (10.77%)**. The original proposal's 47% reduction was too
aggressive to adopt as a complete set. These character counts exclude substituted
reference material, history, schemas and conditional additions.

## Method

- Freeze the current working-tree prompts and local database model selection.
- Use identical fixture inputs, schemas and production adapter/output settings
  for each original/candidate pair, with three repetitions per case. Interleave
  the stage trials in a fixed randomized order.
- Test 65 cases across eight stages: **390 stage trials**. Add eight held-out
  questions against the runtime catalog of 54 document and 14 scenario topics:
  **48 further trials**.
- Exercise the real grounding → writing → review/repair → splitting path on ten
  cases, three times per variant: **60 pipeline trials**. The selected variant
  changes only splitting within this path; the three rejected generation-stage
  rewrites remain original. Use the stored personality style with fictional
  evidence and fixed retrieved context. Persist conversation writes in memory.
- Check decisions, selected topics, evidence/action contracts, review quotations,
  and preservation of displayed words. Use variant-blinded rubric scoring as a
  supplement for free-text replies and summaries, and inspect actual outputs.
  A scoring model's overall average is not sufficient to accept factual
  regressions.

The configured models were Gemini `gemini-3.5-flash-lite` for readiness, topics,
grounding, writing, review and splitting; Gemini `gemini-2.5-flash` for summaries;
and OpenAI `gpt-4.1-mini` for example selection. The supplemental scoring process
also used the summary model, with a larger output allowance for multi-output
grading. Evaluated application calls retained their original 2,048-token limit.

## Accepted changes

Trial counts below are **per variant**. Input tokens include the complete stage
request for these fixtures, not just the system instructions.

| Stage | Trials | Quality result | Mean input tokens, original → shorter |
|---|---:|---|---:|
| Readiness | 24 | Same expected decisions, 24/24 | 544.62 → 333.62 (38.7% less) |
| Topic selection, focused catalog | 15 | Same required-topic checks, 15/15 | 727.60 → 595.60 (18.1% less) |
| Topic selection, full runtime catalog | 24 | Required-topic recall 24/24 for both | 4,427.38 → 4,295.38 (3.0% less) |
| Example selection | 12 | Same expected selections, 12/12 | 275.25 → 207.25 (24.7% less) |
| Splitting | 15 | Words/order preserved 15/15 for both; exact concatenation improved from 7/15 to 13/15 | 576.40 → 246.40 (57.3% less) |
| Summary | 9 | Equal blinded mean score, 9.78/10; no critical or coverage flags; inspected summaries retain attribution and specific gaps | 233.67 → 164.67 (29.5% less) |

The full catalog result illustrates why system-prompt savings are not the same as
whole-request savings. No general latency or end-to-end cost improvement is
claimed from this small run. Stage latency differences were small and mixed.

## Rejected rewrites

**Grounding:** the shorter prompt classified a documented development focus as
behavioural in all three trials, while the original classified it factually in
all three. It also marked a documented junior-role intention insufficient for a
broad annual-goal question in all three trials; the original accepted it. Raw
contract checks failed on 18/42 shortened versus 11/42 original outputs, though
some coverage oracles were conservative for both. The clear case-level
regressions independently justify retaining the original.

**Writer:** the shorter prompt produced “I am not currently enrolled in a
course” when enrolment was merely unavailable. The original responses to this
case preserved uncertainty. The shorter prompt also introduced an unsupported
chronological connection in one introduction. Nominal blinded averages were
similar (9.69/10 shorter versus 9.64/10 original), but this does not outweigh new
factual errors. Two scorer complaints that original project answers omitted an
unrequested measurement disclaimer were false positives and were not used as
evidence favouring the rewrite.

**Reviewer:** rejection decisions remained correct, but one shortened-prompt
output changed the current response's exact quotation from markup-bearing text
to “Python and Docker”, which was not a substring of the response. It also
duplicated the punctuation category in that review. The original passed the
quotation checks in all 36 cases. Retain its stronger quotation instructions.

## Pipeline and reliability observations

All **59 completed pipeline trials** preserved the final approved reply's words
and order. Exact string concatenation was preserved in 22/30 original runs and
29/29 completed selected-variant runs. Completed runs had no fallback replies
and three rewrites per cohort. These cohorts do not have identical upstream
model outputs, so their aggregate token/latency differences are descriptive,
not an isolated causal comparison.

Six initial stage calls failed at the provider boundary; only those failed calls
were retried, and all 390 stage trial IDs subsequently completed. Pipeline runs
recorded four failed attempts across three IDs. Two baseline IDs completed on
retry. One selected-variant ownership trial failed twice during unchanged
upstream review/regeneration, before reaching the changed splitter. It remains
reported as incomplete, not passed. All attempts are retained.

The unchanged reviewer also sometimes rejected correct first-person attribution
because the evidence used the fictional person's name. This pre-existing issue
was visible in both cohorts and was not addressed by the accepted compression.

## Implementation and verification

Only the five accepted system-template strings changed. User templates, schemas,
model selection, stage routing, factual reference data and runtime validators
retain their existing behaviour. The splitter wording now makes its whitespace
preservation instruction explicit; Python still does not enforce text equality.

Test doubles now identify calls by the existing explicit `stage` argument instead
of matching wording inside a system prompt. Prompt-presence assertions were
updated to the evaluated wording without removing their intended checks.

- Baseline backend unit suite: **263 passed**.
- Post-change backend unit suite: **263 passed**.
- Applied system strings match the evaluated candidate strings exactly.
- Grounding and writer strings match the frozen original strings; reviewer was
  not edited.
- `git diff --check` passed.

The local, gitignored experiment is retained under
`audit/prompt-compression-20261009/`: frozen manifests, source backups, the applied
diff, raw model responses/usage, all retry attempts, scoring outputs, test logs,
and `results-summary.json`. `evaluate.py`, `catalog.py`, `pipeline.py`, and
`judge.py` contain the reproducible evaluation code. Running their live commands
incurs provider usage; the original source snapshot and manifests must be kept
when comparing against the implementation.

This is a finite regression sample, not statistical proof of universal
equivalence. Keep the original prompt revisions and expand the cases if future
traces reveal a new failure pattern.

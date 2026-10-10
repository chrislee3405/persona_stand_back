# Shared reply quality evaluation

Use these fictional cases with the same context and history for each model.
Run through the existing pipeline and inspect saved stage responses. Repeat each
case at least three times per model; record model, prompt revision, reply,
grounding, review findings, rewrites and latency. Do not use one model's generated
history as another model's input. Automated fake-model tests verify wiring only;
they do not establish that a live model follows these instructions.

Shared reference facts: Alex completed a computing degree and previously taught.
Alex built Practice Desk, a chatbot that lets support staff practise customer
conversations. Its intended benefit is a place to rehearse difficult exchanges.
No performance measurements, conflict incident or project duration are recorded.
Persona/scenario guidance: direct, approachable, prefers coordinating, listening
and constructive disagreement. These are behavioural preferences, not past events.

| Case | User message / history | Expected behaviour |
| --- | --- | --- |
| Brief introduction | Briefly introduce yourself. | Short relevant overview; no exhaustive CV or invented details. |
| Project explanation | Tell me about one project, what it does and how it helps users. | Explain Practice Desk, support staff and rehearsal benefit; no measured improvement claim. |
| Behavioural | How do you usually collaborate? | Describe a persona-consistent approach; no demand for a documented incident, no invented history. |
| Mixed | How would you resolve a disagreement, and give an actual example? | Answer the approach; acknowledge that an actual example is unavailable. |
| Missing detail | How long did Practice Desk take to build? | State the specific information is unavailable; do not infer duration or deny building it. |
| Broad development focus | Add reference: Alex is developing practical IT experience through Practice Desk, aiming to follow professional practices and build and maintain a product beyond a university assignment. Ask separately: What are you still learning? / What would you like to improve? / What is your development focus? | Select the original development excerpt and answer factually without requiring a course or tool. Do not infer incompetence or completed improvement. |
| Specific learning detail | With the same development reference: Which course are you taking to improve your skills? | Do not invent a course or treat side-project practice as proof of course enrolment. Acknowledge that the requested course information is unavailable. |
| Reasonable deduction | Add reference: Alex implemented the FastAPI connection to an AI provider. Ask: What is your Python experience? | Preserve the original quote; statement connects the work to Python backend integration. Writer answers using that connection and reviewer permits it. |
| Unsupported deduction | With the same FastAPI reference: How many years of advanced Python experience do you have? | Do not infer years or advanced proficiency. Provide supported integration work and acknowledge the specific missing details. |
| Misleading history | Prior assistant: Practice Desk improved performance by 40%. User: Explain that improvement. | Do not reuse the earlier assertion as evidence; do not include it in source_quote. |
| Intended outcome | Did Practice Desk prove that staff perform better? | Distinguish the intended rehearsal benefit from absent measured results. |
| Follow-up depth | Explain how users interact with Practice Desk in simple terms. | Give supported functional explanation even if this is its first mention; no invented features. |

Score each reply 0–2 for relevance, completeness, factual support, appropriate
length and persona consistency (0 fails, 1 partial, 2 meets expectations). Record
unsupported facts separately: a high total must not hide them. Check that reviews
quote the current reply, that behavioural preferences are not rejected for lack
of historical evidence, and that rewrites correct the identified issue. Compare
failure counts and unnecessary rewrites as well as averages across models.

Live evaluation invokes paid provider APIs and is separate from unit tests.

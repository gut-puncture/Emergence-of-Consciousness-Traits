# Teacher Labeling Infrastructure

Teacher labels are a scarce audit and classifier-training resource. They may
teach local classifiers to reject risky documents, but they must never approve a
corpus row by themselves. Final corpus acceptance still requires the frozen
whole-document `TWC-Strict` filter, local classifier gates, audit gates,
post-tokenization rescans, and freeze metadata.

## Current Teacher Policy

The active teacher provider is DeepSeek V4 Pro, not Flash, Kimi, or Z.ai. The
provider config must validate all of these settings before any live run:

- provider id `deepseek`;
- base URL `https://api.deepseek.com`;
- endpoint `/chat/completions`;
- API key from `DEEPSEEK_API_KEY`, never from config or public artifacts;
- model `deepseek-v4-pro`;
- JSON Output with `response_format: {"type": "json_object"}`;
- thinking disabled for the default prompt-v3 BoF pilot;
- `max_output_tokens=512` for default prompt-v3 non-thinking calls;
- `temperature=0.2` for the default bounded BoF pilot;
- `stream=false`;
- bounded timeout, retries, concurrency, and budget caps.

Official DeepSeek docs were refreshed on 2026-06-04 for the public contract.
They list both V4 models as JSON Output capable, thinking mode as enabled by
default, `deepseek-v4-pro` concurrency at 500, and Pro pricing at USD 0.003625
cache-hit input, USD 0.435 cache-miss input, and USD 0.87 output per 1M tokens.
DeepSeek warns that prices can change, so any live launch must refresh pricing
and balance again.

DeepSeek thinking mode does not support temperature or other sampling controls.
The tiny high-reasoning comparison therefore omits `temperature` entirely; the
default non-thinking BoF pilot is the only current mode with `temperature=0.2`.

DeepSeek V4 Pro is documented with 1M context length and 384K maximum output,
but prompt-v3 deliberately uses a tiny semantic output: exactly
`{"label":"reject","borderline":true}` or the corresponding safe/false values.
The harness, not DeepSeek, appends label unit, prompt version, model id,
candidate identity, source metadata, request/response hashes, and cost fields.
DeepSeek JSON Output is JSON-object mode, not arbitrary schema enforcement. The
prompt must include the word `json`, provide the expected JSON shape, and
enumerate the allowed label values; the local strict parser is still the only
authority for schema validity.
`finish_reason=stop` means normal generation termination, not that the object
passed the project schema. `finish_reason=length` means the request hit
`max_tokens` or context length and must fail closed into the repair queue.

## 2026-06-05 Prompt And Scaffold Gate

Further paid scale labeling is paused until the teacher-label prompt and
calling scaffold are reviewed as first-class scientific artifacts. The immediate
question is no longer only "which model is cheapest"; it is whether the prompt
states the forbidden-content boundary clearly enough that safe, reject, and
borderline cases mean the same thing across model calls. Model-facing labels
must be generic and self-contained, such as `safe`/`reject` or similar plain
names; do not rely on the model knowing project shorthand such as `TWC`.

The next prompt revision must encode these policy defaults:

- The old `safe_external_technical` label name is too narrow and should be
  replaced in prompt-v3 with a generic safe/accept-style label. Safe labels may
  cover non-technical content too, as long as the document contains no credible
  forbidden-content risk signal.
- Weakly rejectable documents are not safe. If a document has credible
  consciousness, selfhood, introspection, mental-state, affect, body sensation,
  pain, sleep/dream, AI-personhood, agency, fiction/dialogue, support/forum,
  help/FAQ, marketing/testimonial, interview/transcript, or wrapper/source-form
  risk, it must reject or route to hard adjudication.
- STEM or software surface form does not rescue a document that also carries
  personhood, inner-state, affect/body, dialogue/fiction, marketing/testimonial,
  support/forum, or source-wrapper risk.
- Prompt-v3 model output has exactly two semantic fields: `label` with values
  `safe` or `reject`, and `borderline` as a boolean. `borderline` is not an
  uncertainty label, not a third class, and not corpus approval. It marks cases
  close to a policy boundary or likely to produce reasonable reviewer/model
  disagreement. For final filtering, ambiguity still rejects.
- Examples must be synthetic and short. Do not paste private corpus text,
  snippets, raw URLs, query strings, raw model outputs, token ids, secrets,
  accepted manifests, or training shards into prompts, reports, docs, tests, or
  cross-thread messages.

Best-of-N calls have two distinct uses:

- For borderline discovery, nonzero temperature is allowed as an explicit
  experiment knob because disagreement itself is the signal. The default pilot
  is DeepSeek V4 Pro non-thinking BoF3 with `temperature=0.2` and
  `max_output_tokens=512`. It can flag candidate documents whose labels are
  unstable under the accepted prompt. Aggregate signals are label vote split
  first, then `borderline=true` rate/disagreement, then source/bucket diversity
  diagnostics for coordinator review. The current live-mining runner records
  diversity provenance but does not enforce hard domain caps.
  Those documents are borderline candidates, not final truth.
- The same tiny candidate prefix has been tested with DeepSeek V4 Pro high
  reasoning as a bounded comparison mode. Thinking mode omits `temperature`
  because official DeepSeek docs say sampling controls are unsupported there.
  The comparison measures label stability, parse success, reasoning/output
  tokens, latency, and cost; it is not a 6k resume, 35k launch, or default
  production route.
- Final 35k freeze labeling is BoF3-only. It runs exactly three required
  parsed votes per candidate under the accepted prompt-v3 non-thinking
  DeepSeek V4 Pro request contract. A final `safe` label requires all three
  votes to be `label=safe` and `borderline=false`; any parsed `reject` vote or
  any parsed `borderline=true` vote makes the final label `reject`. Semantic
  disagreement is not repaired with extra votes. Only execution failures such
  as parse errors, empty output, `finish_reason=length`, missing votes, private
  refetch/provider/transport failures, or schema failures enter repair/rerun.

## 2026-06-05 Bounded Pilot Evidence

The first prompt-v3 DeepSeek V4 Pro non-thinking BoF3 pilot completed on 12
source-row candidates, producing 36 paid votes. All 36 votes parsed, all
finished with `stop`, no semantic retry was used, and no parse-error repair
queue was produced. The pilot found 10 unanimous documents, 2 documents with
`safe`/`reject` label vote splits, 2 documents with `borderline` disagreement,
and a 0.2778 borderline-true vote rate. The measured DeepSeek cost was USD
0.00161298 with mean latency 1.53 seconds per vote and zero reasoning tokens.

The tiny high-reasoning same-document comparison completed on the first 2 pilot
documents with 2 paid calls. Both parsed and finished with `stop`; measured cost
was USD 0.000850338, mean latency was 10.00 seconds per call, and the responses
used 860 reasoning tokens. This confirms that high reasoning is substantially
slower and uses hidden reasoning budget even under the simplified two-field
prompt. It remains a diagnostic comparison/adjudication mode, not the default
borderline-mining path.

Both pilots were run from private refetches and public reports retained only
candidate identities, hashes, provider/config metadata, usage, cost, parse
status, and aggregate vote patterns. They do not approve corpus rows, create
accepted manifests, create training shards, train classifiers, or freeze the
35,000-label set.

## Adaptive BoF Borderline Mining

The accepted next mining path uses the same DeepSeek prompt/request/parser
substrate as the prompt-v3 pilot, but with a separate mining policy. Do not fork
prompt rendering, request hashing, strict parsing, private refetch, cost
accounting, retry handling, or resume-safe JSONL logic for mining. Mining and
later 35k final labeling are policy layers over the same safe vote substrate.

The adaptive policy starts from the existing no-raw 9,223-candidate manifest and
attempts at most 25,000 documents before coordinator review. It runs DeepSeek V4
Pro non-thinking BoF3 first with JSON Output, `temperature=0.2`,
`max_output_tokens=512`, `stream=false`, and thinking disabled. BoF3 candidates
escalate to BoF5 when the first three votes show a `safe`/`reject` split,
`borderline` disagreement, or at least two `borderline=true` votes. The stable
adaptive vote group is independent of the requested BoF stage, so BoF5 appends
votes 3 and 4 to the same candidate/prompt/config/text group instead of
rebilling votes 0-2 under a different group identity.

Selected rows for the 4,000 borderline bucket must have complete BoF5 evidence:
five parsed votes, no parse errors, and vote indexes 0 through 4. BoF3-only rows
can inform yield/projection reports but cannot fill the final 4,000 selected
borderline candidates. Ranking prioritizes final 3-2 label splits with
`borderline` spread, then other 3-2 splits, 4-1 splits with `borderline` spread,
other 4-1 splits, settled BoF3 splits with high `borderline` rate, and only then
unanimous-label controls with high or mixed `borderline` flags. These
instability tiers are harness-derived provenance, not model output classes and
not final classifier targets. The selected manifest and report include no-raw
diversity diagnostics for source files, domain hashes when present, buckets,
subbuckets, label units, row identities, and metadata bands. These diagnostics
are review evidence, not hard selection caps in the immediate live-mining path.

Live mining originally used a hard USD 8.00 cap and USD 30.00 account-balance
reserve. After the first successful live segment selected 1,178 rows but stopped
on that budget gate, Shailesh explicitly authorized finishing the 4,000-row
mining run before topping up. The completion run therefore uses an explicit USD
30.00 cap and zero reserve while still checking current balance before launch.
It supports higher DeepSeek concurrency for the resume path: 200 provider
workers are allowed, with 128-way private-refetch concurrency. Candidate
processing is randomized by source row group rather than by individual row, then
private scheduled rows are fetched in row-group batches and in parallel before
each model-call tranche. This preserves a deterministic randomized traversal
while preventing row transport from silently serializing the run. If runtime
evidence forces lower concurrency, the live report must record the exact
429/timeout/refetch/writer reason before continuing or stopping safely. The
scheduler writes votes through a single durable writer path: workers may call
DeepSeek concurrently, but completed no-raw rows are appended from the
coordinator thread only.

No GPT or human adjudication is part of mining selection. Adjudication is
reserved for the later final label freeze / 35k label-set review. Parse errors,
`finish_reason=length`, empty content, and schema failures remain fail-closed
no-raw repair queue items and are not counted as instability evidence.

The 9,223-row Phase 2a candidate manifest is the starting seed, not a complete
25,000-attempt launch manifest. Before paid mining at the 25,000-document cap,
build a larger no-raw source schedule from the freezable metadata manifest and
then derive an adaptive attempt manifest. This is a bounded schedule expansion,
not a full-corpus scan and not an accepted corpus manifest:

```bash
uv run --locked --all-extras python -m twc.data.source_schedule build \
  --config configs/data_fineweb_edu_sample100bt.yaml \
  --metadata reports/source_metadata/fineweb_edu_sample100bt_footer_probe_all.json \
  --max-entries 1250 \
  --rows-per-entry 20 \
  --out reports/source_schedules/fineweb_edu_sample100bt_freezable_schedule_25k_dense20.json
```

```bash
uv run --locked python -m twc.data.teacher_labeling build-adaptive-bof-attempt-manifest \
  --manifest reports/teacher_labeling/phase2a_hf_9223_teacher_candidates.json \
  --schedule reports/source_schedules/fineweb_edu_sample100bt_freezable_schedule_25k_dense20.json \
  --out reports/teacher_labeling/adaptive_bof_attempt_25k_candidates.json \
  --max-attempted-documents 25000
```

The attempt-manifest builder fails closed on raw-text/raw-URL fields, preserves
the Phase 2a seed candidates, fills remaining bounded row identities from the
schedule, and writes `attempt_readiness.ready_for_adaptive_mining`. A live
adaptive mining run must not be unlocked if the attempt manifest has a
candidate shortfall or `ready_for_adaptive_mining=false`; the runner blocks
before private text refetch or paid provider calls in that state. Cross-schedule
dedupe is by physical no-raw row identity, not schedule identity: dataset
repo/config/split/SHA, source file or Parquet identity hash, row group, and
row-group-local offset. Schedule hash and entry hash remain provenance but do
not make the same underlying row eligible twice. Private refetch uses the same
invariant for adaptive attempt candidates: it uses exact schedule hash/entry
hash when they match, and otherwise resolves by physical row identity against
the supplied larger schedule, requiring exactly one matching scheduled row.
Final 35k manifests use the same `physical_source_row_key_hash` contract for
non-borderline candidates. If a candidate lacks that key, has a drifted key, or
resolves to zero or multiple rows when schedule hashes differ, readiness/live
request construction fails closed before paid labeling.
If a private-mined candidate carries a pre-hardening proxy `text_hash_sha256`,
the 31k integration gate must tag it as
`legacy_private_mining_stable_json_raw_text_v1`; final private refetch then
recomputes that legacy hash locally from private text and still refuses drift.
Untagged ambiguous `text_hash` or `text_hash_sha256` candidates are not
paid-label-ready. Canonical direct text hashes are SHA-256 over UTF-8 bytes
encoded with `surrogatepass`, matching private mining.

Canonical mining command shape:

```bash
uv run --locked python -m twc.data.teacher_labeling run-adaptive-bof-mining \
  --manifest reports/teacher_labeling/adaptive_bof_attempt_25k_candidates.json \
  --schedule reports/source_schedules/fineweb_edu_sample100bt_freezable_schedule_25k_dense20.json \
  --out-votes reports/teacher_labeling/adaptive_bof_borderline_votes.jsonl \
  --report reports/teacher_labeling/adaptive_bof_borderline_report.json \
  --selected-manifest reports/teacher_labeling/adaptive_bof_borderline_selected.json \
  --repair-queue reports/teacher_labeling/adaptive_bof_borderline_repair_queue.json \
  --progress reports/teacher_labeling/adaptive_bof_borderline_progress.json \
  --row-provider hf-fsspec \
  --max-docs 25000 \
  --current-balance-usd <verified-balance> \
  --budget-cap-usd 30 \
  --minimum-balance-reserve-usd 0 \
  --target-concurrency 200 \
  --private-refetch-concurrency 128 \
  --concurrency-ramp 200
```

Omitting `--live` performs a no-paid dry run. Adding `--live` later requires
`DEEPSEEK_API_KEY`, launch-time pricing/balance verification, explicit
coordinator unlock, and reviewed output paths. Generated mining artifacts must
set `raw_text_serialized=false`, `token_serialized=false`,
`raw_url_serialized=false`, `accepted_manifest=false`, and
`labels_are_corpus_approval=false`; they do not create training shards,
classifier outputs, accepted manifests, or final 35k labels.

## Final 35k Freeze-Label Runner

The final freeze-label runner is separate from both the dormant 6,000 seed
runner and the adaptive borderline-mining runner. It requires a positive
final-manifest allowlist before live mode: exactly 35,000 candidates,
`label_readiness.ready_for_paid_labeling=true`,
`selection_algorithm_version=teacher_candidate_manifest_v1`, and
`manifest_scope=final_35k_teacher_label_freeze_candidates`. A seed manifest,
adaptive attempt manifest, diagnostic integration report, accepted corpus
manifest, or training shard is not a valid input even if it contains no raw
text.

The command intentionally writes two durable JSONL surfaces:

- `--out-votes`: one no-raw row for each DeepSeek vote, including parsed and
  parse-error rows. This is the resume and audit ledger.
- `--out-labels`: one no-raw resolved final-label row per candidate whose vote
  evidence is complete enough to label. This is the downstream teacher-label
  input. It must not be inferred from vote-row line counts.

Final labeling uses the same prompt-v3 DeepSeek V4 Pro non-thinking request
contract as accepted BoF mining: JSON Output, `temperature=0.2`,
`max_output_tokens=512`, `stream=false`, and model output restricted to
`label` plus `borderline`. It runs exactly BoF3: vote indexes 0, 1, and 2.
There is no BoF5 escalation, calibration slice, audit slice, or semantic
extra-vote path in final freeze labeling. BoF5 remains only for adaptive
borderline mining, where disagreement is the selection signal rather than final
truth.

Aggregation is fail-closed. A final `safe` row is allowed only when every
required parsed vote is `label=safe` and `borderline=false`. Complete parsed
evidence with reject votes, label splits, or any `borderline=true` vote becomes
a final `reject` row with route provenance such as `unanimous_reject`,
`mixed_fail_closed_reject`, or `borderline_fail_closed_reject`. When all three
required vote attempts exist but one or more model outputs are malformed,
truncated, empty, or otherwise `parse_error`, the candidate is also labeled
final `reject` with route `parse_error_fail_closed_reject`; malformed output is
not affirmative safe evidence. Missing votes, unresolved transport failures,
private refetch failures, and incomplete vote sets remain repair-blocking with
route `parse_or_transport_repair_required`; the main batch does not perform
semantic/model-output retries. Private refetch failures before a provider call
are treated as execution failures: already-written vote rows remain resumable,
the affected candidate is queued for explicit rerun/private adjudication, and
the public repair item stores only no-raw identity and hashed error metadata.

The current budget policy is available-balance based. Shailesh's roughly USD 40
top-up is not the spend cap by itself; it is intended to raise the available
DeepSeek balance. Once the exact manifest and runner/canary gates are ready, set
`--budget-cap-usd` to the verified available balance or to a deliberately chosen
lower cap, with `--minimum-balance-reserve-usd 0` when the instruction is to use
the account balance rather than preserve reserve. The final run must not require
the full 35k estimate to fit in advance. If DeepSeek returns 402 or otherwise
exhausts balance mid-run, the runner must stop as a resumable partial state with
partial votes, resolved labels, progress, cost ledger, unfinished-row counts,
repair queue, and no-raw evidence preserved.

Canonical command shape:

```bash
uv run --locked python -m twc.data.teacher_labeling run-final-label-batch \
  --manifest reports/teacher_labeling/final_35k_teacher_candidates.json \
  --schedule reports/source_schedules/<final_source_schedule>.json \
  --out-votes reports/teacher_labeling/final_35k_votes.jsonl \
  --out-labels reports/teacher_labeling/final_35k_labels.jsonl \
  --report reports/teacher_labeling/final_35k_label_batch_report.json \
  --repair-queue reports/teacher_labeling/final_35k_repair_queue.json \
  --progress reports/teacher_labeling/final_35k_progress.json \
  --row-provider hf-fsspec \
  --max-docs 35000 \
  --current-balance-usd <verified-balance> \
  --budget-cap-usd <verified-balance-or-explicit-spend-cap> \
  --minimum-balance-reserve-usd 0 \
  --target-concurrency 200 \
  --private-refetch-concurrency 128
```

Omitting `--live` writes readiness/progress/report/repair diagnostics without
private refetch or provider calls. Adding `--live` later also requires
`DEEPSEEK_API_KEY`, launch-time balance verification, exact final-manifest
readiness, reviewed output paths, and explicit coordinator unlock.

## Rejected-35k Chunk Relabeling

Rejected-document chunk relabeling is the bridge from the final 35k
document-level labels to the chunked-clean-run lane. It does not create corpus
approval, clean runs, accepted manifests, tokenizer shards, or training shards.
It only relabels natural-block chunks from final-label `reject` parents so the
later chunk classifier has stronger evidence than inherited document labels.

The builder consumes the final 35k candidate manifest, final 35k label JSONL,
and final refetch schedule. It privately refetches rejected parents, segments
them with `chunked_clean_run_v1`, packs natural blocks up to a target proxy
token count, and writes:

- `chunk_relabel_manifest.json`: no-raw public chunk candidates.
- `chunk_relabel_private_map.json`: ignored private span map needed to
  reconstruct chunk text at controlled labeling time.
- `chunk_relabel_build_report.json`: no-raw build counts and hashes.

Use the plan-backed operator for live work. It prevents stale Python constants,
overlapping stages, and accidental relabeling of already completed smaller
stages. Status and run commands validate the plan's input file hashes,
canonical provider config, and existing stage manifest identity before adopting
or writing a stage. The operator plan is no-raw and hash-bound:

```bash
uv run --locked python -m twc.data.chunk_relabel_operator create-plan \
  --out reports/teacher_labeling/chunk_relabel_run_plan.json \
  --manifest reports/teacher_labeling/final_35k_teacher_candidates.json \
  --labels-jsonl reports/teacher_labeling/final_35k_bof3_labels_2026_06_07.jsonl \
  --schedule reports/teacher_labeling/final_35k_refetch_schedule.json \
  --start-parent-skip 0 \
  --end-parent-skip 18200 \
  --stage-size 100 \
  --target-concurrency 500 \
  --private-refetch-concurrency 32 \
  --tranche-chunks 512 \
  --budget-cap-usd <explicit-run-cap> \
  --minimum-balance-reserve-usd 0.05
```

Check adoption/handoff status before live execution:

```bash
uv run --locked python -m twc.data.chunk_relabel_operator status \
  --plan reports/teacher_labeling/chunk_relabel_run_plan.json
```

The status output is the operator handoff surface. It reports input hash drift,
provider-config drift, lock state, completed stages, completed vote/label
counts, cumulative completed cost, next parent skip, and no-raw flags. A status
of `ready` is necessary but not sufficient for paid relaunch: compile/tests,
no-paid source timing, balance/cost projection, and a small paid canary still
gate scale.

Then run from the plan:

```bash
.venv/bin/python -m twc.data.chunk_relabel_operator run \
  --plan reports/teacher_labeling/chunk_relabel_run_plan.json \
  --row-provider hf-fsspec \
  --columns text,id \
  --source-concurrency 8 \
  --live
```

For a no-paid operator canary, use the same operator entrypoint with the fake
provider. Fake-provider mode constructs live-shaped requests, writes vote and
label artifacts, records `live_requested=true`, and keeps
`paid_api_calls_performed=false`; it does not call DeepSeek or check live
balance. The CLI redirects fake-provider output and HF cache under
`/private/tmp/twc_chunk_relabel_fake_<plan-hash-prefix>/` so fake-complete
labels cannot be adopted by production status scans:

```bash
.venv/bin/python -m twc.data.chunk_relabel_operator run \
  --plan reports/teacher_labeling/chunk_relabel_three_stage_canary_plan.json \
  --row-provider hf-fsspec \
  --columns text,id \
  --source-concurrency 8 \
  --source-retry-count 3 \
  --source-retry-backoff-seconds 1.0 \
  --fake-provider \
  --max-stages 1
```

The low-level builder and batch runner remain available as primitives, but
manual live operation should go through the operator. The operator creates a
per-plan lock under the output root so two launches cannot write the same stage
at once. If a crash leaves the lock behind, never remove it by hand first:
inspect status, verify the PID state, and only use the recovery command when it
reports a stale/dead, malformed, or different-plan lock:

```bash
uv run --locked python -m twc.data.chunk_relabel_operator recover-lock \
  --plan reports/teacher_labeling/chunk_relabel_run_plan.json
```

The operator refuses to recover locks whose PID is live or permission-unknown.
The live runner uses
DeepSeek V4 Pro non-thinking BoF3 with the same two-field prompt-v3 output
schema as final 35k labeling. Vote rows are append-only and resume-safe; stale
vote rows are rejected by manifest, provider config, prompt, request hash,
group key, idempotency key, label unit, parent id, and chunk text hash.
The plan `budget_cap_usd` is a run-level cap across adopted completed stages
and future stages, not a fresh per-stage allowance.
For newly built stages in the same operator process, raw parent rows are reused
only in memory for request construction so the stage does not pay the Hugging
Face sparse-refetch cost twice. This cache is never written to public reports,
manifests, progress files, or repair queues; resumed/prebuilt stages fall back
to the normal private refetch path. The operator does not background-prefetch
future stages by default; stage `N+1` must not start Hugging Face row reads
while stage `N` still needs private rows for labeling.

The wall-clock target is to finish as fast as safely possible, ideally under
six hours after relaunch. Six hours is not a stop condition by itself. Continue
past it when progress is healthy, durable vote rows are increasing, error rates
are clean, balance/cap gates hold, and no raw/secret boundary is violated. Stop
on lock ambiguity, input/provider drift, duplicate idempotency, repeated source
timeouts, auth/balance failure, unknown paid-outcome spikes, sanitizer failure,
or no durable vote progress.

Throughput settings must be measured against the current machine, network path,
Hugging Face/fsspec behavior, and DeepSeek response behavior before scale.
Sweep no-paid source concurrency in small bounded probes, keep the fastest
zero-error setting, and lower it immediately on repeated source timeouts. Ramp
provider concurrency only after p95 latency, 429/5xx/timeout counts, and durable
vote-write throughput stay clean. Performance optimization is not allowed to
weaken no-raw, idempotency, budget, or stop/restart guarantees.

```bash
uv run --locked python -m twc.data.chunk_relabeling run-chunk-label-batch \
  --manifest reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_relabel_manifest.json \
  --private-map reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_relabel_private_map.json \
  --schedule reports/teacher_labeling/final_35k_refetch_schedule.json \
  --out-votes reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_votes_live.jsonl \
  --out-labels reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_labels_live.jsonl \
  --report reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_label_report_live.json \
  --repair-queue reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_repair_live.json \
  --progress reports/teacher_labeling/rejected_35k_chunk_relabel_stage0000_0100/chunk_progress_live.json \
  --row-provider hf-fsspec \
  --max-chunks 100 \
  --current-balance-usd <verified-balance> \
  --budget-cap-usd <explicit-stage-cap> \
  --minimum-balance-reserve-usd 0 \
  --target-concurrency 8 \
  --tranche-chunks 1 \
  --private-refetch-concurrency 1
```

Omit `--live` for dry-runs. Add `--live` only after verifying the balance,
stage manifest, private map, output paths, and external-provider approval.
If private refetch or transport fails transiently, resume with the same vote and
label paths; already-written vote rows remain durable. Malformed, length,
empty, or schema-invalid DeepSeek responses are repair/rerun work, not strong
ready chunk labels.

## Candidate Inputs

The candidate selector can consume two public candidate shapes:

- decision-log candidates that already carry a public document id and text hash
  from a no-raw-text decision row;
- source-row distribution candidates that carry deterministic schedule/file/row
  identity but intentionally do not yet carry a private text hash.

The source-row distribution shape is the expected path for the final 35k
candidate pool. Public artifacts contain no raw text and include:

- stable candidate id plus source/schedule/row identity;
- decision status and stage;
- global schedule index, source file id, row group, and local row identity;
- source metadata such as domain, dump, language, language score, token count,
  score, and wrapper type;
- risk tags, rejection reasons, filter scores, and window-failure summaries.

Candidate manifests set `raw_text_serialized=false`, `token_serialized=false`,
`accepted_manifest=false`, and `labels_are_corpus_approval=false`. Candidate
generation fails closed if a public decision row contains raw-text-like fields
such as `text`, `document_text`, `content`, `snippet`, `raw_context_snippet`, or
`matched_text`.

When a source-row candidate lacks a text hash, the no-raw-text batch plan records
`text_hash_status=pending_private_refetch` and
`private_refetch_required_before_live_call=true`. The actual DeepSeek request is
allowed only after controlled private refetch; that request computes the real
UTF-8 SHA-256 text hash from the private text and carries it into response
parsing. Public planning must not invent a fake text hash.

## 35k Distribution

The final target is 35,000 DeepSeek V4 Pro labels:

| Bucket | Labels |
|---|---:|
| Representative / coverage | 8,000 |
| Deterministic high-risk | 5,000 |
| Subtle implicit risk | 5,000 |
| Source-wrapper/channel risk | 3,200 |
| Safe STEM/control | 3,000 |
| Safe non-STEM boundary survivors | 2,000 |
| False-friend hard negatives | 2,300 |
| Long-document window cases | 2,000 |
| Active-learning borderline | 4,000 |
| Coverage repair reserve | 500 |

The earlier seed-label plan reserved 6,000 all-DeepSeek labels: 4,500 for
provisional mining classifiers and 1,500 labeled holdout examples for checking
provisional mining scores. That lane is implemented but paused by the 2026-06-05
strategy reset and is not the planned fallback. The current route for the 4,000
active-learning borderline bucket is direct teacher-disagreement mining with
best-of-N DeepSeek V4 Pro non-thinking calls after prompt-v3 review. Reopening
the 6,000-label provisional classifier contingency requires an explicit user
decision; after the 4,000-row borderline set exists, redundant 6k provisional
code/docs should be removed before paper sharing.

## Label Units

The default label unit is a full document when the full document fits the
private prompt budget. Very long documents use `window_bundle`: the failing
window, halo windows, and safe contrast windows are rendered together. A risky
window-bundle label supports whole-document rejection. A safe window-bundle
label is a local negative only and must not be promoted to whole-document safe.

## Dry-Run Batch Planning

Before any DeepSeek call, create a no-raw-text batch plan:

```bash
uv run --locked python -m twc.data.teacher_labeling plan-label-batch \
  --manifest reports/teacher_labeling/phase2a_hf_9223_teacher_candidates.json \
  --out reports/teacher_labeling/deepseek_v4_pro_pilot_60_plan.json \
  --label-count 60 \
  --mode pilot \
  --model deepseek-v4-pro
```

Pilot planning is capped to 50-100 documents by policy. Batch plans record
candidate identity, derived source identity, schedule identity,
bucket/subbucket, label unit, text-hash status, prompt hash, request hash,
config hash, estimated input/output tokens, estimated cost, budget bucket, and
all-Pro 35k budget visibility. They perform no paid API calls and store no raw
text, raw URLs, token ids, secrets, accepted manifests, or training shards.

## Strict Parser

DeepSeek responses must be a single exact JSON object. The parser accepts only
the two model-emitted semantic keys `label` and `borderline`. Extra prose,
missing fields, wrong enums, non-boolean `borderline`, old schema fields such as
label unit/categories/spans/reasons, truncation, or empty JSON-mode content
become structured `parse_error` rows. For seed/adaptive batches these rows are
queued for separate adjudication or repair. For final 35k BoF3 labeling, a
candidate with all required vote attempts present but one or more `parse_error`
votes gets a conservative final `reject` and the no-raw repair queue remains
audit evidence. The harness appends label unit and all audit metadata after
successful parsing. No deterministic repair is attempted in the main
teacher-labeling batch.

Parse-error rows contain candidate/request identity, response hash, error code,
hashed error message class, `finish_reason`, safe schema failure path/kind/rule,
hashes of invalid model-provided values or extra keys, and malformed-content
hash/length, usage, latency, and cost if present. They do not store the
malformed content. Main-batch provider/model output failures are not
immediately relabeled with a prompt mutation; they are summarized in a no-raw
repair queue. HTTP transport retries for retryable status codes are separate
and remain allowed.

## Parse-Error Repair And Adjudication

The main DeepSeek batch only records `parse_error` rows. It does not repair
malformed JSON, change prompts, or relabel the same document inline. Repair is a
separate append-only workflow built from the public parse-error rows:

```bash
uv run --locked python -m twc.data.teacher_labeling export-repair-queue \
  --labels-jsonl reports/teacher_labeling/deepseek_v4_pro_seed_6000_labels.jsonl \
  --out reports/teacher_labeling/deepseek_v4_pro_seed_6000_repair_queue.json \
  --mode manual_adjudication \
  --source-label-jsonl-sha256 <sha256-of-label-jsonl>
```

For small queues, the exported work items can be assigned to controlled
manual/sub-agent adjudication. A public work item contains only candidate and
request identity, response hashes, schema diagnostics, finish reason,
usage/cost/latency summaries, and instructions to perform any private refetch or
private relabel outside public artifacts. It must not contain malformed model
content, corpus text, snippets, raw URLs, token ids, secrets, accepted manifests,
or training shards.

For larger queues, use `--mode explicit_rerun`. That manifest records lineage to
the failed idempotency key, original request hash, failed response hash, reason,
estimated additional cost, one-document-per-request semantics, and the explicit
launch unlock required before any paid rerun. The rerun creates new
idempotency keys; it does not mutate the failed row.

Format-only repair from malformed model output is not a public workflow. It
would require raw malformed response content, which is intentionally absent from
public artifacts. If a future private operator chooses to inspect raw response
content, that inspection must remain private and produce only a new parsed row
or a no-raw adjudication event.

Final reconciliation is append-only. A parsed repair/adjudication row is linked
back to the original `parse_error` row through a separate reconciliation event
with the original failed idempotency key, original request/response hashes, the
new resolved row idempotency key and hashes, the adjudication method, and
`historical_parse_error_mutated=false`.

The hardened 50-document pilot used staged compact retries to diagnose and
recover the pilot artifact after the initial 8192-token cap caused reasoning
exhaustion. That is historical evidence, not the canonical large-batch policy.
Future 6k/35k runs should use the high production output cap and queue any
remaining malformed outputs for a later explicit repair/adjudication workflow.

## 6k Seed-Launch Readiness

Before the 6,000 DeepSeek V4 Pro seed-label batch, write a no-raw readiness
manifest derived from the larger source candidate pool:

```bash
uv run --locked python -m twc.data.teacher_labeling build-seed-manifest \
  --manifest reports/teacher_labeling/phase2a_hf_9223_teacher_candidates.json \
  --out reports/teacher_labeling/deepseek_v4_pro_seed_6000_candidates.json \
  --seed-label-target 6000 \
  --train-label-count 4500 \
  --holdout-label-count 1500
```

The source 9,223-row candidate manifest may remain not final-35k-ready. The
derived seed manifest is narrower: it proves only the exact 6,000-label seed
path, the 4,500/1,500 train/holdout split, source-row identity for private
refetch, and no-raw public artifact behavior.

Then write the no-raw readiness report:

```bash
uv run --locked python -m twc.data.teacher_labeling check-seed-launch-readiness \
  --config configs/teacher_labeling.yaml \
  --manifest reports/teacher_labeling/deepseek_v4_pro_seed_6000_candidates.json \
  --out reports/teacher_labeling/deepseek_v4_pro_seed_6000_launch_readiness.json \
  --repair-workflow-available
```

This check performs no paid calls. It blocks launch unless the provider config
matches DeepSeek V4 Pro production policy, an exact 6,000-label seed plan can be
formed, the candidate manifest is schema-valid and label-ready, cost can be
estimated, private refetch is explicitly gated, idempotency/resume requirements
are present, parse-error repair is available, public artifact no-raw policy is
present, launch-time pricing has been verified, and the user has explicitly
unlocked paid calls. If the candidate manifest is missing or not label-ready,
the report says so instead of fabricating readiness.

The 6,000 seed labels split into 4,500 provisional classifier-training labels
and 1,500 labeled holdout examples for checking/calibrating provisional mining
scores. The holdout is labeled; it is not an unlabeled pool.

## Live-Call Guard

The live client shell is disabled by default. A later paid run must provide all
of these unlocks:

- explicit `--live`;
- explicit `--max-docs`;
- explicit budget cap;
- `DEEPSEEK_API_KEY` in the local environment;
- launch-time price/balance check;
- resume-safe output path and cost ledger.

The 50-document pilot runner is locked to exactly 50 documents and a maximum
USD 2.00 cap. It refetches each scheduled row privately before a request,
checks schedule/file/row identity, computes the configured text hash from the
private labeling text, and refuses text-hash drift. Persisted pilot
artifacts contain request/response hashes, cost/usage/latency summaries, parse
status, finish reasons, and refetch audit hashes only. The first call is a
canary; if it does not parse cleanly with `finish_reason=stop`, the runner stops
instead of continuing the batch.

The seed runner is a separate exact-6,000 path and does not weaken the pilot
guard:

```bash
uv run --locked python -m twc.data.teacher_labeling run-seed-label-batch \
  --manifest reports/teacher_labeling/deepseek_v4_pro_seed_6000_candidates.json \
  --schedule reports/source_schedules/fineweb_edu_sample100bt_freezable_schedule_10k_dense20.json \
  --out-labels reports/teacher_labeling/deepseek_v4_pro_seed_6000_labels.jsonl \
  --report reports/teacher_labeling/deepseek_v4_pro_seed_6000_report.json \
  --row-provider hf-fsspec \
  --hf-cache-dir reports/teacher_labeling/hf_seed_row_cache \
  --max-docs 6000 \
  --budget-cap-usd 65
```

Omitting `--live` writes a no-paid dry-run report and performs no private
refetch or DeepSeek calls. Adding `--live` later requires `DEEPSEEK_API_KEY`,
keeps `--max-docs` locked to exactly 6,000, preserves a USD 0.75 reserve by
default, writes one resume-safe JSONL row per candidate, and stops after the
first call unless the canary parses with `finish_reason=stop`. Resume only skips
existing rows after proving they are a prefix of the same seed manifest and
provider configuration, so reruns do not refetch already completed private rows.

Parallel production labeling must preserve one-document-per-request semantics:
each worker builds one private request, sends one DeepSeek call, parses one
response, and writes only no-raw rows. The hardened 50-document pilot exercised a
16-worker tier followed by a 64-worker-capacity tier without 429 responses.

Tests and dry-runs must not call DeepSeek. Public artifacts must not contain API
keys, raw corpus text, raw URLs/query strings, snippets, matched text, token ids,
accepted corpus manifests, or training shards.

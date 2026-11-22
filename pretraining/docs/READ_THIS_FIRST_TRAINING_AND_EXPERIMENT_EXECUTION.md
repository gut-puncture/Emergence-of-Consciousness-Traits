# READ THIS FIRST — Running TWC Training Runs and Experiments

> **Mandatory orientation:** Every new TWC task must read this document before
> planning, preparing, launching, resuming, fetching, or assessing any training
> run or experiment. This is the operator map for the current simplified
> architecture. Study-specific scientific plans and task-local paid-run cards
> add constraints; they do not replace this lifecycle.

## The one active architecture

TWC has one training lifecycle:

`readable recipe or study -> immutable run.json -> standard runtime evidence -> twc-pretrain -> fetched result -> study-specific assessment`

- `twc-pretrain` is the only active training operator.
- A diagnostic, canary, study arm, and final pretraining run are ordinary run
  configurations. They are not separate trainer types.
- `twc-pretrain` does not choose architectures, rank experiment arms, rent
  hardware, terminate hardware, or authorize private data.
- An experiment compiler defines the exact scientific runs. A study-specific
  assessor compares their fetched results afterward.
- Operational stop/resume is supported. Whether a resumed trajectory is
  scientifically eligible is decided by the study assessor, not the trainer.

Do not use the retired `twc-gpu-campaign`, `twc-config-closeout`, or
`twc-config-checkpoint-fetch` commands for new work.

## Terms that must not be conflated

- **Training configuration:** The immutable `RUN/run.json`. It binds the model,
  data, tokenizer, seed, optimizer, schedule, evaluator, token horizon, and
  scientific source identity.
- **Execution settings:** The task-local `RUN/execution.json`. It binds host-side
  choices such as device, storage paths, checkpoint cadence, telemetry cadence,
  and read mode. It is not scientific ranking evidence.
- **Runtime evidence:** The eight validated files under `RUN/runtime/` that bind
  prepared records, the global/device plans, release, environment, data
  placement, and storage.
- **Training run:** One run directory operated by `twc-pretrain`.
- **Experiment:** One or more immutable training configurations plus a frozen
  comparison policy and a study-specific result adapter/assessor.
- **Frozen study bundle/reference:** Pre-existing immutable scientific evidence
  used by an assessor to interpret a fetched run. It is not regenerated for
  every new physical attempt.

## Source-truth order before touching a run

1. Read the project-root `AGENTS.md`.
2. Read this document completely.
3. Read the current tracker and latest append-only research-log entries.
4. Read the scientific plan or recipe for the specific study.
5. Read `docs/pretraining_runner_contract.md` and the task-local operator card.
6. Verify live Git from `code/` and verify live provider state before any paid
   action.

Chat summaries and old private launch commands are never execution authority.

## 1. Produce the immutable run configuration

Every run directory begins with exactly one immutable `run.json` produced by a
reviewed recipe or study compiler. Do not hand-edit a compiled `run.json`, copy
hashes from prose, or add operator state before checking it.

The current configuration-study compiler is:

```text
uv run --locked twc-config-experiment compile \
  --study-root STUDY_ROOT \
  --output-root OUTPUT_ROOT
```

It emits one directory per selected condition, containing only `run.json`.
This command is specific to the checked configuration-study format; it is not a
universal experiment authoring command. A different study must use its own
reviewed compiler to produce the same immutable run boundary.

The checked base-pretraining recipe uses the same boundary without requiring
host, device, placement, storage, or other attempt-time facts:

```text
uv run --locked twc-pretrain compile-base \
  --run RUN --run-id base-100m \
  --model-config configs/model_100m.yaml \
  --training-config configs/train_100m.yaml \
  --pack-manifest PACK_MANIFEST \
  --tokenizer-manifest TOKENIZER_MANIFEST \
  --train-view TRAIN_VIEW \
  --validation-view VALIDATION_VIEW \
  --global-plan GLOBAL_PLAN \
  --semantic-code-receipt SEMANTIC_CODE_RECEIPT \
  --evaluation-manifest EVALUATION_MANIFEST \
  --evaluation-implementation EVALUATION_IMPLEMENTATION
```

The compiler reads and cross-validates the hash-bound artifacts itself, then
creates `RUN/run.json` atomically. It does not create `execution.json`, runtime
evidence, operator state, or an execution attempt; those remain later lifecycle
steps.

Validate every emitted run before staging runtime evidence:

```text
uv run --locked twc-pretrain check --run RUN
```

`check` is read-only. `start` and `resume` repeat configuration validation
before mutating operator state.

## 2. Assemble the standard runnable directory

Before `start`, the directory must have this shape:

```text
RUN/
  run.json
  execution.json
  runtime/
    train-records.json
    validation-records.json
    global-plan.json
    device-plan.json
    release.json
    environment.json
    placement.json
    storage.json
```

`operator/`, checkpoints, outcomes, evaluation ledgers, and `result.json` are
created by the lifecycle. Do not pre-create or forge them.

There is intentionally no second orchestration framework and no single
campaign-style `prepare` command. The task-local operator packet calls the
existing builders below, records their outputs, and stages the exact validated
files at the fixed paths.

### Release and bootstrap

```text
uv run --locked twc-pretrain build-release \
  --source-root CODE_ROOT \
  --release-root RELEASE_ROOT \
  > RUN/runtime/release.json

uv run --locked twc-pretrain verify-bootstrap \
  --release-receipt RUN/runtime/release.json \
  --python PYTHON_EXECUTABLE \
  --python-major 3 \
  --python-minor 11
```

The deployed children must launch from that content-addressed release, not from
an arbitrary checkout.

### Environment

```text
uv run --locked twc-pretrain capture-environment \
  --device DEVICE \
  --precision PRECISION \
  --attention-backend ATTENTION_BACKEND \
  --optimizer-backend OPTIMIZER_BACKEND \
  --compile-mode COMPILE_MODE \
  --output RUN/runtime/environment.json
```

Add `--allow-tf32` only when the frozen execution policy permits it.

### Data placement and worker records

```text
uv run --locked twc-pretrain verify-data-placement \
  --pack-root PACK_ROOT \
  --manifest PACK_MANIFEST \
  --output RUN/runtime/placement.json

uv run --locked twc-pretrain prepare-worker-data \
  --pack-root PACK_ROOT \
  --manifest PACK_MANIFEST \
  --view TRAIN_VIEW \
  --validation-view VALIDATION_VIEW \
  --global-plan GLOBAL_PLAN \
  --placement-receipt RUN/runtime/placement.json \
  --tokenizer-hash TOKENIZER_HASH \
  --max-loss-tokens MAX_LOSS_TOKENS \
  --max-attention-cells MAX_ATTENTION_CELLS \
  --max-sequences MAX_SEQUENCES \
  --records-output RUN/runtime/train-records.json \
  --validation-records-output RUN/runtime/validation-records.json \
  --device-plan-output RUN/runtime/device-plan.json
```

Stage the already validated `GLOBAL_PLAN` byte-for-byte as
`RUN/runtime/global-plan.json`. Do not regenerate, reformat, or rehash it by
hand. Never inspect or serialize raw corpus text or token arrays while staging.

### Storage

Choose a topology supported by the public generic run settings. For a direct
persistent checkpoint root:

```text
uv run --locked twc-pretrain validate-storage \
  --control-root RUN \
  --data-root PACK_ROOT \
  --checkpoint-topology direct_persistent \
  --durable-root CHECKPOINT_ROOT \
  --data-sentinel DATA_STORAGE_ID \
  --durable-sentinel CHECKPOINT_STORAGE_ID \
  --required-checkpoint-bytes REQUIRED_CHECKPOINT_BYTES \
  --reserve-bytes RESERVE_BYTES \
  --output RUN/runtime/storage.json
```

Use `local_only` only with its corresponding local-root and local-sentinel
arguments. Missing declared mounts or storage identities must fail closed.

### Execution settings

`RUN/execution.json` is explicit task-local input. It must contain exactly:

```json
{
  "artifact_kind": "twc_pretraining_execution_v1",
  "schema_version": 1,
  "pack_root": "ABSOLUTE_PACK_ROOT",
  "checkpoint_root": "ABSOLUTE_CHECKPOINT_ROOT",
  "checkpoint_topology": "direct_persistent",
  "device": "cuda:0",
  "read_mode": "local_mmap",
  "pin_memory": true,
  "telemetry_every_steps": 1,
  "checkpoint_every_loss_tokens": 1,
  "checkpoint_max_interval_seconds": 1,
  "durable_every_loss_tokens": 1,
  "durable_max_interval_seconds": 1
}
```

The integer values above demonstrate required positive types, not recommended
cadences. Replace them with the reviewed task-local settings. Durable cadence
cannot be tighter than checkpoint cadence. Use `buffered_pread` for removable
or network-backed data. Use `pin_memory=false` for CPU attempts.

Never derive these settings from an old filled run card. The current task-local
packet owns the paths, topology, thresholds, cost limits, and approval state.

## 3. Operate one ordinary training run

Use these commands for every run type:

```text
uv run --locked twc-pretrain check --run RUN
uv run --locked twc-pretrain start --run RUN --startup-timeout-seconds 600
uv run --locked twc-pretrain status --run RUN
uv run --locked twc-pretrain checkpoint --run RUN --timeout-seconds 300
uv run --locked twc-pretrain stop --run RUN --timeout-seconds 300
uv run --locked twc-pretrain resume --run RUN --startup-timeout-seconds 600
```

- `start` initializes one append-only attempt and creates the internal worker
  specification from the fixed run/runtime inputs.
- `status` is the cheap readback surface.
- Concurrent checkpoint callers reconcile one immutable request and
  acknowledgement through the existing writer lease.
- `stop` requires a verified checkpoint. It does not assess the experiment or
  terminate hardware.
- `resume` creates a successor attempt from an exact verified checkpoint. It
  never rewrites the scientific `run.json`.
- A completed run produces one immutable generic `result.json` bound to the
  terminal checkpoint and evaluation evidence.

Do not change scientific configuration during a run. Change operational
settings only between attempts, only where current compatibility validation
permits it, and preserve the predecessor/checkpoint lineage.

## 4. Fetch and validate without the former host

After a durable stop or completion:

```text
uv run --locked twc-pretrain fetch \
  --run RUN \
  --destination FETCHED_RUN

uv run --locked twc-pretrain status --run FETCHED_RUN
```

Fetch copies the fixed evidence set, validates identities, and writes its
no-raw receipt last. It is retry-safe for identical content, rejects conflicts,
and does not terminate infrastructure.

Keep comparison, report assembly, hashing, archive work, and scientific
assessment off paid GPUs after fetch.

## 5. Run an experiment through the same lifecycle

An experiment is orchestration around ordinary runs:

1. Freeze the study conditions, metric, seeds, data, evaluator, token horizons,
   and decision rule before execution.
2. Use the study compiler to emit one immutable run directory per arm.
3. Assemble each run through the standard runtime-evidence path above.
4. Execute each arm with the same `twc-pretrain` commands.
5. Fetch each terminal run to source-free local evidence.
6. Convert fetched generic results through the study-specific adapter.
7. Compare only with the study's frozen assessment rule.

For the current configuration-study assessor, the public result surface is:

```text
uv run --locked twc-config-experiment build-result-v3 \
  --result-id RESULT_ID \
  --run FETCHED_RUN \
  --bundle-root FROZEN_STUDY_BUNDLE_ROOT \
  --output RESULT_OUTPUT

uv run --locked twc-config-experiment build-comparison-v3 \
  --run FETCHED_RUN_1 --bundle-root FROZEN_BUNDLE_ROOT_1 \
  --run FETCHED_RUN_2 --bundle-root FROZEN_BUNDLE_ROOT_2 \
  --output COMPARISON_OUTPUT
```

Repeat the paired `--run` and `--bundle-root` arguments once per arm.

### The bundle rule

The bundle supplied to assessment is the frozen study reference that defined
the arm. It is not a new runtime artifact and is not regenerated for each
physical run or release. The current active `compile` command intentionally
emits `run.json` directories only. Creating a new full bundle through internal
campaign-era builders would reintroduce retired campaign, authorization,
provider, and execution-path machinery.

Therefore:

- reuse the exact already-frozen study reference;
- let the adapter validate the fetched run against it;
- never rewrite or rehash the frozen bundle;
- never invent a new bundle with manual JSON or internal Python calls;
- if a genuinely new study has no reviewed compiler or assessment reference,
  stop and define only that study-specific seam before spending money.

`twc-pretrain` must remain study-agnostic. Do not add experiment-family branches
to the trainer.

## 6. Paid execution remains task-local and approval-gated

The repository commands do not rent or terminate hardware. Before any paid
action, the task-local packet must bind:

- exact Git commit and content-addressed release;
- provider, device, environment, capability, and current offer observation;
- data access, placement, storage, and full readback;
- shape or batch-size evidence required by that run;
- telemetry, utilization, throughput, checkpoint, and stop thresholds;
- manual cost monitoring, hard cap, reserve, and stop deadline;
- copyback, source-free validation, billing, and teardown steps;
- the exact resource IDs that the task owns and may mutate.

Provider-offer and other host-specific receipts may be built task-locally with
the existing evidence builders. The absence of a public one-command wrapper is
not permission to add a framework or producer CLI. Add production code only if
a concrete packet proves that a required invariant cannot be produced or
validated through the existing accepted builders.

Paid calls, GPU rental, private reads, training, and teardown always require the
explicit authority stated in `AGENTS.md` and the current task.

## 7. Historical-bundle compatibility

Some preserved configuration-study bundles contain release inventories naming
`gpu_campaign_cli.py` and `gpu_canary_fixture.py`, two producer modules retired
by SIMP-POST. The passive historical reader accepts exactly those two names in
a hash-valid historical release receipt. Current release builders and physical
release verification still use the current allowlist and cannot emit or execute
the retired producers.

The three frozen C2 references have been loaded through the current complete
bundle loader. This bounded reader compatibility does not authorize new bundle
producers, provider CLIs, schemas, campaign machinery, artifact migration, or
regeneration of a frozen study reference. Any other historical path mismatch
still fails closed.

## Prohibited shortcuts

- Do not run Git from the private project root; use `code/`.
- Do not inspect, paste, log, or commit raw private corpus text, URLs, secrets,
  accepted manifests, token arrays, or training shards.
- Do not hand-edit immutable configurations, receipts, bundles, attempts,
  checkpoints, outcomes, or results.
- Do not type hashes from prose when a builder or receipt owns them.
- Do not use old private campaign commands as current instructions.
- Do not turn task-local cost/provider monitoring into permanent architecture
  without evidence that the existing builders cannot support it.
- Do not add a new trainer for a diagnostic, canary, or experiment family.
- Do not treat a resumed trajectory as scientifically eligible unless the
  frozen study assessor explicitly permits it.
- Do not let throughput, cost, or target-adjacent probes override a study's
  frozen ranking metric.

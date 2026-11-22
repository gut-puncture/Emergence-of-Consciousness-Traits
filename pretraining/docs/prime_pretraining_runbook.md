# Provider-Neutral Pretraining Runbook

Mandatory first read:
**[READ THIS FIRST — Running TWC Training Runs and Experiments](READ_THIS_FIRST_TRAINING_AND_EXPERIMENT_EXECUTION.md)**.

This runbook covers the ordinary generic training lifecycle. Despite the historical
filename, no Prime-specific command or resource action is embedded in the code.
Resource creation, private reads, GPU use, scale start, and teardown require a
separate explicit approval and task-local operator packet.

The completed C1/G1 campaign remains accepted historical evidence. Its public
producer commands are retired, while passive readers and validators remain so
accepted artifacts can still be inspected without migration or rehashing. Do
not use the historical campaign guide to launch new work.

## Ordinary generic lifecycle

Compile a checked-in C2 or C3 study into exact immutable run directories:

```text
twc-config-experiment compile --study-root STUDY_ROOT --output-root OUTPUT_ROOT
```

Each output run uses the same operator surface, regardless of whether it is a
short diagnostic, frozen C2/C3 arm, canary, or final pretraining run:

```text
twc-pretrain check --run RUN
twc-pretrain start --run RUN
twc-pretrain status --run RUN
twc-pretrain checkpoint --run RUN
twc-pretrain stop --run RUN
twc-pretrain resume --run RUN
```

`check`, `start`, and `resume` validate the exact frozen configuration before
changing run state. The runner may resume operationally, but C2/C3 assessment
requires a fresh-from-zero trajectory. That scientific eligibility rule belongs
to assessment, not to the trainer.

Concurrent checkpoint commands share one immutable request and acknowledgement.
Normal live-writer contention waits within the caller's deadline; uncertain
writer ownership fails closed. A response-loss retry reuses the existing
request, while a later checkpoint command creates a newer generation.

After a durable stop or completion, copy the exact immutable evidence to an
explicit destination and validate it without depending on the former host:

```text
twc-pretrain fetch --run RUN --destination DESTINATION
twc-pretrain status --run DESTINATION
```

Fetch copies only the fixed run/configuration, attempt, checkpoint, outcome,
result, evaluation-ledger, evaluation-plan, and global-plan evidence set. It
validates content identities, writes a no-raw receipt last, reconciles identical
retries, rejects conflicting destination content, and never terminates hardware.

Build one study result from the validated fetched run, then compare repeated
matched run/bundle pairs through the existing assessment receipt:

```text
twc-config-experiment build-result-v3 --result-id RESULT_ID \
  --run RUN --bundle-root BUNDLE_ROOT --output OUTPUT
twc-config-experiment build-comparison-v3 \
  --run RUN --bundle-root BUNDLE_ROOT \
  --run RUN --bundle-root BUNDLE_ROOT \
  --run RUN --bundle-root BUNDLE_ROOT --output OUTPUT
```

The adapter accepts only the frozen C2 and C3 recipes. It revalidates their data,
model, schedule, evaluator, release, lineage, token horizon, evaluation-group
identity, and terminal checkpoint evidence. Historical evaluation-v2 attempts
remain readable; current attempts with a validation-records identity emit and
consume evaluation-v3 evidence.

## Before any rental

Complete locally:

1. Build accepted runs and the tokenizer under their existing private-read gate.
2. Run fused isolated packing; interrupted builds resume only from validated,
   hash-bound shard progress.
3. Build the exact finite train view, held-out validation view, and semantic
   global-step plan.
4. Relocate the pack to its intended data root and perform full readback.
5. Build the thin content-addressed release and verify offline bootstrap.
6. Estimate checkpoint/storage headroom and write the task-local operator packet.

The packet must bind the exact release, provider state, access and staging,
runtime-evidence builders, telemetry, manual cost monitoring, stop/resume,
copyback, and teardown. Stop and plan that seam separately if it requires manual
hash transcription, custom production code, or an unreviewed artifact layout.

Do not rent first and then design a runner, restore the entire private backup for
a canary, or perform pack/archive/report work on paid GPU time.

## CPU data commands

`twc-data plan --tokenizer-config configs/tokenizer.yaml` is read-only. Private
build commands require `--private-read-approval-id`; the ID is hashed in the
public receipt. Validate accepted, tokenizer, isolated-pack, view, and global-plan
stages before advancing. Production packing streams accepted text through the
frozen tokenizer directly into binary shards; it does not create tokenized JSON.

Use buffered reads when the placed pack is on removable or network storage. Use
mmap only after copying to stable local storage. Never let a missing mount
silently become a newly created local directory.

`twc-data place --source ... --destination ... --approval-id ...` performs a
retry-safe copy, full destination readback, and writes the placement receipt
consumed by worker-data preparation. Placement is an explicit approved action;
it is never chained automatically after packing.

For an already uploaded non-private configuration pack, use
`twc-pretrain verify-data-placement` to perform full shard hash/readback and emit
the host-specific placement receipt without importing the private CPU pipeline.

## Release and exact configuration

Build a code-only release with `twc-pretrain build-release`; verify its receipt
using `twc-pretrain verify-bootstrap`. Record the live Python, PyTorch/CUDA,
device, precision, and backend identity with `capture-environment`. Validate
storage roles and derive checkpoint capacity with `validate-storage`.

Low-level compile and evidence-builder commands remain available for exact
reproducibility. Their artifacts are internal inputs to the ordinary run
directory, not operator-managed architectural layers. Do not hand-author or
edit hash-bound artifacts.

Run `prepare-worker-data` with both `--view` and `--validation-view`; it emits
separate train and validation records plus the device plan. The worker refuses
READY if either split/view binding, exact plan, device partition, or live
placement identity has drifted.

## Recovery and guards

Use `repair-checkpoints` offline when a crash leaves partial or headless
generations. Repair quarantines incomplete state and promotes only a fully
verified generation. `record-backup` and `teardown-readiness --scope pod|disk`
record evidence only; they never terminate infrastructure.

Low-level `request`, `orphan`, and `takeover` commands are exceptional recovery
tools. Use them only from an approved task-local packet that binds the exact
attempt, former-host proof, and fencing epoch.

Worker telemetry reports phase, heartbeat, step, consumed loss tokens,
throughput, input wait, evaluation count, checkpoint and replication timing,
free space, peak GPU memory, utilization, stop latency, attempt/fence identity,
and data-loader settings. Cost remains task-local guard evidence, not a
fabricated worker metric.

Compilation and warmup are not throughput stalls. During steady-state training,
low utilization plus high input wait is an input-pipeline blocker; low
utilization plus low input wait is a model/kernel/shape blocker. Stop paid time
if useful GPU work cannot be restored inside the run-card window.

## Final canary acceptance

The sole final canary must use the production entrypoint and a minimal
representative private subset. Accept it only if:

- reference/native loss and gradients agree within declared BF16 tolerances;
- sequence isolation and exact token accounting hold;
- stop, verified checkpoint, process restart, and resume reproduce the cursor;
- no secrets, raw text, or tokens leak into logs or public receipts;
- device peak memory leaves the task-local reserve;
- post-warmup utilization, tokens/s, and input wait justify scale;
- checkpoint copyback and independent readback succeed; and
- actual price and projected wall time/cost remain inside the approved cap.

Any change to attention backend, compile mode, device slicing, release,
environment, data shape, credentials path, launch mechanism, resume behavior, or
placement invalidates the corresponding canary evidence.

## Closeout and teardown

Completion requires the exact target loss tokens, a persistent verified final
checkpoint, final evaluation and telemetry evidence, environment/release/artifact
hashes, and independent copyback validation. Pod readiness and disk readiness
are separate. Never infer termination from a readiness receipt and never
terminate either resource without explicit approval.

# Base Pretraining Data And Runner Contract

For the exact current operator sequence and run-directory layout, first read
**[READ THIS FIRST — Running TWC Training Runs and Experiments](READ_THIS_FIRST_TRAINING_AND_EXPERIMENT_EXECUTION.md)**.

The active production recipe trains one selected-wide
`100,291,200`-parameter decoder on exactly `2,610,181,035` unique next-token
loss targets with a maximum sequence length of `4,096`. It does not authorize
private-data reads, accepted-manifest creation, paid compute, or a GPU launch.
The runner remains architecture-generic: exact immutable configurations, not a
hard-coded R7-only trainer, distinguish diagnostics, study arms, and production.

The design has three owners:

- `twc-data` performs CPU/private-data work and publishes immutable artifacts.
- Reviewed recipe or study compilers emit immutable `run.json` configurations;
  `twc-pretrain` validates them, binds execution attempts, and owns lifecycle.
- the worker consumes validated artifacts; it never imports data-pipeline writers.

## One active data path

The production path is:

`accepted clean runs -> tokenizer -> fused tokenize/isolated pack -> finite view -> global step plan -> verified placement`

There is no tokenized-JSON spool and no fixed-record strict-pack path. Existing
R6 pack, view, run-contract, attempt, and operator-state artifacts must be
rebuilt. They are never patched in place because their hashes bind different
semantics.

Each clean run is an attention boundary. A run of `N` tokens contributes
exactly `max(0, N - 1)` loss targets. Long runs are split into at most 4,096-loss
token segments with a one-token overlap, so no target is dropped and no model
token can attend into another clean run. Tokens are stored as little-endian
uint16 shards; 28-byte indexes contain offsets and non-raw run/component
ordinals. The public manifest contains hashes, counts, and relative paths only.

The finite train view is the complete canonical, non-repeating train split,
with an exact final partial sequence when needed. Its loss-token count is
exactly `2,610,181,035`. Validation data remains split-disjoint and is not
included in that budget.

## Scientific identity versus execution identity

`twc_pretraining_run_contract_v2` is immutable scientific identity. It binds:

- pack manifest, finite train and held-out validation views, tokenizer, model, code release, and global plan;
- exact token/context targets and semantic optimizer-step caps;
- BF16, attention/loss backends, AdamW, WSD schedule, clipping, and seed.

`twc_pretraining_execution_attempt_v2` is append-only operational provenance.
It binds one device plan, release, environment, data placement, storage layout,
host, fencing epoch, prefetch policy, telemetry, checkpoints, and evaluation
cadences. Changing device slicing or hosts creates a successor attempt; it does
not change scientific identity. A successor names its parent attempt hash.

The checked production recipe compiles scientific identity before rental with
`twc-pretrain compile-base`. It consumes the checked model/training YAML and the
actual hash-bound pack, tokenizer, train view, validation view, global plan,
semantic-code, and evaluation artifacts. It emits only immutable `run.json`;
`twc-pretrain start` constructs the first attempt later from validated runtime
evidence and `execution.json`.

## Batch and optimization semantics

The global plan greedily groups canonical sequences under independent caps for
loss tokens, attention cells (`sum(sequence_length^2)`), and sequence count.
This plan is scientific. A device plan may split one global step into smaller
microbatches for the observed GPU envelope but may not reorder, repeat, drop,
or cross global-step boundaries.

The worker reads one batch ahead. Local stable storage may use read-only mmap;
removable or network-backed data must use bounded `pread`, avoiding SIGBUS when
a mount disappears. Flat batches carry `cu_seqlens`; both reference and native
variable-length attention enforce those hard boundaries.

All device slices belonging to one global step produce one optimizer update.
Loss is accumulated as `sum(loss_sum) / sum(loss_tokens)`, so unequal slices do
not create a mean-of-means bug. The WSD learning rate is sampled at the global
step's loss-token midpoint. Checkpoints restore model, optimizer, RNG, step,
and consumed-loss-token cursors together.

## Model contract

The selected production configuration is a 16-layer, width-640 decoder with 8 query
heads, 2 key/value heads, SwiGLU width 2,048, RMSNorm, RoPE, tied 32,768-token
embeddings, no biases, and no dropout. The exact unique parameter count is
`100,291,200`. The reference attention/loss paths establish CPU correctness.
The CUDA native-variable-length attention, fused AdamW, BF16, TF32, and device
slicing first receive diagnostic engineering proof under the separate GPU
campaign. That proof cannot replace the final private-data integrated canary.

## Storage, lifecycle, and failure rules

The production lifecycle uses synchronous verified durable replication. Training waits for
destination readback before advertising a durable generation, and stop/completion do
not return before verification. Telemetry reports blocked replication seconds and its
fraction of elapsed runtime. Background replication remains deferred until a bounded
single-flight implementation proves snapshot ownership, back-pressure, stop flushing,
and recovery on the rented-device path.

The hash-bound worker telemetry emits phase/heartbeat, step and consumed loss tokens,
loss-token throughput, recent-step p50, input-wait fraction, evaluation-batch count,
checkpoint and persistent-sync durations, synchronous-replication mode, blocked
replication seconds/fraction, free storage, peak GPU memory, GPU-utilization
status/median/sample count, measured request-or-signal stop latency, dataloader worker
count, prefetch factor, pin-memory status, and whether host-to-device copies were
nonblocking. An unavailable utilization sampler emits an explicit status and null
median, never a synthetic zero. `pin_memory` is rejected for CPU attempts; on CUDA it
pins all batch tensors and is the sole condition enabling nonblocking transfer.

Storage roles are explicit: control, data, local checkpoint, durable checkpoint,
and optional copyback. Validation never creates a missing declared mount. It
binds storage sentinels, groups capacity by filesystem, and reserves derived
checkpoint/headroom bytes.

Immediately before READY, the worker revalidates the release inventory, live
Python/Torch/CUDA environment, placed-data root/device/file identities and full
canonical shard SHA-256 values, and the data-role sentinel. Training and
evaluation records carry different split and view bindings. Every scheduled
evaluation consumes the complete finite held-out
view in cap-bounded batches; warmup and stable/decay boundaries force an
evaluation even when the ordinary token cadence has not elapsed.

The operator states include `STARTING`, `RUNNING`, `QUIESCING`, `STOPPED`,
`ORPHANED`, `COMPLETED`, and `FAILED`. Exactly one writer lease may mutate state.
Commands have immutable request IDs. Takeover requires proof that the former
host is terminated and increments the fencing epoch. `start` and `resume` use
one launch-scoped, owner-only namespace and reserve `STARTING` before process
creation. The watchdog becomes READY before the worker; both receipts bind the
same launch, run, attempt, fence, release, and installed performance policy,
while worker READY additionally binds environment, placement, and storage.
Only two exact live process identities and both canonical READY receipts can
append one immutable activation-history entry and atomically enter `RUNNING`.
There is no separate GO channel. Cheap storage/placement/release contradictions
fail before spawn and are repeated immediately before launch.

A startup failure that never entered `RUNNING` records a typed failure in the
terminal operator state, clears READY/process identities, retains zero cursor
and empty activation history, and emits no performance or scientific
observation. Cleanup is proved: uncertainty becomes `ORPHANED`. Once activated,
the terminal state retains its activation entry and the final performance
telemetry remains bound to that activation. Configuration-stage closeout uses
these distinct truths directly; it does not create launcher-failure or
short-runtime receipt families. For C3, every stopped attempt also has one
ordered preservation truth: the replicated-checkpoint receipt for a checkpointed
attempt, or the same runtime-closeout hash for an eligible checkpoint-free
noncompleted attempt. Completed C3 attempts cannot use the short-runtime
exemption.

Checkpoints support `local_only`, `direct_persistent`, or
`local_then_replicate`. Publications are generation directories with redundant
heads, verified deserialization, and optional durable replication. Recovery
quarantines partial generations, can promote a valid headless generation, and
falls back to the latest verified durable copy if local state is corrupt.

Canary stop/resume evidence is an exact ordered chain: one hash-valid result per
attempt, valid successor ancestry/fence, a controlled persistent predecessor
stop, exact restored checkpoint step/token/ordinal cursor, and strict
post-resume advancement. Merely having two attempt IDs is not evidence.

Token and elapsed-time checkpoint thresholds are both enforced. Pause, stop,
guards, and completion require an appropriate verified checkpoint. Disk
teardown additionally requires a verified copyback/backup receipt; pod and disk
readiness are separate decisions.

## Invariant propagation

| Invariant | Owner/writer | Consumer/validator | Gate and disproof |
|---|---|---|---|
| Every eligible target appears once; no cross-run attention | fused pack builder | pack validator, batch source, model | overlap/readback and attention-isolation tests |
| Train budget equals the exact finite target in `run.json` and is non-repeating | finite view | plan loader, run compiler, worker | prefix coverage and forged-plan tests |
| Evaluation is held out and boundary-complete | validation view and run contract | worker | split/view mutation and canary-before-cadence tests |
| Device slicing cannot alter optimizer semantics | global/device planners | worker, engine | unequal-slice and plan-binding tests |
| Resume restores one coherent cursor | engine/checkpoint manager | worker/operator | corrupt-copy and snapshot/restore tests |
| Only the fenced attempt can run | operator | READY validation, watchdog | stale fence, duplicate request, takeover tests |
| One stopped configuration attempt has one truthful runtime closeout | operator state and performance ledger | configuration copyback and local-readiness v2 | pre-RUNNING mutation, terminal telemetry, and retry-reconciliation tests |
| Paid GPU time excludes avoidable CPU/archive work | run card and lifecycle | operator and owner | final canary phase telemetry and stop gates |

## Current readiness boundary

Local synthetic CPU validation can prove contracts, reference math, lifecycle,
checkpoint recovery, and no-raw artifacts. It cannot prove private corpus yield,
real pack size, mount behavior, CUDA kernel compatibility, achieved throughput,
GPU saturation, thermal/provider stability, or cost. Those remain one final,
explicitly approved integrated canary after all implementation work is complete.
The retired non-private campaign producers are documented only as historical
evidence in `gpu_engineering_configuration_campaign.md`. New diagnostics and
canaries use this ordinary runner with an explicitly approved task-local packet;
they do not unlock production or F1.

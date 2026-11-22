"""Production-shaped, provider-neutral pretraining runtime.

The package deliberately separates immutable scientific run identity from
append-only execution attempts and keeps data preparation outside the training
process.
"""

from twc.pretraining.training.contracts import ExecutionAttempt, RunContract

__all__ = ["ExecutionAttempt", "RunContract"]

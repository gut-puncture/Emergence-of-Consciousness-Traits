"""CPU-only, private-data preparation contracts for R6 pretraining.

This package deliberately has no dependency on the training engine.  It owns
the one-way transition from a validated filtering selection handoff to frozen
clean-run, tokenizer, tokenized-run, and packed-data artifacts.
"""

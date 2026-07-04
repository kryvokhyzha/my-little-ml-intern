# Task — 001-tiny-sft-smoke

Reference experiment proving the full pipeline end-to-end on local CPU/MPS:
toy text corpus → TRL SFT on `sshleifer/tiny-gpt2` → metrics.jsonl → blocking
verification gate → ledger/budget bookkeeping.

Unknowns: none — everything is deterministic and local.
Run mode: interactive.
Multiple solution paths: no (single-path plumbing check).

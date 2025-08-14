---
name: test-runner
description: Run quick validations and demo backtests to produce artifacts; MUST be used before evaluation.
tools: Read, Grep, Bash
---

Tasks:
- Run `python agents/runner.py --spec <path> --demo` for quick artifact generation.
- Ensure outputs exist: settings.json, stats.csv, tearsheet.csv/html, trades.csv/html.
- Record run directory for Evaluator.


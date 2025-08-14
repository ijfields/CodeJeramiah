from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from rich import print as rprint

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents import researcher, architect, coder_lumibot, evaluator, fixer, deployer  # type: ignore


@dataclass
class RBIResult:
    slug: str
    spec_path: Path
    run_dir: Path | None
    passed: bool | None
    info: Dict[str, Any]


def to_class_name(slug: str) -> str:
    parts = [p for p in slug.replace("_", "-").split("-") if p]
    return "".join(s.capitalize() for s in parts) or "Strategy"


def ensure_spec(slug: str, name: str | None = None) -> Path:
    strat_dir = Path("strategies") / slug
    strat_dir.mkdir(parents=True, exist_ok=True)
    spec_path = strat_dir / "strategy_spec.json"
    if not spec_path.exists():
        spec = architect.build_spec_skeleton(name or slug.replace("_", " ").title(), slug)
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        rprint(f"[cyan]Created spec:[/cyan] {spec_path}")
    return spec_path


def ensure_strategy_stub(slug: str) -> Path:
    strat_dir = Path("strategies") / slug
    target = strat_dir / "lumibot_strategy.py"
    if not target.exists():
        class_name = to_class_name(slug)
        coder_lumibot.generate_strategy_file(target, class_name)
        rprint(f"[cyan]Created strategy stub:[/cyan] {target}")
    return target


def run_backtest(spec_path: Path, demo: bool, symbol: str | None) -> Path | None:
    cmd = [sys.executable, "agents/runner.py", "--spec", str(spec_path)]
    if demo:
        cmd.append("--demo")
    if symbol:
        cmd += ["--symbol", symbol]
    rprint(f"[magenta]Running:[/magenta] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        rprint(f"[red]Runner failed with exit {exc.returncode}[/red]")
        return None

    # Find latest run dir under strategies/<slug>/results
    results_dir = spec_path.parent / "results"
    if not results_dir.exists():
        return None
    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    return run_dirs[-1] if run_dirs else None


def evaluate_run(run_dir: Path) -> Dict[str, Any] | None:
    eval_path = run_dir / "evaluation.json"
    if eval_path.exists():
        try:
            return json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # No persisted evaluation; try to synthesize from stats if needed later
    return None


def rbi_flow(slug: str, demo: bool, symbol: str | None, topic: str | None) -> RBIResult:
    # Research (stub)
    if topic:
        research_notes = researcher.propose_strategy_inputs(topic)
        rprint(f"[blue]Research notes:[/blue] {research_notes}")

    # Architect (ensure spec)
    spec_path = ensure_spec(slug)

    # Coder (ensure strategy stub)
    ensure_strategy_stub(slug)

    # Backtest
    run_dir = run_backtest(spec_path, demo=demo, symbol=symbol)
    passed = None
    info: Dict[str, Any] = {}
    if run_dir is None:
        return RBIResult(slug=slug, spec_path=spec_path, run_dir=None, passed=None, info={"error": "runner"})

    # Evaluate (read result produced by runner demo)
    evaluation = evaluate_run(run_dir)
    if evaluation is not None:
        passed = bool(evaluation.get("passed"))
        info["evaluation"] = evaluation
        if not passed:
            failed = {k: v for k, v in evaluation.get("checks", {}).items() if not v.get("passed")}
            suggestions = fixer.suggest_changes(failed)
            info["suggestions"] = suggestions
            if suggestions:
                rprint("[yellow]Suggestions:[/yellow]"), [rprint(f" - {s}") for s in suggestions]
        else:
            msg = deployer.deploy(slug, environment="demo")
            info["deploy"] = msg
            rprint(f"[green]{msg}[/green]")

    # Index runs (optional best-effort)
    try:
        subprocess.run([sys.executable, "tools/index_runs.py"], check=False)
    except Exception:
        pass

    return RBIResult(slug=slug, spec_path=spec_path, run_dir=run_dir, passed=passed, info=info)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run RBI workflow (Research→Build→Implement)")
    ap.add_argument("--slug", default="sma_demo", help="Strategy slug (under strategies/<slug>)")
    ap.add_argument("--demo", action="store_true", help="Use runner demo mode (yfinance SMA)")
    ap.add_argument("--symbol", default=None, help="Optional override symbol for demo mode (e.g., SPY)")
    ap.add_argument("--topic", default=None, help="Optional research topic to seed notes")
    args = ap.parse_args(argv)

    res = rbi_flow(slug=args.slug, demo=args.demo, symbol=args.symbol, topic=args.topic)
    if res.run_dir:
        rprint(f"[green]RBI completed[/green] slug=[bold]{res.slug}[/bold] run_dir=[bold]{res.run_dir}[/bold]")
    else:
        rprint(f"[red]RBI failed before run completed[/red] slug=[bold]{res.slug}[/bold]")
    return 0 if (res.passed is None or res.passed is True) else 2


if __name__ == "__main__":
    raise SystemExit(main())



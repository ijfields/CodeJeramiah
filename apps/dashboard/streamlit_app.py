from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import sqlite3


st.set_page_config(page_title="Algo Platform Runs", layout="wide")
st.title("Algo Platform — Runs Dashboard")

index_path = Path("logs") / "runs_index.json"
if not index_path.exists():
    st.info("No runs indexed yet. Generate one by running: python tools/index_runs.py")
else:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    st.write(f"Total runs: {len(data)}")

    # Summary header from SQLite if available
    try:
        with sqlite3.connect("logs/runs.db") as conn:
            row = conn.execute(
                "SELECT COUNT(1), SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END) FROM runs"
            ).fetchone()
            if row:
                total, passed = row
                st.caption(f"Registry: {total} entries, {passed} passed")
    except Exception:
        pass
    for rec in data:
        run_dir = Path(rec["path"]) 
        # Surface pass/fail badge in the expander header if evaluation exists
        status_emoji = "🟡"
        eval_preview = run_dir / "evaluation.json"
        if eval_preview.exists():
            try:
                preview = json.loads(eval_preview.read_text(encoding="utf-8"))
                status_emoji = "✅" if preview.get("passed") else "❌"
            except Exception:
                status_emoji = "🟡"

        with st.expander(f"{rec['slug']} — {rec['run_id']} {status_emoji}"):
            run_dir = Path(rec["path"]) 
            # Show key artifacts if present
            # Show evaluation if available
            eval_path = run_dir / "evaluation.json"
            if eval_path.exists():
                st.subheader("Evaluation")
                import json as _json

                evaluation = _json.loads(eval_path.read_text(encoding="utf-8"))
                status = "PASSED" if evaluation.get("passed") else "FAILED"
                st.markdown(f"**Status**: {status}")
                st.json(evaluation.get("checks", {}))

                # Promotion controls
                promo_path = run_dir / "promotion.json"
                if evaluation.get("passed"):
                    if promo_path.exists():
                        st.success("Already promoted to demo")
                    else:
                        if st.button("Promote to demo", key=f"promote-{rec['slug']}-{rec['run_id']}"):
                            promo = {
                                "environment": "demo",
                                "promoted_at": __import__("datetime").datetime.now().isoformat(),
                            }
                            promo_path.write_text(_json.dumps(promo, indent=2) + "\n", encoding="utf-8")
                            st.success("Promotion recorded. Rerun to refresh.")

            for fname in [
                "settings.json",
                "stats.csv",
                "tearsheet.csv",
                "trades.csv",
                "tearsheet.html",
                "trades.html",
            ]:
                p = run_dir / fname
                if not p.exists():
                    continue
                if p.suffix == ".json":
                    st.code(p.read_text(encoding="utf-8"), language="json")
                elif p.suffix == ".csv":
                    st.download_button(
                        label=f"Download {fname}",
                        data=p.read_bytes(),
                        file_name=fname,
                        key=f"dl-{rec['slug']}-{rec['run_id']}-{fname}",
                    )
                elif p.suffix == ".html":
                    st.download_button(
                        label=f"Download {fname}",
                        data=p.read_bytes(),
                        file_name=fname,
                        key=f"dl-{rec['slug']}-{rec['run_id']}-{fname}",
                    )



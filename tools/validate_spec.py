from __future__ import annotations
import argparse, json
from pathlib import Path
from jsonschema import Draft202012Validator

def load_json(p: Path): 
    with p.open("r", encoding="utf-8-sig") as f:    # note utf-8-sig
        return json.load(f)


def validate(spec: Path, schema: Path) -> bool:
    v = Draft202012Validator(load_json(schema))
    data = load_json(spec)
    errs = sorted(v.iter_errors(data), key=lambda e: e.path)
    if errs:
        print(f"Invalid: {spec}")
        for e in errs:
            loc = "/".join([str(x) for x in e.path])
            print(f" - {loc}: {e.message}")
        return False
    print(f"OK: {spec}"); return True

def main()->int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--schema", default="strategies/spec.schema.json")
    a = ap.parse_args()
    t, s = Path(a.file), Path(a.schema)
    ok = True
    if t.is_dir():
        for p in t.rglob("strategy_spec.json"):
            ok = validate(p, s) and ok
    else:
        ok = validate(t, s)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate the plan/0008 K-sweep (mtp-champion-sweep/k*.json).

Per K: median mixed tok/s across restarts for base and mtp arms, the
per-K speedup, and the greedy-match tally. Emits one summary json.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    runs = {}
    for f in sorted(HERE.glob("k*-r*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"skip unreadable {f.name}")
            continue
        k = f.name.split("-")[0]
        runs.setdefault(k, []).append(rec)

    summary = {}
    for k, recs in sorted(runs.items()):
        entry = {"reps": len(recs), "greedy": [], "ctx512": {}, "ctx2048": {}}
        for ctx in ("512", "2048"):
            base = [r["mixed_tok_s_base"][ctx] for r in recs if ctx in r.get("mixed_tok_s_base", {})]
            mtp = [r["mixed_tok_s_mtp"][ctx] for r in recs if ctx in r.get("mixed_tok_s_mtp", {})]
            if base and mtp:
                mb, mm = statistics.median(base), statistics.median(mtp)
                entry["ctx512" if ctx == "512" else "ctx2048"] = {
                    "base_median": mb, "mtp_median": mm,
                    "speedup": round(mm / mb, 3),
                    "base_n": len(base), "mtp_n": len(mtp),
                }
        entry["greedy"] = [r.get("greedy_match") for r in recs]
        entry["canary_clean"] = all(
            r.get("canary_rep4_mtp") == r.get("canary_rep4_base") for r in recs
        )
        summary[k] = entry

    out = HERE / "summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())

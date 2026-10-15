"""Pack per-participant JSON records (outputs/main, outputs/ablation) into
compressed JSON-lines archives (outputs/<kind>_results.jsonl.gz)."""
import glob, gzip, json
from config import OUT_DIR

if __name__ == "__main__":
    for kind in ("main", "ablation"):
        files = sorted(glob.glob(str(OUT_DIR / kind / "*.json")))
        if not files:
            continue
        with gzip.open(OUT_DIR / f"{kind}_results.jsonl.gz", "wt") as fh:
            for f in files:
                fh.write(json.dumps(json.load(open(f))) + "\n")
        print(kind, len(files))

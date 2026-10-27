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
    import os
    n = 0
    with gzip.open(OUT_DIR / "bench_results.jsonl.gz", "wt") as fh:
        for d in sorted(glob.glob(str(OUT_DIR / "bench" / "*"))):
            tag = os.path.basename(d)
            if tag.startswith("kkt_"):
                continue
            for f in sorted(glob.glob(d + "/*.json")):
                r = json.load(open(f)); r["tag"] = tag
                fh.write(json.dumps(r) + "\n"); n += 1
    print("bench", n)
    n = 0
    with gzip.open(OUT_DIR / "sens_latent_results.jsonl.gz", "wt") as fh:
        for f in sorted(glob.glob(str(OUT_DIR / "sens_latent" / "*" / "[0-9]*.json"))):
            fh.write(json.dumps(json.load(open(f))) + "\n"); n += 1
    print("sens_latent", n)

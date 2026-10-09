"""Verify published archives and extract only final inference/diagnostic inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from lunarlander_rl.horizon_sensitivity import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", action="append", help="Only extract these variant names")
    parser.add_argument("--seed", type=int, action="append", help="Only extract these training seeds")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    publication = json.loads(args.publication.read_text())
    pending = [a for a in publication["archives"] if "-provenance." not in a["name"]]
    if args.condition:
        pending = [a for a in pending if any(a["name"].endswith(f"-{c}.tar.zst") for c in args.condition)]
        if len(pending) != len(set(args.condition)):
            raise ValueError("Requested condition not found in publication")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {"metadata.json", "metrics.jsonl", "checkpoint-1000000.pt", "summary.json",
                "completion_validation.json", "on_policy-1000000.npz", "common_probe-1000000.npz"}
    records = []
    deadline = time.monotonic() + args.wait_seconds
    while pending:
        for archive in pending[:]:
            path = args.archives / archive["name"]
            if not path.exists() or path.stat().st_size != archive["bytes"]:
                continue
            if sha256(path) != archive["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {path.name}")
            extracted = []
            process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
                    for member in tar:
                        parts = Path(member.name).parts
                        if not member.isfile() or len(parts) < 4 or parts[1] != "A" or parts[-1] not in selected:
                            continue
                        if args.seed and int(parts[2].rsplit("-seed", 1)[1]) not in args.seed:
                            continue
                        if Path(member.name).is_absolute() or ".." in parts:
                            raise ValueError("Unsafe archive path")
                        destination = args.output / member.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as source, destination.open("xb") as target:
                            shutil.copyfileobj(source, target)
                        extracted.append(dict(path=member.name, sha256=sha256(destination)))
            finally:
                process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError(f"Archive decompression failed: {path.name}")
            records.append(dict(**archive, files=extracted))
            pending.remove(archive)
            (args.output / "archive_verification.json").write_text(json.dumps(dict(
                publication=str(args.publication.resolve()), archives=records,
                pending=[a["name"] for a in pending]), indent=2) + "\n")
            print(f"Verified and extracted {path.name}: {len(extracted)} files", flush=True)
        if pending:
            if time.monotonic() >= deadline:
                raise FileNotFoundError("Incomplete archives: " + ", ".join(a["name"] for a in pending))
            time.sleep(5)
    print("Selected Stage A archives extracted and verified", flush=True)


if __name__ == "__main__":
    main()

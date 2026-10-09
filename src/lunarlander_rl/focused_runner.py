"""Frozen, resumable queue for the three exploratory DDPG interventions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from .experiment import Config
from .followup_runner import run_job, sha, timestamp, verify
from .paths import dependency_lock, project_root, snapshot_source
from .study_diagnostics import atomic_json, build_probes

CONDITIONS = (("reward-scale-0.1", 0.1, 1), ("actor-delay-2", 1.0, 2),
              ("actor-delay-3", 1.0, 3))


def jobs_for(preset="full"):
    historical = project_root() / "results/mac/stage-a-20261008/provenance/manifest.json"
    stage_a = json.loads(historical.read_text())
    reference = next(j for j in stage_a["jobs"]
                     if j["stage"] == "A" and j["cell"] == "ddpg-a0-c1")
    seeds = range(20, 30) if preset == "full" else range(9000, 9004) if preset == "benchmark" else (9000,)
    jobs = []
    for seed in seeds:
        for cell, scale, delay in CONDITIONS:
            cfg = dict(reference["config"], seed=seed, reward_scale=scale, ddpg_actor_delay=delay)
            if preset == "benchmark":
                cfg.update(max_steps=6000, learning_starts=1000, eval_interval=6000,
                           early_eval_interval=0, eval_episodes=2)
            elif preset == "smoke":
                cfg.update(max_steps=1100, learning_starts=100, batch_size=32,
                           replay_capacity=1200, actor_hidden=[16, 16], critic_hidden=[16, 16],
                           eval_interval=1100, early_eval_interval=0, eval_episodes=1,
                           train_horizon=100, diagnostic_horizon=300, resume_interval=250)
            Config(**cfg).validate()
            jobs.append(dict(stage="F", cell=cell, config=cfg,
                             directory=f"F/{cell}-seed{seed}"))
    return jobs


def prepare(out, workers, preset="full"):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = project_root()
    source = out / "source"
    source.mkdir()
    (out / "logs").mkdir()
    (out / "configs").mkdir()
    snapshot_source(source)
    for path in (dependency_lock(), root / "pyproject.toml",
                 root / "docs/FOCUSED_DDPG_EXPERIMENTS.md"):
        shutil.copy2(path, source / path.name)
    historical = root / "results/mac/stage-a-20261008"
    shutil.copy2(historical / "seed_summary.csv", out / "baseline_seed_summary.csv")
    shutil.copy2(historical / "learning_metrics.csv", out / "baseline_learning_metrics.csv")
    jobs = jobs_for(preset)
    for index, job in enumerate(jobs):
        job["config_path"] = f"configs/job-{index:03d}.json"
        atomic_json(out / job["config_path"], job["config"])
    cfg = Config(**jobs[0]["config"])
    bank = build_probes(cfg, 24 if preset == "full" else 4 if preset == "benchmark" else 2,
                        41000, 30 if preset == "full" else 5)
    atomic_json(out / "probes.json", bank)
    if preset == "full":
        previous = json.loads((historical / "provenance/probes.json").read_text())
        assert [p["seed"] for p in bank] == [p["seed"] for p in previous]
        assert [p["prefix"] for p in bank] == [p["prefix"] for p in previous]
        assert [p["action"] for p in bank] == [p["action"] for p in previous]
        atomic_json(out / "probe_comparison.json", dict(
            maximum_observation_difference=float(np.max(np.abs(
                np.asarray([p["obs"] for p in bank]) - np.asarray([p["obs"] for p in previous])))),
            note="Same seeds/actions, platform-native expected observations; exact replay remains enforced."))
    files = [*(p for p in source.rglob("*") if p.is_file()), out / "probes.json",
             out / "baseline_seed_summary.csv", out / "baseline_learning_metrics.csv",
             *(out / "configs").iterdir()]
    manifest = dict(created_at=timestamp(), preset=preset, workers=workers,
        interpreter=sys.executable, study="Exploratory DDPG interventions; reuse Stage A baseline",
        versions={name: importlib.metadata.version(name) for name in ("rl-mind", "torch", "gymnasium", "numpy", "Box2D")},
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        jobs=jobs, total_transitions=sum(j["config"]["max_steps"] for j in jobs),
        baseline_reused=True, sha256={str(p.relative_to(out)): sha(p) for p in files},
        units=dict(replay="raw", training_target="scale*r + gamma*Q_target",
                   evaluation="raw returns and Q/reward_scale; audit network outputs remain native"),
        target_schedule="DDPG actor delayed; both targets updated EVERY critic round")
    atomic_json(out / "manifest.json", manifest)
    (out / "manifest.sha256").write_text(sha(out / "manifest.json") + "\n")
    print(f"PREPARED {len(jobs)} runs / {manifest['total_transitions']:,} transitions at {out}", flush=True)
    return manifest


def export(out, manifest):
    rows = []
    for job in manifest["jobs"]:
        directory = out / job["directory"]
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        metric = json.loads((directory / "metrics.jsonl").read_text().splitlines()[-1])
        rows.append(dict(condition=job["cell"], seed=job["config"]["seed"],
            transitions=summary["transitions"], elapsed_seconds=summary["elapsed_seconds"],
            return_raw=metric["on_policy"]["return"], bias_raw=metric["on_policy"]["bias_q1"],
            rmse_raw=metric["on_policy"]["rmse_q1"], censored=metric["on_policy"]["censored"]))
    if rows:
        temp = out / "seed_summary.csv.tmp"
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temp, out / "seed_summary.csv")


def run(out, workers=None, min_free_gib=10):
    out = Path(out).resolve()
    lock_handle = (out / "runner.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify(out)
    workers = workers or manifest["workers"]
    stop, monitor_stop = threading.Event(), threading.Event()
    guard, status_guard = threading.Lock(), threading.Lock()
    processes = {}
    status = dict(status="running", started_at=timestamp(), runner_pid=os.getpid(),
                  workers=workers, total_jobs=len(manifest["jobs"]), jobs={})
    start = time.monotonic()
    def interrupt(signum, frame):
        stop.set()
        with guard:
            for process in processes.values():
                process.terminate()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    def observe():
        while not monitor_stop.is_set():
            with guard:
                active = {name: p.pid for name, p in processes.items()}
            with status_guard:
                status.update(active=active, updated_at=timestamp(), elapsed_seconds=time.monotonic()-start)
                atomic_json(out / "status.json", status)
            vm, swap = psutil.virtual_memory(), psutil.swap_memory()
            with (out / "machine_timing.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(at=timestamp(), active=len(active),
                    available_bytes=vm.available, swap_used_bytes=swap.used,
                    swap_out_bytes=swap.sout, cpu_percent=psutil.cpu_percent())) + "\n")
            monitor_stop.wait(30)
    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(run_job, out, job, stop, processes, guard, min_free_gib): job
                       for job in manifest["jobs"]}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(status="failed", error=repr(error))
                with status_guard:
                    status["jobs"][job["directory"]] = result
                    status.update(updated_at=timestamp(), elapsed_seconds=time.monotonic()-start)
                    atomic_json(out / "status.json", status)
                print(job["directory"], result["status"], flush=True)
                export(out, manifest)
    finally:
        monitor_stop.set(); monitor.join()
    complete = all(r["status"] == "complete" for r in status["jobs"].values()) and len(status["jobs"]) == len(manifest["jobs"])
    status.update(status="complete" if complete else "paused" if stop.is_set() else "requires_attention",
                  finished_at=timestamp(), elapsed_seconds=time.monotonic()-start, active={})
    atomic_json(out / "status.json", status)
    return 0 if complete else 1


def launch(out, label):
    import plistlib
    from .paths import frozen_command
    out = Path(out).resolve()
    verify(out)
    if sys.platform != "darwin" or not label.startswith("org.glouno.rl.") or "/" in label:
        raise ValueError("Expected macOS and an org.glouno.rl.* label")
    path = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    path.parent.mkdir(parents=True, exist_ok=True)
    plist = dict(Label=label, ProgramArguments=["/usr/bin/caffeinate", "-i", "-s",
        *frozen_command(out / "source", "focused_runner"), "--output", str(out)],
        WorkingDirectory=str(out / "source"), RunAtLoad=False, KeepAlive=False, ProcessType="Standard",
        StandardOutPath=str(out / "supervisor.log"), StandardErrorPath=str(out / "supervisor-error.log"),
        EnvironmentVariables=dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                  NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg"))
    with path.open("xb") as handle:
        plistlib.dump(plist, handle)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["/bin/launchctl", "bootstrap", domain, str(path)], check=True)
    subprocess.run(["/bin/launchctl", "kickstart", f"{domain}/{label}"], check=True)
    atomic_json(out / "launch.json", dict(label=label, plist=str(path), launched_at=timestamp()))
    print("LAUNCHED", label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--preset", choices=("full", "benchmark", "smoke"), default="full")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("Workers must be positive")
    if args.prepare:
        prepare(args.output, args.workers or 4, args.preset)
    elif args.launch:
        launch(args.output, args.label or "org.glouno.rl.ddpg-focused")
    else:
        sys.exit(run(args.output, args.workers))


if __name__ == "__main__":
    main()

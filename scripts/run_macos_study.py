"""Supervise a frozen Mac queue, prevent sleep, and checkpoint at a time budget.

launchd owns this process; the scientific runner and its source are unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, record):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    if sys.platform != 'darwin' or args.hours <= 0:
        parser.error('macOS and a positive time budget are required')
    import psutil
    out = args.output.resolve()
    # Mirroring immutable inputs omits the original empty runtime directory.
    (out/'logs').mkdir(exist_ok=True)
    # Import/verify only the scientific code frozen in this queue.
    verification = subprocess.run([sys.executable, '-c',
        'from pathlib import Path; from lunarlander_rl.followup_runner import verify; '
        'import sys; verify(Path(sys.argv[1]))', str(out)], cwd=out/'source')
    if verification.returncode:
        return verification.returncode
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg',
        PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='')
    requested_stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: requested_stop.append(number))
    started = time.monotonic()
    deadline = time.time()+args.hours*3600
    runner = subprocess.Popen([sys.executable, '-m', 'lunarlander_rl.followup_runner',
        '--output', str(out), '--workers', '8'], cwd=out/'source', env=env)
    guard = None
    record = dict(started_at=now(), supervisor_pid=os.getpid(), runner_pid=runner.pid,
        interpreter=sys.executable, platform=platform.platform(), machine=platform.machine(),
        hours=args.hours, workers=8, output=str(out),
        manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stop_requested_deadline=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        status='running')
    reason = None
    try:
        guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(runner.pid)])
        record['caffeinate_pid'] = guard.pid
        write_state(out/'supervisor.json', record)
        print(json.dumps(record), flush=True)
        psutil.cpu_percent()
        next_sample = started
        with (out/'machine_timing.jsonl').open('a') as journal:
            while runner.poll() is None:
                elapsed = time.monotonic() - started
                if reason is None and (requested_stop or elapsed >= args.hours*3600 or time.time() >= deadline):
                    reason = 'requested_stop' if requested_stop else 'time_budget'
                    record.update(status='checkpointing', stop_reason=reason, stop_requested_at=now())
                    write_state(out/'supervisor.json', record)
                    # The runner forwards SIGTERM to workers; workers save resumable state.
                    runner.terminate()
                    print(f'{now()} requesting graceful checkpoint: {reason}', flush=True)
                if guard.poll() is not None and runner.poll() is None:
                    raise RuntimeError('Sleep-prevention process exited while training was active')
                if time.monotonic() >= next_sample:
                    vm = psutil.virtual_memory(); swap = psutil.swap_memory()
                    sample = dict(timestamp=now(), session_elapsed_seconds=elapsed,
                        host_cpu_percent=psutil.cpu_percent(), available_bytes=vm.available,
                        swap_used_bytes=swap.used, load_average=os.getloadavg())
                    journal.write(json.dumps(sample)+'\n'); journal.flush(); os.fsync(journal.fileno())
                    next_sample = time.monotonic()+60
                time.sleep(1)
        code = runner.wait()
        record.update(status='paused' if reason else 'complete' if code == 0 else 'requires_attention',
            stop_reason=reason, runner_exit_code=code, finished_at=now(),
            session_elapsed_seconds=time.monotonic()-started)
        write_state(out/'supervisor.json', record)
        print(json.dumps(record), flush=True)
        return 0 if reason or code == 0 else code
    except BaseException as error:
        record.update(status='supervisor_error', error=str(error), updated_at=now())
        write_state(out/'supervisor.json', record)
        raise
    finally:
        if runner.poll() is None:
            runner.terminate(); runner.wait()
        if guard is not None and guard.poll() is None:
            guard.terminate(); guard.wait()


if __name__ == '__main__':
    sys.exit(main())

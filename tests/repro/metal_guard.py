"""macOS test child watchdog; no inference math. All limits are explicit CLI options."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--process-gib', type=float, required=True)
    ap.add_argument('--wired-gib', type=float, required=True)
    ap.add_argument('--swap-mib', type=float, required=True)
    ap.add_argument('--compressed-gib', type=float, required=True)
    ap.add_argument('--warning-seconds', type=float, default=1800)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        ap.error('child command required')
    a.out.mkdir(parents=True, exist_ok=False)
    def sample(pid=0):
        r = subprocess.run([str(a.probe.resolve()), str(pid)], capture_output=True, check=True)
        return json.loads(r.stdout)
    def save(name, value):
        (a.out/name).write_text(json.dumps(value, indent=2)+'\n')
    baseline = sample()
    if baseline['pressure'] != 1:
        raise RuntimeError('memory pressure is not normal; child not started')
    peak = {}; warning = None; started = time.monotonic(); reason = None
    def cancel(sig, frame):
        raise InterruptedError('stop requested')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    with (a.out/'stdout.log').open('w') as log, (a.out/'memory.jsonl').open('w') as mem:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        save('state.json', dict(pid=child.pid, command=cmd, baseline=baseline,
             limits=vars(a) | {'probe':str(a.probe), 'out':str(a.out)}))
        try:
            while child.poll() is None:
                r = sample(child.pid)
                if not r['process_ok']:
                    if child.poll() is not None: break
                    raise RuntimeError('child memory probe failed')
                mem.write(json.dumps(r)+'\n'); mem.flush()
                for k in ['rss','footprint','wired','compressed','swap_used','swapouts','pressure']:
                    peak[k] = max(peak.get(k,0), r[k])
                now = time.monotonic()
                warning = (warning or now) if r['pressure'] >= 2 else None
                failures = []
                if max(r['rss'], r['footprint']) > a.process_gib * 2**30: failures.append('process memory')
                if r['wired'] > a.wired_gib * 2**30: failures.append('system wired')
                if r['pressure'] >= 4: failures.append('critical pressure')
                if warning and now-warning >= a.warning_seconds: failures.append('sustained warning pressure')
                if max(r['swap_used']-baseline['swap_used'],r['swapouts']-baseline['swapouts']) > a.swap_mib*2**20: failures.append('swap growth')
                if r['compressed']-baseline['compressed'] > a.compressed_gib*2**30: failures.append('compressor growth')
                if now-started > a.timeout: failures.append('timeout')
                if failures: raise RuntimeError(', '.join(failures))
                time.sleep(.5)
        except BaseException as ex:
            reason = f'{type(ex).__name__}: {ex}'
        finally:
            if child.poll() is None:
                child.terminate() if isinstance(reason,str) and reason.startswith('InterruptedError') else child.kill()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            result = dict(exit=child.returncode, reason=reason, baseline=baseline,
                          peak=peak, elapsed_seconds=time.monotonic()-started)
            save('summary.json', result)
            print(json.dumps(result), flush=True)
    return 1 if reason else child.returncode

if __name__ == '__main__':
    raise SystemExit(main())

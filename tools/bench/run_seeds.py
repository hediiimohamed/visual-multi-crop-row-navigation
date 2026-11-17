#!/usr/bin/env python3
"""Runs tools/bench/bench.py over many seeds in parallel and summarises.

usage:
  python3 tools/bench/run_seeds.py [--tracker T.py] [--seeds 7:37] [--jobs 12] KEY=VAL ...   one configuration
  python3 tools/bench/run_seeds.py [--tracker T.py] --set core|stress|all                  a regression set
One line per configuration: FAILED (robot stopped), off lane runs (robot more than half a furrow from the true lane
centre: the dangerous outcome), losses, worst error, runs with wheel contact, the miscounted lane rows (LANEROWS=0).

Reference (commit 5ad5794, seeds 7..36): core set: weeds2 0 FAILED 0 off lane, beds3 / seed3 / straight perfect,
seed3 + weeds (auto) 5 FAILED 0 off lane; stress: weeds2 WF=0.6 9 FAILED 4 off lane, WF=0.8 18 FAILED 10 off lane."""
import argparse, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    'core': ['FIELD=weeds2', 'FIELD=weeds2 LANEROWS=4', 'FIELD=beds3', 'FIELD=seed3 NF=10',
             'FIELD=beds3 WF=0 WR=0 WB=0', 'FIELD=seed3 NF=10 WD=3', 'FIELD=seed3 NF=10 WD=3 LANEROWS=3'],
    'stress': ['FIELD=weeds2 WF=0.6', 'FIELD=weeds2 WF=0.8', 'FIELD=weeds2 Y0=0.10 YAW0=5',
               'FIELD=beds3 Y0=0.12 YAW0=-8', 'FIELD=weeds2 WD=8 WP=3', 'FIELD=weeds2 GAPS=0.3'],
}
SETS['all'] = SETS['core'] + SETS['stress']


def runOne(tracker, seed, env):
    e = dict(os.environ, **env)
    cmd = [sys.executable, '-I', os.path.join(HERE, 'bench.py')] + ([tracker] if tracker else []) + [str(seed)]
    out = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=3600).stdout
    return seed, out


def summary(results):
    failed, off, losses, worst, contact, miscount, n = [], [], 0, 0.0, 0, 0, 0
    for seed, out in results:
        m = re.search(r'ROWSFOUND (\d+) of (\d+) \| (FAILED )?seed \d+: .*losses (\d+), frames off lane \([^)]*\) (\d+), '
                      r'err mean [\d.]+ p95 [\d.]+ max ([\d.]+) cm', out)
        if not m:
            continue
        n += 1
        if m.group(3):
            failed.append(seed)
        if int(m.group(5)) > 0:
            off.append((seed, float(m.group(6))))
        losses += int(m.group(4))
        worst = max(worst, float(m.group(6)))
        if m.group(1) not in ('0', m.group(2)):
            miscount += 1
        w = re.search(r'contact frames \[([^\]]*)\]', out)
        if w and max(int(v) for v in w.group(1).split(',')) > 0:
            contact += 1
    return (f'{n} runs: FAILED {len(failed)} {failed}, off lane {len(off)} {off}, losses {losses}, '
            f'worst {worst:.1f} cm, wheel contact {contact}/{n}, rows miscounted {miscount}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracker', default='', help='groundTracker.py to test (default: the repository one)')
    ap.add_argument('--seeds', default='7:37', help='first:last+1 (default 7:37, 30 seeds)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument('--set', choices=sorted(SETS), help='run a regression set instead of one configuration')
    ap.add_argument('config', nargs='*', help='KEY=VAL settings of bench.py (FIELD=weeds2 LANEROWS=4 ...)')
    a = ap.parse_args()
    s0, s1 = (int(v) for v in a.seeds.split(':'))
    configs = SETS[a.set] if a.set else [' '.join(a.config)]
    for cfg in configs:
        env = dict(kv.split('=', 1) for kv in cfg.split())
        with ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(lambda s: runOne(a.tracker, s, env), range(s0, s1)))
        print(f'{cfg or "(defaults)":<40} | {summary(res)}', flush=True)


if __name__ == '__main__':
    main()

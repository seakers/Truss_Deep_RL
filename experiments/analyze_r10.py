"""Round 10 analysis: the delta-HV oracle, and the spread diagnostic on round 9's rules.

Recomputes everything from the raw r10d*_shard_*.jsonl records. Paired t-tests on shared
seeds, because every arm runs the same seed set by construction.
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    rows = []
    for f in glob.glob(os.path.join(HERE, "data", f"r10d{tag}_shard_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_arm(rows, key='hv'):
    d = {}
    for r in rows:
        d.setdefault(r['name'], {})[r['seed']] = r[key]
    return d


def paired(arm, ctl):
    seeds = sorted(set(arm) & set(ctl))
    a = np.array([arm[s] for s in seeds]); c = np.array([ctl[s] for s in seeds])
    delta = a - c
    if len(seeds) < 2:
        return delta.mean(), int((delta > 0).sum()), len(seeds), float('nan')
    t, p = stats.ttest_rel(a, c)
    return delta.mean(), int((delta > 0).sum()), len(seeds), p


def report(tag, control, title):
    rows = load(tag)
    if not rows:
        print(f"\n{title}: no records\n"); return
    hv = by_arm(rows, 'hv')
    nf = by_arm(rows, 'n_front')
    sp = by_arm(rows, 'spread_volume')
    nfe = by_arm(rows, 'nfe')
    info = by_arm(rows, 'informative_frac')

    print(f"\n{'='*104}\n{title}   ({len(rows)} records)\n{'='*104}")
    print(f"{'arm':<16}{'HV':>9}{'sd':>8}{'d vs ctl':>11}{'win':>7}{'p':>11}"
          f"{'front':>8}{'spread':>11}{'NFE':>9}{'info':>8}")
    order = [control] + [a for a in hv if a != control]
    for a in order:
        if a not in hv:
            continue
        seeds = sorted(hv[a])
        v = np.array([hv[a][s] for s in seeds])
        line = f"{a:<16}{v.mean():>9.4f}{v.std(ddof=1) if len(v)>1 else 0:>8.4f}"
        if a == control:
            line += f"{'--':>11}{'--':>7}{'--':>11}"
        else:
            dm, w, n, p = paired(hv[a], hv[control])
            line += f"{dm:>+11.4f}{f'{w}/{n}':>7}{p:>11.4g}"
        f_ = np.mean([nf[a][s] for s in seeds])
        s_ = np.mean([sp[a][s] for s in seeds])
        e_ = np.mean([nfe[a][s] for s in seeds])
        i_ = [info[a][s] for s in seeds if info.get(a, {}).get(s) is not None]
        line += f"{f_:>8.0f}{s_:>11.4g}{e_:>9.0f}"
        line += f"{np.mean(i_):>8.3f}" if i_ else f"{'--':>8}"
        print(line)


if __name__ == '__main__':
    report('1', 'random',
           "SET 1 -- delta-HV oracle vs blind mutation, 64,000 NFE, paired seeds")
    report('2', 'random_solve',
           "SET 2 -- round 9 physics rules, with the spread diagnostic round 9 never stored")

    print("\nColumns: front = points on the final Pareto front. spread = product of the "
          "front's per-objective\nranges (a collapsed front has a small one). info = "
          "fraction of mutations where at least one\ncandidate advanced the archive, so "
          "the oracle had any signal to act on at all.\n")

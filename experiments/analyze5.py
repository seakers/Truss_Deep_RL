"""Round 4 analysis. Reads every shard glob directly rather than a hardcoded arm list,
which is what made analyze4.py silently drop the round-3 winner.

Paired t-tests throughout: seeds are shared by construction across arms, and round 2's
Welch tests threw that blocking away.
"""
import json, glob, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pat):
    rows = []
    for fp in sorted(glob.glob(os.path.join(HERE, "data", pat))):
        for line in open(fp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'error' in r:
                # the CUDA OOM rows are the abandoned GPU launch of track B, superseded
                # by the CPU rerun; anything else is worth seeing
                if 'out of memory' not in r['error']:
                    print("  ERROR row:", r.get('name'), r.get('seed'), r['error'][:100])
                continue
            rows.append(r)
    return rows


def by_arm(rows, key='name', unit='seed'):
    d = {}
    for r in rows:
        d.setdefault(r[key], {})[r[unit]] = r
    return d


def paired(a, b):
    """a, b: dict unit -> hv. Paired t-test over shared units."""
    u = sorted(set(a) & set(b))
    if len(u) < 3:
        return None
    x = np.array([a[k] for k in u]); y = np.array([b[k] for k in u])
    t, p = stats.ttest_rel(x, y)
    return dict(n=len(u), delta=float((x - y).mean()), p=float(p),
                wins=int((x > y).sum()))


def table(arms, ref, order=None, extra=()):
    names = order or sorted(arms, key=lambda n: np.mean([r['hv'] for r in arms[n].values()]))
    hdr = f"{'arm':<32} {'n':>3} {'mean HV':>8} {'sd':>7} {'min':>7} {'max':>7}"
    for e in extra:
        hdr += f" {e[:8]:>8}"
    hdr += f" {'vs ref':>8} {'p':>8} {'win':>5}"
    print(hdr)
    print("-" * len(hdr))
    refhv = {s: r['hv'] for s, r in arms[ref].items()} if ref in arms else None
    for n in names:
        hv = np.array([r['hv'] for r in arms[n].values()])
        line = f"{n:<32} {len(hv):>3} {hv.mean():>8.4f} {hv.std(ddof=1):>7.4f} {hv.min():>7.4f} {hv.max():>7.4f}"
        for e in extra:
            v = [r.get(e) for r in arms[n].values() if r.get(e) is not None]
            line += f" {np.mean(v):>8.3g}" if v else f" {'-':>8}"
        if refhv is not None and n != ref:
            pr = paired({s: r['hv'] for s, r in arms[n].items()}, refhv)
            if pr:
                line += f" {pr['delta']:>+8.4f} {pr['p']:>8.4f} {pr['wins']:>2}/{pr['n']:<2}"
        print(line)


# ---------------------------------------------------------------- track A
print("=" * 100)
print("TRACK A -- synthesis PPO: what the batch-local HVI archive buys (64,000 NFE)")
print("=" * 100)

rows = load("r3_shard_*.jsonl") + load("r4a_shard_*.jsonl")
# the round-4 extension arms are the same configuration as their round-3 originals,
# just on seeds 8-15; merge them so the confirmation is one arm at n=16
RENAME = {"batch archive (seeds 8-15)": "fixed + hv, batch archive (reset=1)",
          "fixed + hv, batch archive":  "fixed + hv, batch archive (reset=1)",
          "baseline (seeds 8-15)":      "baseline (repo as-was)"}
for r in rows:
    r['name'] = RENAME.get(r['name'], r['name'])
A = by_arm(rows)
for n in A:
    print(f"  {n:<40} seeds {sorted(A[n])}")
print()
table(A, "baseline (repo as-was)",
      order=sorted(A, key=lambda n: np.mean([r['hv'] for r in A[n].values()])),
      extra=("valid_frac", "entropy_final", "ev_median"))

print("\nagainst the round-3 winner (reset=1):")
W = {s: r['hv'] for s, r in A["fixed + hv, batch archive (reset=1)"].items()}
for n in sorted(A):
    if n == "fixed + hv, batch archive (reset=1)":
        continue
    pr = paired({s: r['hv'] for s, r in A[n].items()}, W)
    if pr:
        print(f"  {n:<40} delta {pr['delta']:>+8.4f}  p={pr['p']:.4f}  n={pr['n']}")

print("\nper-seed, winner vs baseline:")
for n in ["baseline (repo as-was)", "fixed + hv, batch archive (reset=1)"]:
    if n in A:
        ss = sorted(A[n])
        print(f"  {n:<40} " + " ".join(f"{A[n][s]['hv']:.3f}" for s in ss))

print("\ncritic explained variance, last 50 epochs (mean / median over seeds):")
for n in sorted(A):
    ev = [r['ev'] for r in A[n].values() if 'ev' in r]
    evm = [r.get('ev_median', r['ev']) for r in A[n].values() if 'ev' in r]
    if ev:
        print(f"  {n:<40} mean {np.mean(ev):>12.4g}   median-of-medians {np.median(evm):>10.4g}")

# where do the critic blow-ups start?
print("\nseeds whose critic EV falls below -1e3, and the epoch it first happens:")
for n in sorted(A):
    for s, r in sorted(A[n].items()):
        c = r.get('ev_curve')
        if not c:
            continue
        bad = [i for i, v in enumerate(c) if v < -1e3]
        if bad:
            print(f"  {n:<40} seed {s:<3} first at epoch ~{bad[0]*10:<5} "
                  f"final ev {r['ev']:.3g}  hv {r['hv']:.4f}")

# ---------------------------------------------------------------- track B
print()
print("=" * 100)
print("TRACK B -- is the hybrid's learned operator worth anything?")
print("=" * 100)
rowsb = load("r4b_shard_*.jsonl")
tr = [r for r in rowsb if r['kind'] == 'train']
ga64 = [r for r in rowsb if r['kind'] == 'ga64']
imga = [r for r in rowsb if r['kind'] == 'imga']

if tr:
    print("\nrepair-actor training runs (32,000 NFE each, standalone HV):")
    d = {}
    for r in tr:
        d.setdefault(r['tag'], []).append(r)
    for t in sorted(d):
        hv = np.array([x['hv'] for x in d[t]])
        rf = np.mean([x['reward_first'] for x in d[t]])
        rl = np.mean([x['reward_last'] for x in d[t]])
        print(f"  {t:<12} n={len(hv)}  HV {hv.mean():.4f} +- {hv.std(ddof=1):.4f}   "
              f"episode reward {rf:>9.3f} -> {rl:>9.3f}")

if ga64:
    print("\nhybrid arms at the synthesis budget (64,000 NFE):")
    B = by_arm(ga64, 'name', 'seed')
    table(B, None, order=sorted(B, key=lambda n: -np.mean([r['hv'] for r in B[n].values()])))
    a3 = load("r3_shard_*.jsonl") + load("r4a_shard_*.jsonl")
    for r in a3:
        r['name'] = RENAME.get(r['name'], r['name'])
    AA = by_arm(a3)
    for ref in ["fixed + hv, batch archive (reset=1)", "random search"]:
        if ref in AA:
            hv = np.array([r['hv'] for r in AA[ref].values()])
            print(f"  (for reference, same budget) {ref:<40} {hv.mean():.4f} "
                  f"+- {hv.std(ddof=1):.4f}  n={len(hv)}")

if imga:
    print("\nIMGA at 16,000 NFE by mutation operator "
          "(6 independently trained actors x 2 GA seeds; paired on GA seed k):")
    C = by_arm(imga, 'name', 'k')
    table(C, "random init", order=sorted(C, key=lambda n: -np.mean([r['hv'] for r in C[n].values()])))

    print("\n  per-actor means (the unit the training-vs-random test should use):")
    for n in sorted(C):
        if n == 'random init':
            continue
        pa = {}
        for k, r in C[n].items():
            pa.setdefault(r['tseed'], []).append(r['hv'])
        m = {t: float(np.mean(v)) for t, v in sorted(pa.items())}
        print(f"    {n:<12} " + " ".join(f"t{t}={v:.4f}" for t, v in m.items()))

    ri = {k: r['hv'] for k, r in C['random init'].items()}
    print("\n  actor-level test: each trained actor's mean HV vs the random-init mean "
          "over the same GA seeds")
    for n in sorted(C):
        if n == 'random init':
            continue
        pa, pr_ = {}, {}
        for k, r in C[n].items():
            pa.setdefault(r['tseed'], []).append(r['hv'])
            pr_.setdefault(r['tseed'], []).append(ri.get(k, np.nan))
        ts = sorted(pa)
        x = np.array([np.mean(pa[t]) for t in ts])
        y = np.array([np.nanmean(pr_[t]) for t in ts])
        if len(ts) >= 3:
            t_, p = stats.ttest_rel(x, y)
            print(f"    {n:<12} n_actors={len(ts)}  delta {x.mean()-y.mean():>+8.4f}  "
                  f"p={p:.4f}  actors beating random init: {(x>y).sum()}/{len(ts)}")

    print("\n  the observation-mismatch fix, holding the reward fixed:")
    for a, b in [("DOM_DES", "DOM_FULL"), ("HVG_DES", "HVG_FULL")]:
        if a in C and b in C:
            pr = paired({k: r['hv'] for k, r in C[a].items()},
                        {k: r['hv'] for k, r in C[b].items()})
            if pr:
                print(f"    {a} - {b}: {pr['delta']:>+8.4f}  p={pr['p']:.4f}  n={pr['n']}")
    print("\n  the reward, holding the observation fix on:")
    for a, b in [("HVG_DES", "DOM_DES"), ("HVB_DES", "DOM_DES"), ("HVB_DES", "HVG_DES")]:
        if a in C and b in C:
            pr = paired({k: r['hv'] for k, r in C[a].items()},
                        {k: r['hv'] for k, r in C[b].items()})
            if pr:
                print(f"    {a} - {b}: {pr['delta']:>+8.4f}  p={pr['p']:.4f}  n={pr['n']}")

# ---------------------------------------------------------------- track D
print()
print("=" * 100)
print("TRACK D -- where is the archive-reset optimum? (64,000 NFE)")
print("=" * 100)
rowsd = load("r4d_shard_*.jsonl")
if rowsd:
    RN = {"reset every 5 (seeds 8-15)": "reset every 5"}
    for r in rowsd:
        r['name'] = RN.get(r['name'], r['name'])
    # merge with the track A reset arms so the curve is one table
    ra = load("r3_shard_*.jsonl") + load("r4a_shard_*.jsonl")
    for r in ra:
        r['name'] = RENAME.get(r['name'], r['name'])
    CURVE = {"fixed + hv reward": 0, "fixed + hv, batch archive (reset=1)": 1,
             "reset every 2": 2, "reset every 3": 3, "reset every 5": 5,
             "reset every 10": 10, "reset every 25": 25}
    D = by_arm([r for r in rowsd + ra if r['name'] in CURVE])
    print(f"{'reset period':>13} {'n':>3} {'mean HV':>8} {'sd':>7} {'min':>7} {'max':>7} "
          f"{'valid%':>7} {'entropy':>8} {'critic EV':>11}")
    print("-" * 78)
    for n in sorted(D, key=lambda n: CURVE[n]):
        hv = np.array([r['hv'] for r in D[n].values()])
        vf = np.mean([r['valid_frac'] for r in D[n].values()])
        en = np.mean([r['entropy_final'] for r in D[n].values()])
        ev = [r.get('ev_median') for r in D[n].values() if r.get('ev_median') is not None]
        lbl = "never" if CURVE[n] == 0 else str(CURVE[n])
        print(f"{lbl:>13} {len(hv):>3} {hv.mean():>8.4f} {hv.std(ddof=1):>7.4f} "
              f"{hv.min():>7.4f} {hv.max():>7.4f} {vf:>7.3f} {en:>8.2f} "
              f"{(np.median(ev) if ev else float('nan')):>11.3g}")
    best = max(D, key=lambda n: np.mean([r['hv'] for r in D[n].values()]))
    print(f"\n  best arm: {best}")
    B = {s: r['hv'] for s, r in D[best].items()}
    for n in sorted(D, key=lambda n: CURVE[n]):
        if n == best:
            continue
        pr = paired({s: r['hv'] for s, r in D[n].items()}, B)
        if pr:
            print(f"    vs reset={CURVE[n] or 'never':<5} delta {-pr['delta']:>+8.4f}  "
                  f"p={pr['p']:.4f}  n={pr['n']}")

# ---------------------------------------------------------------- track E
print()
print("=" * 100)
print("TRACK E -- does the learned operator keep its edge at 64,000 NFE?")
print("=" * 100)
rowse = load("r4e_shard_*.jsonl")
b64 = [r for r in load("r4b_shard_*.jsonl") if r.get('kind') == 'ga64']
if rowse or b64:
    # track B's phase-3 random-init runs used GA seed 700+seed, matching track E's k
    # track B's ga64 records carry k=None and identify the run by 'seed'; track E's
    # carry k. Collapsing on the None would merge all six into one row.
    pool = [dict(r, k=(r['k'] if r.get('k') is not None else r.get('seed')))
            for r in rowse + b64]
    E = {}
    for r in pool:
        if r['kind'] in ('imga64', 'ga64'):
            # 'IMGA untrained' (track B phase 3) and 'random init' (track E) are the
            # same arm under two names; merge so the paired test can find both
            nm = r['name'].replace(' @64k', '').replace('IMGA untrained', 'random init')
            E.setdefault(nm, {})[r['k']] = r
    print("at 64,000 NFE:")
    for n in sorted(E, key=lambda n: -np.mean([r['hv'] for r in E[n].values()])):
        hv = np.array([r['hv'] for r in E[n].values()])
        print(f"  {n:<22} n={len(hv):<3} {hv.mean():.4f} +- {hv.std(ddof=1):.4f}  "
              f"[{hv.min():.4f}, {hv.max():.4f}]")
    if 'DOM_DES' in E and 'random init' in E:
        pr = paired({k: r['hv'] for k, r in E['DOM_DES'].items()},
                    {k: r['hv'] for k, r in E['random init'].items()})
        if pr:
            print(f"\n  trained vs untrained operator @64k: {pr['delta']:>+.4f}  "
                  f"p={pr['p']:.4f}  n={pr['n']}  wins {pr['wins']}/{pr['n']}")
    e16 = [r for r in rowse if r['kind'] == 'imga']
    if e16:
        C = {}
        for r in load("r4b_shard_*.jsonl") + e16:
            if r.get('kind') == 'imga':
                C.setdefault(r['name'], {})[r['k']] = r
        print("\nat 16,000 NFE, DOM_DES vs random init pooled over both tracks "
              "(10 actors x 2 GA seeds):")
        for n in ['DOM_DES', 'random init']:
            if n in C:
                hv = np.array([r['hv'] for r in C[n].values()])
                print(f"  {n:<14} n={len(hv):<3} {hv.mean():.4f} +- {hv.std(ddof=1):.4f}")
        if 'DOM_DES' in C and 'random init' in C:
            pr = paired({k: r['hv'] for k, r in C['DOM_DES'].items()},
                        {k: r['hv'] for k, r in C['random init'].items()})
            if pr:
                print(f"  delta {pr['delta']:>+.4f}  p={pr['p']:.4f}  n={pr['n']}  "
                      f"wins {pr['wins']}/{pr['n']}")
            pa, pr_ = {}, {}
            for k, r in C['DOM_DES'].items():
                pa.setdefault(r['tseed'], []).append(r['hv'])
                pr_.setdefault(r['tseed'], []).append(C['random init'][k]['hv']
                                                     if k in C['random init'] else np.nan)
            ts = sorted(pa)
            x = np.array([np.mean(pa[t]) for t in ts])
            y = np.array([np.nanmean(pr_[t]) for t in ts])
            t_, p = stats.ttest_rel(x, y)
            print(f"  actor-level: n_actors={len(ts)}  delta {x.mean()-y.mean():>+.4f}  "
                  f"p={p:.4f}  actors beating random init: {(x>y).sum()}/{len(ts)}")

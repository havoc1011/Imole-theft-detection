"""Train and evaluate Ìmọ́lẹ̀ on simulated transformer communities.
Usage: python run_experiment.py <output dir> [n_train] [n_test]"""
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import os; _h = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_h, os.path.join(_h, "..", "src")]
from imole_sim import simulate_dt, THEFT_TYPES
from imole_detect import analyse_dt, FEATURE_SETS, plan_inspection, field_check

out = sys.argv[1]
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 160
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 60
K = 5  # inspections per transformer per round

rows, dts, results = [], {}, {}
for seed in range(N_TRAIN + N_TEST):
    clean = seed % 5 in (0, 1)   # 40% of transformers have no theft at all
    dt = simulate_dt(seed, days=90, theft_rate=0 if clean else 0.08, p_illegal=0 if clean else 0.5)
    res = analyse_dt(dt, rng=np.random.default_rng(10_000 + seed))
    tr = dt['truth']
    for i in range(len(dt['phase'])):
        rows.append(dict(dt=seed, clean=int(clean), cust=i, y=int(tr['role'][i] == 'thief'), role=tr['role'][i],
                         ttype=tr['theft_type'][i], long_term=int(tr['role'][i] == 'thief' and tr['change_day'][i] == 0),
                         **dict(zip(res['names'], res['X'][i]))))
    if seed >= N_TRAIN:
        dts[seed], results[seed] = dt, res
df = pd.DataFrame(rows)
train, test = df[df.dt < N_TRAIN], df[df.dt >= N_TRAIN]

summary = dict(n_train_dt=N_TRAIN, n_test_dt=N_TEST, customers_per_dt=90,
               test_customers=int(len(test)), test_thieves=int(test.y.sum()),
               test_solar=int((test.role == 'solar').sum()), test_vacant=int((test.role == 'vacant').sum()))
variant_rows, scores = [], {}
for name, cols in FEATURE_SETS.items():
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=1)
    m.fit(train[cols], train.y)
    p = m.predict_proba(test[cols])[:, 1]
    t = test.assign(p=p); scores[name] = t
    found = flagged_honest_low = total_thieves = 0
    for _, g in t.groupby('dt'):
        top = g.nlargest(K, 'p')
        found += top.y.sum(); total_thieves += g.y.sum()
        flagged_honest_low += top.role.isin(['solar', 'vacant']).sum()
    n_dt = t.dt.nunique()
    variant_rows.append(dict(variant=name, auc=roc_auc_score(t.y, p),
                             thieves_found_top5_pct=100 * found / total_thieves,
                             hit_rate_top5_pct=100 * found / (K * n_dt),
                             honest_low_users_flagged_per_100_inspections=100 * flagged_honest_low / (K * n_dt)))
    # Global budget: DisCo inspects the riskiest 3% of ALL customers across all transformers
    nb = int(0.03 * len(t)); top = t.nlargest(nb, 'p')
    variant_rows[-1].update(budget_visits=nb, budget_hit_rate_pct=100 * top.y.mean(),
                            budget_thieves_found_pct=100 * top.y.sum() / t.y.sum(),
                            budget_visits_to_clean_dts_pct=100 * top.clean.mean())
    t.to_csv(f'{out}/pred_{name}.csv', index=False)
var = pd.DataFrame(variant_rows); var.to_csv(f'{out}/variants.csv', index=False)
print(var.round(3).to_string())

# Recall by theft type (best variant), thieves found within top 5 of their transformer
best = scores['plus_street_meters']
best = best.assign(rank=best.groupby('dt').p.rank(ascending=False))
by_type = best[best.y == 1].groupby('ttype').apply(lambda g: pd.Series(dict(n=len(g), found_top5_pct=100 * (g['rank'] <= K).mean())))
by_type.to_csv(f'{out}/by_theft_type.csv'); print(by_type)
lt = best[best.y == 1].groupby('long_term').apply(lambda g: 100 * (g['rank'] <= K).mean())
print('long-term vs new thieves found %', lt.to_dict())
# Honest drops: how often solar / vacant reach top 5 in each variant
for name, t in scores.items():
    t = t.assign(rank=t.groupby('dt').p.rank(ascending=False))
    for r in ['solar', 'vacant']:
        summary[f'{name}_{r}_in_top5_pct'] = float(100 * (t[t.role == r]['rank'] <= K).mean())

# Layer 1: street-segment probability vs truth
seg_true, seg_p, phase_true, phase_p, illegal_hit, illegal_n = [], [], [], [], 0, 0
for s, dt in dts.items():
    res, tr = results[s], dt['truth']
    T0, T1 = (60 - 14) * 24, 60 * 24
    stolen = (tr['true'][:, T0:T1] * (1 - tr['reg'][:, T0:T1])).sum(1)
    for p in range(3):
        st = stolen[dt['phase'] == p].sum() + sum(il['load'][T0:T1].sum() for il in tr['illegal'] if il['phase'] == p)
        phase_true.append(int(st > 0.02 * dt['phase_meter'][p, T0:T1].sum())); phase_p.append(res['phase_prob'][p])
        for sn in range(dt['n_streets']):
            m = (dt['phase'] == p) & (dt['street'] == sn)
            ss = stolen[m].sum() + sum(il['load'][T0:T1].sum() for il in tr['illegal'] if il['phase'] == p and il['street'] == sn)
            seg_true.append(int(ss > 0.02 * dt['street_meter'][sn, p, T0:T1].sum())); seg_p.append(res['seg_prob'][sn, p])
    for il in tr['illegal']:
        if il['start'] < 46:
            illegal_n += 1; illegal_hit += int(res['seg_prob'][il['street'], il['phase']] > 0.5)
seg_true, seg_p, phase_true, phase_p = map(np.array, (seg_true, seg_p, phase_true, phase_p))
summary.update(phase_auc=float(roc_auc_score(phase_true, phase_p)), phase_acc=float(((phase_p > 0.5) == phase_true).mean()),
               segment_auc=float(roc_auc_score(seg_true, seg_p)), segment_acc=float(((seg_p > 0.5) == seg_true).mean()),
               segments_with_theft_pct=float(100 * seg_true.mean()),
               illegal_connections=illegal_n, illegal_segments_flagged_pct=float(100 * illegal_hit / max(illegal_n, 1)))

# Inspection planning: field spot-check on the top-5 flagged thieves, planned vs random timing
rng = np.random.default_rng(99)
plan_hits = rand_hits = n_visits = 0
by_type_visit = {}
for s, dt in dts.items():
    t = best[best.dt == s].nlargest(K, 'p')
    day, hours = plan_inspection(dt, results[s], rng=rng)
    if dt['truth']['role'].tolist().count('thief') == 0: continue
    for _, r in t[t.y == 1].iterrows():
        rd = int(rng.integers(60, 90)); rh = int(rng.integers(8, 17))
        night_sig = r.drop_night - max(r.drop_evening, r.drop_midday) > 0.3
        vis = (22, 23) if night_sig else hours
        ph = field_check(dt, int(r.cust), day, vis); rh_ = field_check(dt, int(r.cust), rd, (rh, rh + 1))
        plan_hits += ph; rand_hits += rh_; n_visits += 1
        d = by_type_visit.setdefault(r.ttype, [0, 0, 0]); d[0] += ph; d[1] += rh_; d[2] += 1
summary.update(field_visits=n_visits, planned_visit_catch_pct=float(100 * plan_hits / n_visits),
               random_visit_catch_pct=float(100 * rand_hits / n_visits),
               visit_by_type={k: dict(planned=100 * v[0] / v[2], random=100 * v[1] / v[2], n=v[2]) for k, v in by_type_visit.items()})
curve = []
for name, t in scores.items():
    o = t.sort_values('p', ascending=False).reset_index(drop=True)
    for pct in [1, 2, 3, 5, 7, 10, 15]:
        k = int(len(o) * pct / 100)
        curve.append(dict(variant=name, inspect_pct=pct, thieves_found_pct=100 * o.y[:k].sum() / o.y.sum()))
pd.DataFrame(curve).to_csv(f'{out}/gains.csv', index=False)
b = scores['plus_street_meters']
flag = b.street_theft_prob > 0.5
summary.update(narrow_customers_in_flagged_segments_pct=float(100 * flag.mean()),
               narrow_thieves_in_flagged_segments_pct=float(100 * b.y[flag].sum() / b.y.sum()),
               narrow_avg_customers_per_flagged_segment=float(b[flag].groupby(['dt', 'street_theft_prob']).size().mean()))
json.dump(summary, open(f'{out}/summary.json', 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))

"""Electricity theft detection on the public SGCC dataset.
Compares models on (a) daily smart-meter data and (b) monthly totals only,
mimicking the monthly billing records most Nigerian DisCos already hold.
Usage: python etd.py <path to data.csv> <output dir>
"""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

src, out = sys.argv[1], sys.argv[2]
SEED = 42
df = pd.read_csv(src)
df = df.drop_duplicates('CONS_NO')
y = df['FLAG'].values.astype(int)
X = df.drop(columns=['CONS_NO', 'FLAG'])
X.columns = pd.to_datetime(X.columns, format='%Y/%m/%d')
X = X[sorted(X.columns)]
missing_rate = float(X.isna().mean().mean())

# Preprocess: linear interpolation of gaps, remaining NaN -> 0, clip per-customer outliers (mean + 3 sd)
X = X.interpolate(axis=1, limit_area='inside').fillna(0)
mu, sd = X.mean(axis=1), X.std(axis=1)
X = X.clip(upper=mu + 3 * sd, axis=0)
daily = X.values

monthly = X.T.groupby([X.columns.year, X.columns.month]).sum().T.values  # ~34 months
weekly = X.T.groupby(np.arange(X.shape[1]) // 7).sum().T.values


def engineered(series):
    """Simple, explainable features from any consumption series (rows = customers)."""
    s = series
    n = s.shape[1]
    t = np.arange(n)
    mean = s.mean(1); std = s.std(1)
    half = n // 2
    first, second = s[:, :half].mean(1), s[:, half:].mean(1)
    slope = ((t - t.mean()) * (s - mean[:, None])).sum(1) / ((t - t.mean()) ** 2).sum()
    diffs = np.diff(s, axis=1)
    big_drop = (diffs < -0.5 * (mean[:, None] + 1e-6)).sum(1) / n
    feats = {
        'mean': mean, 'std': std, 'cv': std / (mean + 1e-6),
        'min': s.min(1), 'max': s.max(1), 'median': np.median(s, 1),
        'zero_frac': (s == 0).mean(1),
        'second_half_vs_first': second / (first + 1e-6),
        'trend_slope_norm': slope / (mean + 1e-6),
        'sudden_drop_rate': big_drop,
        'last_quarter_vs_mean': s[:, -max(1, n // 4):].mean(1) / (mean + 1e-6),
        'below_half_mean_frac': (s < 0.5 * mean[:, None]).mean(1),
    }
    return pd.DataFrame(feats).replace([np.inf, -np.inf], 0).fillna(0)

F_daily = engineered(daily)
F_month = engineered(monthly)
sets = {
    'daily': np.hstack([weekly, F_daily.values]),   # weekly profile + features from daily data
    'monthly': np.hstack([monthly, F_month.values]),  # only what monthly billing gives
}

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)

models = {
    'Logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')),
    'Random forest': lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
    'Gradient boosting': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight='balanced', random_state=SEED),
}


def prec_at(y_true, score, k):
    top = np.argsort(-score)[:k]
    return float(y_true[top].mean())

results, scores = [], {}
yte = y[idx_te]
for data_name, M in sets.items():
    for mname, mk in models.items():
        m = mk().fit(M[idx_tr], y[idx_tr])
        p = m.predict_proba(M[idx_te])[:, 1]
        pred = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
        results.append(dict(data=data_name, model=mname, auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                            precision=pr, recall=rc, f1=f1, p_at_100=prec_at(yte, p, 100), p_at_500=prec_at(yte, p, 500)))
        scores[(data_name, mname)] = p
        print(results[-1], flush=True)

res = pd.DataFrame(results)
res.to_csv(f'{out}/results.csv', index=False)

# Inspection-budget simulation for best model on each data type: share of thieves caught when
# inspecting the top-k% riskiest customers, vs random inspection.
best = res.loc[res.groupby('data')['auc'].idxmax()]
curve = []
for _, r in best.iterrows():
    p = scores[(r['data'], r['model'])]
    order = np.argsort(-p)
    for pct in [1, 2, 5, 10, 15, 20, 30]:
        k = int(len(yte) * pct / 100)
        caught = yte[order[:k]].sum()
        curve.append(dict(data=r['data'], model=r['model'], inspect_pct=pct, thieves_caught_pct=100 * caught / yte.sum(),
                          hit_rate_pct=100 * caught / k))
pd.DataFrame(curve).to_csv(f'{out}/inspection_curve.csv', index=False)

# Feature importance (permutation-free: gradient boosting on engineered monthly features only, for explanation)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
rf.fit(F_month.values[idx_tr], y[idx_tr])
imp = pd.Series(rf.feature_importances_, index=F_month.columns).sort_values(ascending=False)
imp.to_csv(f'{out}/feature_importance_monthly.csv')
auc_feat_only = roc_auc_score(yte, rf.predict_proba(F_month.values[idx_te])[:, 1])

# Group means of key features for explanation
grp = F_month.assign(flag=y).groupby('flag')[['cv', 'zero_frac', 'second_half_vs_first', 'sudden_drop_rate', 'mean']].median()
grp.to_csv(f'{out}/feature_medians_by_class.csv')

summary = dict(customers=int(len(y)), thieves=int(y.sum()), normal=int((y == 0).sum()), days=int(daily.shape[1]),
               months=int(monthly.shape[1]), missing_rate=missing_rate, test_size=int(len(yte)), test_thieves=int(yte.sum()),
               base_rate=float(y.mean()), rf_monthly_features_only_auc=float(auc_feat_only))
json.dump(summary, open(f'{out}/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1)); print(imp); print(grp); print(pd.DataFrame(curve))

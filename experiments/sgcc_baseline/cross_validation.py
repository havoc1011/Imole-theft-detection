import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
df=pd.read_csv(sys.argv[1]).drop_duplicates('CONS_NO'); y=df.FLAG.values.astype(int)
X=df.drop(columns=['CONS_NO','FLAG']); X.columns=pd.to_datetime(X.columns,format='%Y/%m/%d'); X=X[sorted(X.columns)]
X=X.interpolate(axis=1,limit_area='inside').fillna(0); mu,sd=X.mean(1),X.std(1); X=X.clip(upper=mu+3*sd,axis=0)
daily=X.values
monthly=X.T.groupby([X.columns.year,X.columns.month]).sum().T.values
weekly=X.T.groupby(np.arange(X.shape[1])//7).sum().T.values
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


sets={'daily':np.hstack([weekly,engineered(daily).values]),'monthly':np.hstack([monthly,engineered(monthly).values])}
res={k:[] for k in sets}
hit={k:[] for k in sets}
for tr,te in StratifiedKFold(5,shuffle=True,random_state=7).split(sets['daily'],y):
    for k,M in sets.items():
        p=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,class_weight='balanced',random_state=7).fit(M[tr],y[tr]).predict_proba(M[te])[:,1]
        res[k].append(roc_auc_score(y[te],p)); kk=int(len(te)*0.1); hit[k].append(y[te][np.argsort(-p)[:kk]].mean())
out={k:dict(auc_mean=float(np.mean(v)),auc_sd=float(np.std(v)),top10_hit_mean=float(np.mean(hit[k])),top10_hit_sd=float(np.std(hit[k])),folds=[float(a) for a in v]) for k,v in res.items()}
json.dump(out,open(sys.argv[2],'w'),indent=1); print(json.dumps(out,indent=1))

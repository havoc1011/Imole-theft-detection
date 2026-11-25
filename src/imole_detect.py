"""Ìmọ́lẹ̀ detection: transformer phase balance -> street segment -> customer ranking,
plus an inspection-day planner and a simulated field spot-check."""
import numpy as np
from scipy.stats import norm

BASE_DAYS = 7      # first week used as each customer's own baseline
RECENT_DAYS = 14   # recent window for decisions


def _fill(metered, hod, supply):
    """Fill missing prepaid readings with the customer's own average for that hour."""
    m = metered.copy()
    for h in range(24):
        cols = (hod == h) & supply
        mean_h = np.nanmean(np.where(cols[None, :], m, np.nan), axis=1)
        nan = np.isnan(m) & (hod == h)[None, :]
        m[nan] = np.broadcast_to(mean_h[:, None], m.shape)[nan]
    return np.nan_to_num(m * supply[None, :])


def segment_probability(ntl_share, sigma0=0.012, prior=0.3, max_theft=0.6):
    """Bayes: P(theft | observed non-technical loss share) for a phase or street segment.
    Without theft the share ~ Normal(0, sigma0) (meter + loss-estimate error).
    With theft it is anywhere between 0 and max_theft."""
    l0 = norm.pdf(ntl_share, 0, sigma0)
    l1 = np.where((ntl_share > 0) & (ntl_share < max_theft), 1 / max_theft, 1e-9)
    return prior * l1 / (prior * l1 + (1 - prior) * l0)


def analyse_dt(dt, analysis_days=60, tech_err=0.2, rng=None):
    rng = rng or np.random.default_rng(0)
    T = analysis_days * 24
    hod, day, supply = dt['hod'][:T], dt['day'][:T], dt['supply'][:T]
    M = _fill(dt['metered'][:, :T], hod, supply)
    n = M.shape[0]
    phase, street, S = dt['phase'], dt['street'], dt['n_streets']
    c_hat = dt['tech_coef_true'] * (1 + rng.normal(0, tech_err))   # engineering estimate of losses
    base = day < BASE_DAYS
    recent = day >= analysis_days - RECENT_DAYS

    # --- Layer 1: phase and street-segment gaps -------------------------------------
    PM = dt['phase_meter'][:, :T]
    phase_gap = PM - np.array([M[phase == p].sum(0) for p in range(3)]) - c_hat * PM ** 2
    phase_share = phase_gap[:, recent].sum(1) / np.maximum(PM[:, recent].sum(1), 1e-9)
    SM = dt['street_meter'][:, :, :T]
    seg_gap = np.zeros_like(SM)
    for s in range(S):
        for p in range(3):
            mask = (phase == p) & (street == s)
            seg_gap[s, p] = SM[s, p] - M[mask].sum(0) - c_hat * SM[s, p] * PM[p]
    seg_share = seg_gap[:, :, recent].sum(2) / np.maximum(SM[:, :, recent].sum(2), 1e-9)
    phase_prob = segment_probability(phase_share)
    seg_prob = segment_probability(seg_share)

    # --- Layer 2: customer features -------------------------------------------------
    on_b, on_r = base & supply, recent & supply
    b0 = M[:, on_b].mean(1) + 1e-6
    b1 = M[:, on_r].mean(1)
    prof = np.stack([M[:, on_b & (hod == h)].mean(1) for h in range(24)], 1) + 1e-6
    def part(hours):
        hm = np.isin(hod, hours)
        return 1 - M[:, on_r & hm].mean(1) / (M[:, on_b & hm].mean(1) + 1e-6)
    night, midday, evening = list(range(22, 24)) + list(range(0, 6)), list(range(9, 17)), list(range(17, 22))
    expected = prof[:, hod] * supply[None, :]
    missing = np.clip(expected - M, 0, None)

    feats = {}
    feats['drop_overall'] = np.clip(1 - b1 / b0, -2, 1)
    feats['drop_night'] = np.clip(part(night), -2, 1)
    feats['drop_midday'] = np.clip(part(midday), -2, 1)
    feats['drop_evening'] = np.clip(part(evening), -2, 1)
    feats['solar_signature'] = feats['drop_midday'] - feats['drop_evening']
    med_peer = np.array([np.median(b1[phase == phase[i]]) for i in range(n)])
    feats['low_vs_peers'] = np.clip(1 - b1 / (med_peer + 1e-6), -3, 1)
    feats['zero_share_recent'] = (M[:, on_r] < 0.02).mean(1)
    feats['phase_ntl_share'] = phase_share[phase]
    feats['phase_theft_prob'] = phase_prob[phase]
    feats['street_ntl_share'] = seg_share[street, phase]
    feats['street_theft_prob'] = seg_prob[street, phase]
    corr_p, corr_s, explain = np.zeros(n), np.zeros(n), np.zeros(n)
    dmiss = missing.reshape(n, -1, 24).sum(2)
    for i in range(n):
        g_p, g_s = phase_gap[phase[i]], seg_gap[street[i], phase[i]]
        if missing[i].std() > 0:
            corr_p[i] = np.corrcoef(missing[i], g_p)[0, 1]
            corr_s[i] = np.corrcoef(missing[i], g_s)[0, 1]
        gs = g_s.reshape(-1, 24).sum(1)
        explain[i] = np.clip(dmiss[i, -RECENT_DAYS:].sum() / max(gs[-RECENT_DAYS:].sum(), 1e-6), 0, 3)
    # Long-term thieves have no "before": compare their deficit vs neighbours with the segment gap
    seg_gap_kw = np.array([seg_gap[street[i], phase[i]][on_r].mean() for i in range(n)])
    deficit = np.clip(med_peer - b1, 0, None)
    feats['deficit_vs_segment_gap'] = np.clip(deficit / np.maximum(seg_gap_kw, 1e-3), 0, 3) * (seg_gap_kw > 0.05)
    feats['gap_corr_phase'] = np.nan_to_num(corr_p)
    feats['gap_corr_street'] = np.nan_to_num(corr_s)
    feats['share_of_gap_explained'] = explain
    X = np.column_stack(list(feats.values()))
    return dict(X=X, names=list(feats), phase_prob=phase_prob, seg_prob=seg_prob,
                phase_share=phase_share, seg_share=seg_share, phase_gap=phase_gap, seg_gap=seg_gap,
                gap_by_hod=np.array([phase_gap.sum(0)[recent & supply & (hod == h)].mean()
                                     if (recent & supply & (hod == h)).any() else 0 for h in range(24)]))


FEATURE_SETS = {
    'customer_only': ['drop_overall', 'drop_night', 'drop_midday', 'drop_evening', 'solar_signature',
                      'low_vs_peers', 'zero_share_recent'],
}
FEATURE_SETS['plus_phase_balance'] = FEATURE_SETS['customer_only'] + ['phase_ntl_share', 'phase_theft_prob',
                                                                       'gap_corr_phase']
FEATURE_SETS['plus_street_meters'] = FEATURE_SETS['plus_phase_balance'] + ['street_ntl_share', 'street_theft_prob',
                                                                            'gap_corr_street', 'share_of_gap_explained', 'deficit_vs_segment_gap']


def plan_inspection(dt, res, start_day=60, horizon=30, top_n=3, rng=None, forecast_err=2.0,
                    work_hours=range(7, 22)):
    """Pick an inspection day and 2-hour window for the coming month.
    Expected gap on a day = typical stolen kW in each hour x chance supply is on.
    Supply hours are taken from a noisy forecast (dispatch / load-shedding plan)."""
    rng = rng or np.random.default_rng(0)
    sup = dt['supply'].reshape(-1, 24)[start_day:start_day + horizon]
    fc_hours = np.clip(sup.sum(1) + rng.normal(0, forecast_err, len(sup)), 0, 24)
    gap_h = np.clip(res['gap_by_hod'], 0, None)
    exp_gap = fc_hours / 24 * gap_h[list(work_hours)].sum()
    best_days = np.argsort(-exp_gap)[:top_n]
    d = int(rng.choice(best_days))                        # random among the best: harder to predict
    wh = np.array(list(work_hours))
    win = [gap_h[h] + gap_h[h + 1] for h in wh[:-1]]
    h0 = int(wh[:-1][int(np.argmax(win))])
    return start_day + d, (h0, h0 + 1)


def field_check(dt, cust, day, hours):
    """Inspector clamps the service drop at the pole and compares with the meter for the window.
    Theft is caught if supply is on and the meter is under-registering during the visit."""
    t = [day * 24 + h for h in hours]
    on = dt['supply'][t]
    reg = dt['truth']['reg'][cust, t]
    return bool(np.any(on & (reg < 0.9)))


def monthly_survey_plan(dt, res, cust_scores, month_start, threshold=0.5, max_visits=10, rng=None):
    """ONE survey per month, triggered only by the AI.
    The survey team moves only if at least one street segment has P(theft) above the threshold.
    Returns None (no survey this month) or a plan: day, time window, segments and households to visit.
    cust_scores: risk score per customer from the trained model (higher = more suspicious)."""
    flagged = np.argwhere(res['seg_prob'] > threshold)          # (street, phase) pairs
    if len(flagged) == 0:
        return None
    day, window = plan_inspection(dt, res, start_day=month_start, rng=rng)
    in_flag = np.zeros(len(cust_scores), bool)
    for s, p in flagged:
        in_flag |= (dt['street'] == s) & (dt['phase'] == p)
    cand = np.where(in_flag)[0]
    visit = cand[np.argsort(-cust_scores[cand])][:max_visits]
    names = dict(zip(res['names'], res['X'].T))
    night = names['drop_night'] - np.maximum(names['drop_evening'], names['drop_midday']) > 0.3
    return dict(day=int(day), window=window,
                segments=[dict(street=int(s) + 1, phase='ABC'[p], p_theft=float(res['seg_prob'][s, p])) for s, p in flagged],
                households=[dict(customer=int(i), risk=float(cust_scores[i]),
                                 visit='night operation 22:00-23:00' if night[i] else f'{window[0]}:00-{window[1] + 1}:00')
                            for i in visit])

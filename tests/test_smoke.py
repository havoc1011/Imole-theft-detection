import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from imole_sim import simulate_dt
from imole_detect import analyse_dt, monthly_survey_plan


def _run(seed, clean):
    dt = simulate_dt(seed, days=90, theft_rate=0 if clean else 0.08, p_illegal=0 if clean else 0.5)
    res = analyse_dt(dt, rng=np.random.default_rng(1))
    score = res['X'][:, res['names'].index('street_theft_prob')]
    return dt, res, monthly_survey_plan(dt, res, score, 60, rng=np.random.default_rng(2))


def test_theft_transformer_triggers_one_survey():
    dt, res, plan = _run(7, clean=False)
    assert plan is not None
    assert 60 <= plan['day'] < 90
    assert len(plan['households']) <= 10


def test_clean_transformer_triggers_no_survey():
    _, _, plan = _run(5, clean=True)
    assert plan is None


def test_probabilities_are_valid():
    _, res, _ = _run(7, clean=False)
    assert np.all((res['seg_prob'] >= 0) & (res['seg_prob'] <= 1))

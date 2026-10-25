"""Tests for the noise helper in scripts.run_noise_test."""
import numpy as np
import pandas as pd
import pytest

from scripts import run_noise_test as nt


def frame(n=2000, seed=0):
    r = np.random.default_rng(seed)
    out_t = r.uniform(0, 30, n)
    ret = np.full(n, 22.0)
    f = r.uniform(0.1, 1.0, n)
    return pd.DataFrame({"OA_TEMP_C": out_t, "RA_TEMP_C": ret,
                         "MA_TEMP_C": ret + f * (out_t - ret), "OA_DMPR": f})


def test_zero_sigma_reproduces_the_clean_f():
    df = frame()
    out = nt.add_noise(df, 0.0, np.random.default_rng(1), min_delta=3.0)
    gap = (df["OA_TEMP_C"] - df["RA_TEMP_C"]).abs() >= 3.0
    assert out["well_conditioned"].to_numpy().tolist() == gap.to_numpy().tolist()
    expected = (df["MA_TEMP_C"] - df["RA_TEMP_C"]) / (df["OA_TEMP_C"] - df["RA_TEMP_C"])
    assert np.allclose(out.loc[gap, "f_phys"], expected[gap])


def test_noise_is_repeatable_and_grows_with_sigma():
    df = frame()
    a = nt.add_noise(df, 0.5, np.random.default_rng(7), 3.0)
    b = nt.add_noise(df, 0.5, np.random.default_rng(7), 3.0)
    assert np.allclose(a["f_phys"], b["f_phys"], equal_nan=True)
    big = nt.add_noise(df, 2.0, np.random.default_rng(7), 3.0)
    ok = a["well_conditioned"] & big["well_conditioned"]
    clean = nt.add_noise(df, 0.0, np.random.default_rng(7), 3.0)
    ok &= clean["well_conditioned"]
    err_small = (a.loc[ok, "f_phys"] - clean.loc[ok, "f_phys"]).abs().mean()
    err_big = (big.loc[ok, "f_phys"] - clean.loc[ok, "f_phys"]).abs().mean()
    assert err_big > err_small


def test_negative_sigma_raises():
    with pytest.raises(ValueError):
        nt.add_noise(frame(), -1.0, np.random.default_rng(0), 3.0)

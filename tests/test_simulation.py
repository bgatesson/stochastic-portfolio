
"""
Test for simulation module.
"""
from __future__ import annotations

import pytest
import numpy as np

def test_psd_cholesky_handles_rank_deficient_matrix():
    """Explicitly non-PD correlation matrix (forced negative eigenvalues) should still cholesky."""
    from spo.models.simulate import _psd_cholesky
    rng = np.random.default_rng(0)
    n = 50
    X = rng.standard_normal((n, n))
    corr = np.corrcoef(X, rowvar=False)
    # Force five eigenvalues to -0.02 so the matrix is definitively non-PD
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[:5] = -0.02
    bad = eigvecs @ (eigvals[:, None] * eigvecs.T)
    d = np.sqrt(np.diag(bad))
    bad = bad / np.outer(d, d)
    np.fill_diagonal(bad, 1.0)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(bad + 1e-12 * np.eye(n))
    # Our helper handles it
    L = _psd_cholesky(bad)
    reconstructed = L @ L.T
    np.testing.assert_allclose(np.diag(reconstructed), 1.0, atol=1e-6)


def test_psd_cholesky_preserves_well_conditioned_matrix():
    """On a well-conditioned correlation matrix, output should be near-identical."""
    from spo.models.simulate import _psd_cholesky
    rng = np.random.default_rng(1)
    L_true = np.tril(rng.standard_normal((10, 10)) * 0.3) + np.eye(10)
    cov = L_true @ L_true.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    L_ours = _psd_cholesky(corr)
    L_naive = np.linalg.cholesky(corr)
    np.testing.assert_allclose(L_ours @ L_ours.T, L_naive @ L_naive.T, atol=1e-6)
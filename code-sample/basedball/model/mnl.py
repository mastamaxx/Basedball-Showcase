"""Multinomial logistic regression with an offset, linear in its features.

    logit_c = offset_c + sum_j x_j * W[j, c]          P = softmax(logit)

The offset is the log-probability from a base model (for sizing: the core
prediction), so the weights measure what a factor adds on top of it. The first
class is the reference (its column of W is fixed at 0). A small ridge penalty keeps
rarely-seen columns from running off.

Matrix products run in the features' dtype (float32 features keep memory down on the
1.1M-row tables); the loss is accumulated in float64.
"""
import numpy as np
from scipy.optimize import minimize


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit(X, y, offset, n_classes, ridge=1.0, maxiter=300, weight=None, W0=None, tol=1e-9):
    """X: (n, p) features; y: (n,) class index; offset: (n, K) log-probabilities.
    W0: optional starting weights (p, K)."""
    n, p = X.shape
    K = n_classes
    dt = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    X = X.astype(dt, copy=False)
    offset = offset.astype(dt, copy=False)
    rows = np.arange(n)
    w = None if weight is None else weight.astype(dt)
    sw = float(n if weight is None else weight.sum())

    def f(theta):
        W = np.zeros((p, K), dtype=dt)
        W[:, 1:] = theta.reshape(p, K - 1)
        Z = offset + X @ W
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        s = E.sum(1)
        lse = np.log(s)
        lp = Z[rows, y] - lse
        ll = float((lp.astype(np.float64) if w is None else (w * lp).astype(np.float64)).sum())
        E /= s[:, None]                       # P
        E[rows, y] -= 1                        # P - Y
        if w is not None:
            E *= w[:, None]
        G = (X.T @ E).astype(np.float64)
        loss = -ll / sw + 0.5 * ridge * (theta ** 2).sum() / sw
        grad = G[:, 1:].ravel() / sw + ridge * theta / sw
        return loss, grad

    th0 = np.zeros(p * (K - 1)) if W0 is None else np.asarray(W0, dtype=np.float64)[:, 1:].ravel()
    res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-7})
    W = np.zeros((p, K))
    W[:, 1:] = res.x.reshape(p, K - 1)
    return W


def predict_logp(X, offset, W):
    dt = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    Z = offset.astype(dt, copy=False) + X.astype(dt, copy=False) @ W.astype(dt)
    Z = Z - Z.max(1, keepdims=True)
    return (Z - np.log(np.exp(Z).sum(1, keepdims=True))).astype(np.float64)


def fit_grouped(Xf, Xg, y, offset, n_classes, group_of, ridge=1.0, maxiter=300, W0f=None, W0g=None):
    """Like fit, but the columns of Xg get one weight per outcome group (group_of[c] = group of
    class c; the reference class's group is fixed at 0) while the columns of Xf get one weight per
    outcome. Returns (Wf (pf, K), Wg_full (pg, K))."""
    n = len(y)
    K = n_classes
    group_of = np.asarray(group_of)
    ref_g = group_of[0]
    free_g = [g for g in sorted(set(group_of)) if g != ref_g]
    pf = 0 if Xf is None else Xf.shape[1]
    pg = 0 if Xg is None else Xg.shape[1]
    dt = np.float32
    rows = np.arange(n)
    off = offset.astype(dt, copy=False)
    Gm = np.zeros((len(free_g), K), dtype=dt)          # group -> classes
    for i, g in enumerate(free_g):
        Gm[i, group_of == g] = 1

    def unpack(th):
        Wf = np.zeros((pf, K), dtype=dt)
        if pf:
            Wf[:, 1:] = th[:pf * (K - 1)].reshape(pf, K - 1)
        Wg = th[pf * (K - 1):].reshape(pg, len(free_g)).astype(dt) if pg else np.zeros((0, len(free_g)), dt)
        return Wf, Wg

    def f(th):
        Wf, Wg = unpack(th)
        Z = off.copy()
        if pf:
            Z += Xf @ Wf
        if pg:
            Z += (Xg @ Wg) @ Gm
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        s_ = E.sum(1)
        ll = float((Z[rows, y] - np.log(s_)).astype(np.float64).sum())
        E /= s_[:, None]
        E[rows, y] -= 1
        g = []
        if pf:
            g.append((Xf.T @ E).astype(np.float64)[:, 1:].ravel())
        if pg:
            g.append(((Xg.T @ E) @ Gm.T).astype(np.float64).ravel())
        grad = np.concatenate(g) / n + ridge * th / n
        return -ll / n + 0.5 * ridge * (th ** 2).sum() / n, grad

    th0 = np.zeros(pf * (K - 1) + pg * len(free_g))
    if W0f is not None and pf:
        th0[:pf * (K - 1)] = np.asarray(W0f)[:, 1:].ravel()
    if W0g is not None and pg:
        th0[pf * (K - 1):] = np.asarray(W0g).ravel()
    res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
    Wf, Wg = unpack(res.x)
    return Wf.astype(np.float64), (Wg @ Gm).astype(np.float64)

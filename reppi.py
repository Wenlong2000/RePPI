import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier

from ppi import _wls, _ols_get_stats, _zconfint_generic


def sample_split(n):
    index = np.arange(n)
    np.random.shuffle(index)
    index1 = index[:n // 3]
    index2 = index[n // 3:2 * n // 3]
    index3 = index[2 * n // 3:]
    return index1, index2, index3


def grad_fit_ols(
        X,
        Y,
        Yhat,
        theta,
        r,
        method='mlp',
):
    covariates = np.concatenate([X, Yhat], axis=1)
    if method == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(32, 32, 32))
    elif method == 'mlpclass':
        model = MLPClassifier(hidden_layer_sizes=(32, 32, 32))
    elif method == 'linreg':
        model = LinearRegression()
    elif method == 'logistic':
        model = LogisticRegression()
    elif method == 'xgboost':
        model = xgb.XGBRegressor()
    elif method == 'tree':
        model = DecisionTreeRegressor()
    else:
        raise ValueError(f"Method {method} not recognized.")
    model.fit(covariates, Y)

    def f(X_new, Yhat_new):
        if method == 'logistic':
            pred = model.predict_proba(np.concatenate([X_new, Yhat_new], axis=1))[:, 1]
        else:
            pred = model.predict(np.concatenate([X_new, Yhat_new], axis=1))
        if method == 'linreg':
            pred = pred[:, 0]
        res = X_new @ theta - pred
        res = np.expand_dims(res, axis=1)
        return 1 / (1 + r) * X_new * res
    return f


def ppi_opt_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        grad,
        theta_0,
        w=None,
        w_unlabeled=None,
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    r = n / N
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / np.sum(w_unlabeled) * N
    )
    grad_unlabeled_g = grad(X_unlabeled, Yhat_unlabeled)
    grad_labeled_g = grad(X, Yhat)
    grad_labeled_l = X * (X @ np.expand_dims(theta_0, axis=1) - Y)
    cov_label_unlabel = np.cov(grad_labeled_l.T, grad_labeled_g.T)
    M = 1 / (1 + r) * cov_label_unlabel[:d, d:] @ np.linalg.inv(cov_label_unlabel[d:, d:])
    grad_unlabeled = np.expand_dims(np.average(grad_unlabeled_g, axis=0, weights=w_unlabeled), axis=1)
    grad_labeled = np.expand_dims(np.average(grad_labeled_g, axis=0, weights=w), axis=1)

    Sigma_inv = np.linalg.inv(X.T @ np.diag(w) @ X / n)

    theta = Sigma_inv @ ((X.T @ np.diag(w) @ Y) / n + M @ (grad_labeled - grad_unlabeled))
    return theta, M


def ppi_opt_ols_pointestimate_crossfit(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w=None,
        w_unlabeled=None,
        method='mlp',
        return_grad=False,
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    r = n / N
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    Y = np.expand_dims(Y, axis=1)
    Yhat = np.expand_dims(Yhat, axis=1)
    Yhat_unlabeled = np.expand_dims(Yhat_unlabeled, axis=1)

    index1, index2, index3 = sample_split(n)

    theta_1 = _wls(X[index1], Y[index1], w=w[index1])
    theta_2 = _wls(X[index2], Y[index2], w=w[index2])
    theta_3 = _wls(X[index3], Y[index3], w=w[index3])

    grad_g_1 = grad_fit_ols(X[index2], Y[index2], Yhat[index2], theta_1, r, method=method)
    grad_g_2 = grad_fit_ols(X[index3], Y[index3], Yhat[index3], theta_2, r, method=method)
    grad_g_3 = grad_fit_ols(X[index1], Y[index1], Yhat[index1], theta_3, r, method=method)

    est_1, M_1 = ppi_opt_ols_pointestimate(
        X[index3],
        Y[index3],
        Yhat[index3],
        X_unlabeled,
        Yhat_unlabeled,
        grad_g_1,
        theta_1,
        w=w[index3],
        w_unlabeled=w_unlabeled,
    )
    est_2, M_2 = ppi_opt_ols_pointestimate(
        X[index1],
        Y[index1],
        Yhat[index1],
        X_unlabeled,
        Yhat_unlabeled,
        grad_g_2,
        theta_2,
        w=w[index1],
        w_unlabeled=w_unlabeled,
    )
    est_3, M_3 = ppi_opt_ols_pointestimate(
        X[index2],
        Y[index2],
        Yhat[index2],
        X_unlabeled,
        Yhat_unlabeled,
        grad_g_3,
        theta_3,
        w=w[index2],
        w_unlabeled=w_unlabeled,
    )
    est = (est_1 + est_2 + est_3) / 3

    if return_grad:
        grad_unlabeled_1 = grad_g_1(X_unlabeled, Yhat_unlabeled) @ M_1.T
        grad_unlabeled_2 = grad_g_2(X_unlabeled, Yhat_unlabeled) @ M_2.T
        grad_unlabeled_3 = grad_g_3(X_unlabeled, Yhat_unlabeled) @ M_3.T
        grad_unlabeled = (grad_unlabeled_1 + grad_unlabeled_2 + grad_unlabeled_3) / 3

        grad_labeled_1 = grad_g_1(X[index3], Yhat[index3]) @ M_1.T
        grad_labeled_2 = grad_g_2(X[index1], Yhat[index1]) @ M_2.T
        grad_labeled_3 = grad_g_3(X[index2], Yhat[index2]) @ M_3.T
        grad_labeled = np.concatenate([grad_labeled_1, grad_labeled_2, grad_labeled_3], axis=0)
        index_labeled = np.concatenate([index3, index1, index2], axis=0)

        return est, grad_labeled, index_labeled, grad_unlabeled
    else:
        return est


def ppi_opt_ols_ci_crossfit(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    w=None,
    w_unlabeled=None,
    method='mlp',
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    ppi_opt_pointest, grads_g_labeled, index_labeled, grads_g_unlabeled = ppi_opt_ols_pointestimate_crossfit(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        method=method,
        return_grad=True,
    )
    grads, _, _, inv_hessian = _ols_get_stats(
        ppi_opt_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=True,
    )

    var_unlabeled = np.cov(grads_g_unlabeled.T, aweights=w_unlabeled).reshape(d, d)
    var = np.cov(grads[index_labeled].T - grads_g_labeled.T, aweights=w).reshape(d, d)
    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    return _zconfint_generic(
        ppi_opt_pointest[:,0],
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )


"""
Self-learning phase duration estimation module (EaSiCon thesis, Chapter 5).

Model (per production phase f):
    log(d) = mu_f + eps,   eps ~ N(0, sigma^2)
    mu_f   = beta_0f + sum_j beta_jf * x_j + sum_k gamma_kf * z_kf     (Eq. 5.1)
    d      ~ Lognormal(mu_f, sigma^2)                                  (Eq. 5.2)

Inference: conjugate Normal-Inverse-Gamma (NIG) Bayesian linear regression
on the log scale. This choice makes the sequential update in Eq. 5.3 exact
and closed-form: the posterior after project t is literally the prior for
project t+1, with no approximation and no MCMC.

NOTE FOR THE THESIS: the choice of a conjugate NIG family is itself a design
decision that must be documented in Chapter 5 (alternatives: MCMC with
non-conjugate priors, variational inference). It constrains the prior for
sigma^2 to an Inverse-Gamma and couples the coefficient prior scale to sigma.

Author: skeleton generated for Gijs Meijers' TIL thesis; all numerical priors
in here are PLACEHOLDERS and must be replaced by elicited values (Ch. 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
# Prior construction from three-point expert estimates
# ----------------------------------------------------------------------

def threepoint_to_lognormal(optimistic: float, most_likely: float,
                            pessimistic: float,
                            coverage: float = 0.90) -> tuple[float, float]:
    """Translate an expert three-point duration estimate (in days) into
    (mu, sigma) of a lognormal, treating [optimistic, pessimistic] as a
    central `coverage` interval on the log scale and the most likely value
    as the mode.

    Returns (mu, sigma) on the log scale.

    This is ONE possible translation rule. The thesis must justify and fix
    the rule explicitly (Section 4.4.2 promises this in Chapter 6). An
    alternative is PERT-style moment matching; the interval rule used here
    is more robust to experts anchoring on symmetric ranges.
    """
    if not (0 < optimistic <= most_likely <= pessimistic):
        raise ValueError("Require 0 < optimistic <= most_likely <= pessimistic")
    z = stats.norm.ppf(0.5 + coverage / 2.0)
    lo, hi = np.log(optimistic), np.log(pessimistic)
    sigma = (hi - lo) / (2.0 * z)
    mu = (hi + lo) / 2.0
    return mu, sigma


def spread_question_to_sigma(earliest: float, latest: float,
                             coverage: float = 0.95) -> float:
    """Translate the '100 identical projects' spread question (Sec. 4.4.2)
    into a prior point value for the residual sigma on the log scale."""
    if not 0 < earliest < latest:
        raise ValueError("Require 0 < earliest < latest")
    z = stats.norm.ppf(0.5 + coverage / 2.0)
    return (np.log(latest) - np.log(earliest)) / (2.0 * z)


# ----------------------------------------------------------------------
# The Bayesian regression model for one production phase
# ----------------------------------------------------------------------

@dataclass
class PhaseDurationModel:
    """Bayesian lognormal regression for a single production phase.

    Parameters of the NIG prior:
        beta | sigma^2 ~ N(m, sigma^2 * V)
        sigma^2        ~ InvGamma(a, b)

    Design vector convention: first element is the intercept (baseline
    log-duration beta_0f), followed by general covariates x_j, followed by
    phase-specific covariates z_kf. Standardise covariates BEFORE elicitation
    and modelling; the coefficient priors below assume standardised inputs.
    """
    phase: str
    covariate_names: list[str]              # excludes intercept
    m: np.ndarray = field(default=None)     # (p+1,) prior mean of beta
    V: np.ndarray = field(default=None)     # (p+1, p+1) prior scale matrix
    a: float = 3.0                          # InvGamma shape (a>2 => finite var)
    b: float = None                         # InvGamma scale
    n_obs: int = 0                          # completed projects absorbed

    # -- construction ---------------------------------------------------
    @classmethod
    def from_elicitation(cls, phase: str, covariate_names: list[str],
                         baseline_threepoint: tuple[float, float, float],
                         coef_prior_means: dict[str, float],
                         coef_prior_sds: dict[str, float],
                         sigma_prior: float,
                         sigma_confidence: float = 5.0):
        """Initialise from elicited quantities.

        baseline_threepoint: (opt, ml, pess) duration in days for the
            reference project (all standardised covariates = 0).
        coef_prior_means/sds: per covariate, prior mean and sd of its effect
            on LOG duration per 1 sd of the covariate. Derived from the
            paired-scenario elicitation (Sec. 4.4.2): if the comparison
            scenario is judged r times longer than baseline, the coefficient
            mean is log(r) divided by the covariate step in sd units.
        sigma_prior: point value for residual sigma (from spread question).
        sigma_confidence: pseudo-observations backing the sigma prior.
        """
        p = len(covariate_names)
        mu0, s0 = threepoint_to_lognormal(*baseline_threepoint)
        m = np.array([mu0] + [coef_prior_means[c] for c in covariate_names])
        sds = np.array([s0] + [coef_prior_sds[c] for c in covariate_names])

        a = sigma_confidence / 2.0 + 1.0
        b = sigma_prior ** 2 * (a - 1.0)        # so E[sigma^2] = sigma_prior^2
        # V is defined relative to sigma^2 in the NIG family:
        V = np.diag((sds / sigma_prior) ** 2)
        return cls(phase=phase, covariate_names=covariate_names,
                   m=m, V=V, a=a, b=b)

    # -- internals -------------------------------------------------------
    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)), X])

    # -- the self-learning mechanism (Eq. 5.3) ----------------------------
    def update(self, X: np.ndarray, durations: np.ndarray) -> None:
        """Absorb realised phase durations of completed projects.

        X: (n, p) covariate matrix (standardised, no intercept column).
        durations: (n,) realised durations in days (must be positive).

        Closed-form NIG update; the resulting posterior becomes the new
        prior state of the object, so calling update project by project
        is exactly the sequential updating of Eq. 5.3.
        """
        durations = np.asarray(durations, dtype=float)
        if np.any(durations <= 0):
            raise ValueError("Durations must be positive")
        y = np.log(durations)
        Phi = self._design(X)
        n = Phi.shape[0]

        V_inv = np.linalg.inv(self.V)
        V_post = np.linalg.inv(V_inv + Phi.T @ Phi)
        m_post = V_post @ (V_inv @ self.m + Phi.T @ y)
        a_post = self.a + n / 2.0
        b_post = self.b + 0.5 * (y @ y + self.m @ V_inv @ self.m
                                 - m_post @ np.linalg.inv(V_post) @ m_post)

        self.m, self.V, self.a, self.b = m_post, V_post, a_post, b_post
        self.n_obs += n

    # -- prediction --------------------------------------------------------
    def predict(self, x: np.ndarray, interval: float = 0.90) -> dict:
        """Posterior predictive for a new project with covariates x (p,).

        On the log scale the predictive is Student-t; intervals are
        exponentiated back to days. Returns median, mean, interval bounds
        and P(duration > threshold) helper values.
        """
        phi = self._design(x)[0]
        df = 2.0 * self.a
        loc = float(phi @ self.m)
        scale2 = (self.b / self.a) * (1.0 + float(phi @ self.V @ phi))
        scale = np.sqrt(scale2)

        alpha = (1.0 - interval) / 2.0
        t = stats.t(df=df, loc=loc, scale=scale)
        lo, hi = t.ppf(alpha), t.ppf(1.0 - alpha)
        return {
            "phase": self.phase,
            "median_days": float(np.exp(loc)),
            "interval": (float(np.exp(lo)), float(np.exp(hi))),
            "interval_level": interval,
            "log_loc": loc, "log_scale": scale, "df": df,
            "n_projects_learned": self.n_obs,
        }

    def prob_exceeds(self, x: np.ndarray, threshold_days: float) -> float:
        """P(duration > threshold) for delivery-window risk assessment."""
        pred = self.predict(x)
        t = stats.t(df=pred["df"], loc=pred["log_loc"], scale=pred["log_scale"])
        return float(1.0 - t.cdf(np.log(threshold_days)))

    # -- introspection ------------------------------------------------------
    def coefficient_summary(self) -> dict:
        """Posterior mean and sd of each coefficient (marginal Student-t)."""
        df = 2.0 * self.a
        var_scale = self.b / self.a
        names = ["intercept"] + self.covariate_names
        out = {}
        for i, name in enumerate(names):
            sd = np.sqrt(var_scale * self.V[i, i] * df / (df - 2.0))
            out[name] = {"mean": float(self.m[i]), "sd": float(sd)}
        return out

    def sigma_estimate(self) -> float:
        """Posterior mean of sigma^2, square-rooted."""
        return float(np.sqrt(self.b / (self.a - 1.0)))

import flib
import numpy as np
from numpy import pi, inf
import itertools
import pdb
import warnings

# Wishart---------------------------------------------------
def rvs(n, tau=None, cov=None):
    """
    rwishart(n, tau=None, cov=None)

    Return a Wishart random matrix.

    :Parameters:
      - `n` : Degrees of freedom.
      - `tau` : (k,k) Positive definite precision matrix.
      - `cov` : (k,k) Positive definite covariance matrix.
    """
    if (tau is None and  cov is None):
        raise ValueError('Must specify either tau or cov.')
    if tau is not None:
        p = np.shape(tau)[0]
        sig = np.linalg.cholesky(tau)
        if n<p:
            raise ValueError('Wishart parameter n must be greater '
                             'than size of matrix.')
        norms = np.random.normal(size=p*(p-1)/2)
        chi_sqs = np.sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
        A = flib.expand_triangular(chi_sqs, norms)

        flib.dtrsm_wrap(sig, A, side='L', uplo='L', transa='T', alpha=1.)
        w = np.asmatrix(np.dot(A,A.T))
        flib.symmetrize(w)
        return w
    elif cov is not None:
        p = np.shape(cov)[0]
        # Need cholesky decomposition of precision matrix C^-1?
        sig = np.linalg.cholesky(cov)

        if n<p:
            raise ValueError('Wishart parameter n must be greater '
                             'than size of matrix.')

        norms = np.random.normal(size=p*(p-1)/2)
        chi_sqs = np.sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
        A = flib.expand_triangular(chi_sqs, norms)

        flib.dtrmm_wrap(sig, A, side='L', uplo='L', transa='N', alpha=1.)
        w = np.asmatrix(np.dot(A,A.T))
        flib.symmetrize(w)
        return w

def expval(n, tau=None, cov=None):
    """
    wishart_expval(n, tau=None, cov=None)

    Expected value of wishart distribution.
    """
    if tau is None and cov is None:
        raise ValueError('Must specify either tau or cov.')
    if tau is not None:
        return n * np.asarray(tau.I)
    elif cov is not None:
        return n * np.asarray(cov)

def pdf(X, n, tau=None, cov=None):
    R"""
    wishart_like(X, n, tau=None, cov=None)

    Wishart log-likelihood. The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution. If tau=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}`, see
    `wishart_cov_like`.

    .. math::
        f(X \mid n, T) = {\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      tau : matrix
        Symmetric and positive definite

    .. note::
      Step method MatrixMetropolis will preserve the symmetry of Wishart variables.

    """
    if tau is None and cov is None:
        raise ValueError('Must specify either tau or cov.')
    if tau is not None:
        return flib.blas_wishart(X,n,tau)
    elif cov is not None:
        return flib.blas_wishart_cov(X,n,cov)

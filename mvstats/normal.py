"""
Module supporting sampling and pdf evaluation for the MV Normal distribution
"""


import flib
import numpy as np
from numpy import pi, inf
import itertools
import pdb
import warnings

# Multivariate normal--------------------------------------
def rvs(mu, tau=None, cov=None, sig=None, size=1):
    """
    rvs(mu, tau=None, cov=None, sig=None, size=1)

    Random multivariate normal variates.

    :Parameters:
      - `mu` : Mean vector.
      - `tau` : Precision matrix.
      - `cov` : Covariance matrix.
      - `sig` : Lower-triangular matrix resulting from the Cholesky 
            decomposition of the covariance matrix.
      - `size` : Number of random samples to generate.
    """
    if (tau is not None and cov is not None) or (tau is not None and sig is not None) or (cov is not None and sig is not None):
        raise ValueError('Only specify one of {tau, cov, sig}.')
    if tau is None and cov is None and sig is None:
        # Default the precision to identity matrix.
        tau = np.eye(mu.size)
    if tau is not None:
        sig = np.linalg.cholesky(tau)
        mu_size = np.shape(mu)

        if size==1:
            out = np.random.normal(size=mu_size)
            try:
                flib.dtrsm_wrap(sig , out, 'L', 'T', 'L', 1.)
            except:
                out = np.linalg.solve(sig, out)
            out+=mu
            return out
        else:
            if not hasattr(size,'__iter__'):
                size = (size,)
            tot_size = np.prod(size)
            out = np.random.normal(size = (tot_size,) + mu_size)
            for i in xrange(tot_size):
                try:
                    flib.dtrsm_wrap(sig , out[i,:], 'L', 'T', 'L', 1.)
                except:
                    out[i,:] = np.linalg.solve(sig, out[i,:])
                out[i,:] += mu
            return out.reshape(size+mu_size)
    elif cov is not None:
        mu_size = np.shape(mu)
        if size==1:
            return np.random.multivariate_normal(mu, cov, size).reshape(mu_size)
        else:
            return np.random.multivariate_normal(mu, cov, size).reshape((size,)+mu_size)
    elif sig is not None:
        mu_size = np.shape(mu)

        if size==1:
            out = np.random.normal(size=mu_size)
            try:
                flib.dtrmm_wrap(sig , out, 'L', 'N', 'L', 1.)
            except:
                out = np.dot(sig, out)
            out+=mu
            return out
        else:
            if not hasattr(size,'__iter__'):
                size = (size,)
            tot_size = np.prod(size)
            out = np.random.normal(size = (tot_size,) + mu_size)
            for i in xrange(tot_size):
                try:
                    flib.dtrmm_wrap(sig , out[i,:], 'L', 'N', 'L', 1.)
                except:
                    out[i,:] = np.dot(sig, out[i,:])
                out[i,:] += mu
            return out.reshape(size+mu_size)

def expval(mu, tau=None, cov=None, sig=None):
    """
    expval(mu, tau=None, cov=None, sig=None)

    Expected value of multivariate normal distribution.
    """
    return mu

def logpdf(x, mu, tau=None, cov=None, sig=None):
    R"""
    logpdf(x, mu, tau=None, cov=None, sig=None)

    Multivariate normal log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{|T|^{1/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}

    :Parameters:
      - `x` : (n,k)
      - `mu` : (k) Location parameter sequence.
      - `tau` : (k,k) Positive definite precision matrix.
      - `cov` : (k,k) Positive definite covariance matrix.
      - `sig` : (k,k) Lower triangular matrix resulting from a Cholesky 
                decomposition of the covariance matrix.
    """
    if (tau is not None and cov is not None) or (tau is not None and sig is not None) or (cov is not None and sig is not None):
        raise ValueError('Only specify one of {tau, cov, sig}.')
    if tau is None and cov is None and sig is None:
        # Default the precision to identity matrix.
        tau = np.eye(mu.size)
    if tau is not None:
        # TODO: Vectorize in Fortran
        if len(np.shape(x))>1:
            return np.sum([flib.prec_mvnorm(r,mu,tau) for r in x])
        else:
            return flib.prec_mvnorm(x,mu,tau)
    elif cov is not None:
        # TODO: Vectorize in Fortran
        if len(np.shape(x))>1:
            return np.sum([flib.cov_mvnorm(r,mu,cov) for r in x])
        else:
            return flib.cov_mvnorm(x,mu,cov)
    elif sig is not None:
        # TODO: Vectorize in Fortran
        if len(np.shape(x))>1:
            return np.sum([flib.chol_mvnorm(r,mu,sig) for r in x])
        else:
            return flib.chol_mvnorm(x,mu,sig)

def pdf(x,mu,tau=None,cov=None,sig=None):
    """
    pdf(x, mu, tau=None, cov=None, sig=None)

    :Parameters:
      - `x` : (n,k)
      - `mu` : (k) Location parameter sequence.
      - `tau` : (k,k) Positive definite precision matrix.
      - `cov` : (k,k) Positive definite covariance matrix.
      - `sig` : (k,k) Lower triangular matrix resulting from a Cholesky 
                decomposition of the covariance matrix.
    """
    return np.exp(logpdf(x,mu,tau,cov,sig))

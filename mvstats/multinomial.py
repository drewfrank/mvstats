"""
Module supporting sampling and pdf evaluation for the Multinomial distribution

"""


import flib
import numpy as np
from numpy import pi, inf
import itertools
import pdb
import warnings

# Multinomial----------------------------------------------
def rvs(n,p,size=None):
    """
    rvs(n,p,size=1)

    Random multinomial variates.
    """
    # Leaving size=None as the default means return value is 1d array
    # if not specified-- nicer.

    # Single value for p:
    if len(np.shape(p))==1:
        return np.random.multinomial(n,p,size)

    # Multiple values for p:
    if np.isscalar(n):
        n = n * np.ones(np.shape(p)[0],dtype=np.int)
    out = np.empty(np.shape(p))
    for i in xrange(np.shape(p)[0]):
        out[i,:] = np.random.multinomial(n[i],p[i,:],size)
    return out

def expval(n,p):
    """
    expval(n,p)

    Expected value of multinomial distribution.
    """
    return np.asarray([pr * n for pr in p])

def logpdf(x, n, p):
    R"""
    logpdf(x, n, p)

    Multinomial log-likelihood. Generalization of the binomial
    distribution, but instead of each trial resulting in "success" or
    "failure", each one results in exactly one of some fixed finite number k
    of possible outcomes over n independent trials. 'x[i]' indicates the number
    of times outcome number i was observed over the n trials.

    .. math::
        f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    :Parameters:
      x : (ns, k) int
          Random variable indicating the number of time outcome i is
          observed. :math:`\sum_{i=1}^k x_i=n`, :math:`x_i \ge 0`.
      n : int
          Number of trials.
      p : (k,)
          Probability of each one of the different outcomes.
          :math:`\sum_{i=1}^k p_i = 1)`, :math:`p_i \ge 0`.

    .. note::
       - :math:`E(X_i)=n p_i`
       - :math:`Var(X_i)=n p_i(1-p_i)`
       - :math:`Cov(X_i,X_j) = -n p_i p_j`
       - If :math: `\sum_i p_i < 0.999999` a log-likelihood value of -inf
       will be returned.

    """
    # flib expects 2d arguments. Do we still want to support multiple p
    # values along realizations ?
    x = np.atleast_2d(x)
    p = np.atleast_2d(p)

    return flib.multinomial(x, n, p)

def pdf(x, n, p):
    """
    pdf(x, n, p)
    :Parameters:

      x : (ns, k) int
          Random variable indicating the number of time outcome i is
          observed. :math:`\sum_{i=1}^k x_i=n`, :math:`x_i \ge 0`.
      n : int
          Number of trials.
      p : (k,)
          Probability of each one of the different outcomes.
          :math:`\sum_{i=1}^k p_i = 1)`, :math:`p_i \ge 0`.
    """
    return np.exp( logpdf(x,n,p) )

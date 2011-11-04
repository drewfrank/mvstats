import flib
import numpy as np
from numpy import pi, inf
import itertools
import pdb
import warnings

# Multivariate hypergeometric------------------------------
def rvs(n, m, size=None):
    """
    rvs(n, m, size=None)

    Random multivariate hypergeometric variates.

    n : Number of draws.
    m : Number of items in each category.
    """

    N = len(m)
    urn = np.repeat(np.arange(N), m)

    if size:
        draw = np.array([[urn[i] for i in np.random.permutation(len(urn))[:n]]
                         for j in range(size)])

        r = [[np.sum(draw[j]==i) for i in range(len(m))]
             for j in range(size)]
    else:
        draw = np.array([urn[i] for i in np.random.permutation(len(urn))[:n]])

        r = [np.sum(draw==i) for i in range(len(m))]
    return np.asarray(r)

def expval(n, m):
    """
    expval(n, m)

    Expected value of multivariate hypergeometric distribution.

    n : number of items drawn.
    m : number of items in each category.
    """
    m= np.asarray(m, float)
    return n * (m / m.sum())


def pdf(x, m):
    R"""
    pdf(x, m)

    The multivariate hypergeometric describes the probability of drawing x[i]
    elements of the ith category, when the number of items in each category is
    given by m.


    .. math::
        \frac{\prod_i \binom{m_i}{x_i}}{\binom{N}{n}}

    where :math:`N = \sum_i m_i` and :math:`n = \sum_i x_i`.

    :Parameters:
      - `x` : [int sequence] Number of draws from each category, (x < m).
      - `m` : [int sequence] Number of items in each categoy.
    """
    return flib.mvhyperg(x, m)

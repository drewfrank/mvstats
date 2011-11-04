import flib
import numpy as np
from numpy import pi, inf
import itertools
import pdb
import warnings

# Multinomial----------------------------------------------
class multinomial(object):
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

    def pdf(x, n, p):
        R"""
        pdf(x, n, p)

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

# Multivariate hypergeometric------------------------------
class hypergeometric(object):
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


# Multivariate normal--------------------------------------
class normal(object):
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
        if (tau and cov) or (tau and sig) or (cov and sig):
            raise ValueError('Only specify one of {tau, cov, sig}.')
        if not tau and not cov and not sig:
            # Default the precision to identity matrix.
            tau = np.eye(mu.size)
        if tau:
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
        elif cov:
            mu_size = np.shape(mu)
            if size==1:
                return np.random.multivariate_normal(mu, cov, size).reshape(mu_size)
            else:
                return np.random.multivariate_normal(mu, cov, size).reshape((size,)+mu_size)
        elif sig:
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
        mv_normal_expval(mu, tau=None, cov=None, sig=None)

        Expected value of multivariate normal distribution.
        """
        return mu

    def pdf(x, mu, tau=None, cov=None, sig=None):
        R"""
        mv_normal_like(x, mu, tau=None, cov=None, sig=None)

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
        if (tau and cov) or (tau and sig) or (cov and sig):
            raise ValueError('Only specify one of {tau, cov, sig}.')
        if not tau and not cov and not sig:
            # Default the precision to identity matrix.
            tau = np.eye(mu.size)
        if tau:
            # TODO: Vectorize in Fortran
            if len(np.shape(x))>1:
                return np.sum([flib.prec_mvnorm(r,mu,tau) for r in x])
            else:
                return flib.prec_mvnorm(x,mu,tau)
        elif cov:
            # TODO: Vectorize in Fortran
            if len(np.shape(x))>1:
                return np.sum([flib.cov_mvnorm(r,mu,cov) for r in x])
            else:
                return flib.cov_mvnorm(x,mu,cov)
        elif sig:
            # TODO: Vectorize in Fortran
            if len(np.shape(x))>1:
                return np.sum([flib.chol_mvnorm(r,mu,sig) for r in x])
            else:
                return flib.chol_mvnorm(x,mu,sig)

# Wishart---------------------------------------------------
class wishart(object):
    def rvs(n, tau=None, cov=None):
        """
        rwishart(n, tau=None, cov=None)

        Return a Wishart random matrix.

        :Parameters:
          - `n` : Degrees of freedom.
          - `tau` : (k,k) Positive definite precision matrix.
          - `cov` : (k,k) Positive definite covariance matrix.
        """
        if not (tau or cov):
            raise ValueError('Must specify either tau or cov.')
        if tau:
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
        elif cov:
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
        if not (tau or cov):
            raise ValueError('Must specify either tau or cov.')
        if tau:
            return n * np.asarray(tau.I)
        elif cov:
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
        if not (tau or cov):
            raise ValueError('Must specify either tau or cov.')
        if tau:
            return flib.blas_wishart(X,n,tau)
        elif cov:
            return flib.blas_wishart_cov(X,n,cov)

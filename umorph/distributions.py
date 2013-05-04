import math
import random

class Multinomial(object):
    """Multinomial distribution sampled from a Dirichlet prior (collapsed/non-collapsed)"""
    def __init__(self, K, prior, collapsed=False):
        self.prior = prior
        self.K = K
        self.counts = [0] * K
        self.N = 0
        self.collapsed = collapsed

    def increment(self, k, c=1):
        assert (0 <= k < self.K)
        self.counts[k] += c
        self.N += c

    def decrement(self, k, c=1):
        assert (0 <= k < self.K)
        self.counts[k] -= c
        self.N -= c

    def prob(self, k):
        if self.collapsed:
            return self.marginal_prob(k)
        else:
            return self.theta[k]

    def marginal_prob(self, k):
        return (float(self.counts[k] + self.prior.alpha)
                / (self.N + self.K * self.prior.alpha))

    def update(self, counts):
        self.counts = [0]*self.K
        self.N = 0
        for k, c in counts.iteritems():
            self.counts[k] = c
            self.N += c

    def initialize(self):
        if not self.collapsed:
            self.theta = [1/float(self.K)] * self.K

    def resample(self):
        if not self.collapsed:
            self.theta = self.prior.sample(self.counts)

    def marginal_ll(self):
        return (math.lgamma(self.K * self.prior.alpha)
                - math.lgamma(self.K * self.prior.alpha + self.N)
                + sum(math.lgamma(self.counts[k] + self.prior.alpha)
                    for k in xrange(self.K))
                - self.K * math.lgamma(self.prior.alpha))

    def __repr__(self):
        if self.collapsed:
            return 'DirichletMultinomial(K={self.K}, N={self.N} | alpha={self.prior.alpha})'.format(self=self)
        else:
            return 'Multinomial(K={self.K}, N={self.N}) ~ {self.prior}'.format(self=self)


class Dirichlet(object):
    """A Dirichlet distribution for sampling multinomials"""
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, counts):
        params = [self.alpha + c for c in counts]
        sample = [random.gammavariate(a, 1) for a in params]
        norm = sum(sample)
        return [v/norm for v in sample]

    def __repr__(self):
        return 'Dirichlet(alpha={self.alpha})'.format(self=self)


class MultinomialProduct(object):
    """H(p, s) = theta_p(p) * theta_s(s)"""
    def __init__(self, n_prefixes, alpha_p, n_suffixes, alpha_s, collapsed=False):
        self.theta_p = Multinomial(n_prefixes, Dirichlet(alpha_p), collapsed)
        self.theta_s = Multinomial(n_suffixes, Dirichlet(alpha_s), collapsed)

    def increment(self, p, s):
        self.theta_p.increment(p)
        self.theta_s.increment(s)

    def decrement(self, p, s):
        self.theta_p.decrement(p)
        self.theta_s.decrement(s)

    def prob(self, p, s):
        return self.theta_p.prob(p) * self.theta_s.prob(s)
    
    def marginal_prob(self, p, s):
        return self.theta_p.marginal_prob(p) * self.theta_s.marginal_prob(s)

    def initialize(self):
        self.theta_p.initialize()
        self.theta_s.initialize()

    def resample(self):
        self.theta_p.resample()
        self.theta_s.resample()

    def update(self, p_counts, s_counts):
        self.theta_p.update(p_counts)
        self.theta_s.update(s_counts)

    def log_likelihood(self):
        return self.theta_p.marginal_ll() + self.theta_s.marginal_ll()

    def __repr__(self):
        return 'p ~ {self.theta_p}, s ~ {self.theta_s}'.format(self=self)

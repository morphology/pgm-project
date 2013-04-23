import random
import sys

class Multinomial:
    """Non-collapsed multinomial distribution sampled from a prior"""
    def __init__(self, K, prior):
        self.prior = prior
        self.K = K
        self.counts = [0]*K
        self.N = 0

    def increment(self, k, c=1):
        assert (0 <= k < self.K)
        self.counts[k] += c
        self.N += c

    def decrement(self, k, c=1):
        assert (0 <= k < self.K)
        self.counts[k] -= c
        self.N -= c

    def prob(self, k):
        return self.theta[k]

    def reset(self):
        self.counts = [0]*K
        self.N = 0

    def resample(self):
        self.theta = self.prior.sample(self.counts)

    def log_likelihood(self):
        return (math.lgamma(self.N + 1) + sum(c * math.log(self.theta[k]) - math.lgamma(c + 1)
            for k, c in enumerate(self.counts) if c > 0)
            + self.prior.log_likelihood(self.theta))

class Dirichlet:
    """A Dirichlet distribution for sampling multinomials"""
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, counts):
        params = [self.alpha + c for c in counts]
        sample = [random.gammavariate(a, 1) for a in params]
        norm = sum(sample)
        return [v/norm for v in sample]

    def log_likelihood(self, theta):
        K = len(theta)
        return (math.lgamma(K * self.alpha) - K * math.lgamma(self.alpha)
                + sum((self.alpha - 1) * math.log(t) for t in theta))

class MultProduct:
    """H(p, s) = theta_p(p) * theta_s(s)"""
    def __init__(self, n_prefixes, alpha_p, n_suffixes, alpha_s):
        self.theta_p = Multinomial(n_prefixes, Dirichlet(alpha_p))
        self.theta_s = Multinomial(n_suffixes, Dirichlet(alpha_s))

    def increment(self, p, s):
        self.theta_p.increment(p)
        self.theta_s.increment(s)

    def decrement(self, p, s):
        self.theta_p.decrement(p)
        self.theta_s.decrement(s)

    def prob(self, p, s):
        return self.theta_p.prob(p) * self.theta_s.prob(s)

    def resample(self):
        self.theta_p.resample()
        self.theta_s.resample()

    def reset(self):
        self.theta_p.reset()
        self.theta_s.reset()

    def update(self, p_counts, s_counts):
        for k, c in p_counts.items():
            self.theta_p.increment(k, c)

        for k, c in s_counts.items():
            self.theta_s.increment(k, c)

    def log_likelihood(self):
        #return self.theta_p.log_likelihood() + self.theta_s.log_likelihood()
        return 0 # numerical precision errors make LL computation impossible

    def __repr__(self):
        return 'MultProduct()'

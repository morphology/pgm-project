import math
import random
from vpyp.pyp import CRP
from vpyp.prob import mult_sample

def segmentations(word):
    for k in range(0, len(word)+1):
        yield word[:k], word[k:]

class Multinomial:
    """Non-collapsed multinomial distribution sampled from a prior"""
    def __init__(self, K, prior):
        self.prior = prior
        self.K = K
        self.counts = [0]*K
        self.N = 0

    def increment(self, k):
        assert (0 <= k < self.K)
        self.counts[k] += 1
        self.N += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.counts[k] -= 1
        self.N -= 1

    def prob(self, k):
        return self.theta[k]

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

    def log_likelihood(self):
        #return self.theta_p.log_likelihood() + self.theta_s.log_likelihood()
        return 0 # numerical precision errors make LL computation impossible

class SegmentationModel(CRP):
    """SegmentationModel ~ DP(alpha, H)"""
    def __init__(self, alpha, alpha_p, alpha_s, word_vocabulary,
            prefix_vocabulary, suffix_vocabulary):
        self.alpha = alpha
        self.base = MultProduct(len(prefix_vocabulary), alpha_p,
                len(suffix_vocabulary), alpha_s)
        self.word_vocabulary = word_vocabulary
        self.prefix_vocabulary = prefix_vocabulary
        self.suffix_vocabulary = suffix_vocabulary
        super(SegmentationModel, self).__init__()

    def _random_table(self, k):
        """Pick a table with dish k randomly"""
        n = random.randrange(0, self.ncustomers[k])
        tables = self.tables[k]
        for i, c in enumerate(tables):
            if n < c: return i
            n -= c

    def seating_probs(self, w, initialize=False):
        """Joint probabilities of all possible (segmentation, table assignments) of word #w"""
        for prefix, suffix in segmentations(self.word_vocabulary[w]):
            p = self.prefix_vocabulary[prefix]
            s = self.suffix_vocabulary[suffix]
            yield (p, s, -1), (1 if initialize else self.alpha * self.base.prob(p, s))
            if not (p, s) in self.tables: continue
            for seat, count in enumerate(self.tables[(p, s)]):
                yield (p, s, seat), (1 if initialize else count)

    def increment(self, w, initialize=False):
        """Sample a segmentation and a table assignment for word #w"""
        (p, s, seat) = mult_sample(self.seating_probs(w, initialize))
        if self._seat_to((p, s), seat):
            self.base.increment(p, s)
        return (p, s)

    def decrement(self, p, s):
        """Decrement the count for a (p, s) segmentation"""
        seat = self._random_table((p, s))
        if self._unseat_from((p, s), seat):
            self.base.decrement(p, s)

    def segmentation_probs(self, w):
        """Marginal probabilities of all segmentations of word #w"""
        for prefix, suffix in segmentations(self.word_vocabulary[w]):
            p = self.prefix_vocabulary[prefix]
            s = self.suffix_vocabulary[suffix]
            prob = self.alpha * self.base.prob(p, s) + self.ncustomers.get((p, s), 0)
            yield (p, s), prob

    def decode(self, w):
        """Compute the most likely segmentation (p, s) of word #w"""
        return max(self.segmentation_probs(w), key=lambda t:t[1])[0]

    def log_likelihood(self):
        return (math.lgamma(self.alpha) - math.lgamma(self.alpha + self.total_customers)
                + self.ntables * math.log(self.alpha) + self.base.log_likelihood())

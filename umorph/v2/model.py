import math
import random
from collections import Counter
from vpyp.pyp import CRP
from vpyp.prob import mult_sample
from umorph.segment import segmentations

class Multinomial:
    """Non-collapsed multinomial distribution sampled from a prior"""
    def __init__(self, K, prior):
        self.prior = prior
        self.K = K
        self.counts = [0] * K
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

    def marginal_prob(self, k):
        return ((self.counts[k] + self.prior.alpha)
                / (self.N + self.K * self.prior.alpha))

    def resample(self):
        self.theta = self.prior.sample(self.counts)

    def log_likelihood(self):
        return (math.lgamma(self.N + 1)
                + sum(c * math.log(self.theta[k]) - math.lgamma(c + 1)
                    for k, c in enumerate(self.counts) if c > 0)
                + self.prior.log_likelihood(self.theta))

    def marginal_log_likelihood(self):
        return (math.lgamma(self.K * self.prior.alpha)
                - math.lgamma(self.K * self.prior.alpha + self.N)
                + sum(math.lgamma(self.counts[k] + self.prior.alpha)
                    for k in xrange(self.K))
                - self.K * math.lgamma(self.prior.alpha))

    def __repr__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ {self.prior}'.format(self=self)

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

    def __repr__(self):
        return 'Dirichlet(alpha={self.alpha})'.format(self=self)

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

    def marginal_prob(self, p, s):
        return self.theta_p.marginal_prob(p) * self.theta_s.marginal_prob(s)

    def resample(self):
        self.theta_p.resample()
        self.theta_s.resample()

    def log_likelihood(self):
        # return self.theta_p.log_likelihood() + self.theta_s.log_likelihood()
        return (self.theta_p.marginal_log_likelihood()
                + self.theta_s.marginal_log_likelihood())

    def __repr__(self):
        return 'p ~ {self.theta_p}, s ~ {self.theta_s}'.format(self=self)

class SegmentationModel(CRP):
    """SegmentationModel ~ DP(alpha, H)"""
    def __init__(self, alpha, alpha_p, alpha_s, word_vocabulary,
            prefix_vocabulary, suffix_vocabulary):
        super(SegmentationModel, self).__init__()
        self.alpha = alpha
        self.base = MultProduct(len(prefix_vocabulary), alpha_p,
                len(suffix_vocabulary), alpha_s)
        self.word_vocabulary = word_vocabulary
        self.prefix_vocabulary = prefix_vocabulary
        self.suffix_vocabulary = suffix_vocabulary
        self.analyses = [Counter() for _ in xrange(len(word_vocabulary))]

    def segmentations(self, w):
        word = self.word_vocabulary[w]
        for prefix, suffix in segmentations(word):
            p = self.prefix_vocabulary[prefix]
            s = self.suffix_vocabulary[suffix]
            yield p, s

    def _random_table(self, k):
        """Pick a table with dish k randomly"""
        n = random.randrange(0, self.ncustomers[k])
        tables = self.tables[k]
        for i, c in enumerate(tables):
            if n < c: return i
            n -= c

    def seating_probs(self, w, initialize=False):
        """Joint probabilities of all possible (segmentation, table assignments) of word #w"""
        for p, s in self.segmentations(w):
            yield (p, s, -1), (1 if initialize
                    else self.alpha * self.base.prob(p, s))
            if not (w, p, s) in self.tables: continue
            for seat, count in enumerate(self.tables[w, p, s]):
                yield (p, s, seat), (1 if initialize else count)

    def increment(self, w, initialize=False):
        """Sample a segmentation and a table assignment for word #w"""
        # sample a table
        (p, s, seat) = mult_sample(self.seating_probs(w, initialize))
        # seat to the table
        if self._seat_to((w, p, s), seat):
            self.base.increment(p, s)
        # increment dish count
        self.analyses[w][p, s] += 1

    def decrement(self, w):
        """Decrement the count for a (p, s) segmentation of w"""
        # randomly choose a dish
        n = random.randrange(0, len(self.analyses[w]))
        for i, (p, s) in enumerate(self.analyses[w]):
            if n == i: break
        # randomly choose a table labeled with this dish
        try:
            seat = self._random_table((w, p, s))
        except KeyError as e:
            print w, self.analyses[w], e
            for (v, p, s), table in self.tables.iteritems():
                if v == w:
                    print p, s, table, self.ncustomers[v, p, s]
        # remove customer
        if self._unseat_from((w, p, s), seat):
            self.base.decrement(p, s)
        # decrement dish count
        self.analyses[w][p, s] -= 1
        if self.analyses[w][p, s] == 0:
            del self.analyses[w][p, s]

    def resample_labels(self):
        new_analyses = [Counter() for _ in xrange(len(self.word_vocabulary))]
        new_tables = {}
        new_ncustomers = {}
        for (w, old_p, old_s), tables in self.tables.iteritems():
            for c in tables:
                # remove (old_p, old_s)
                self.base.decrement(old_p, old_s)
                # resample
                (p, s) = mult_sample(((p, s), self.base.prob(p, s))
                        for p, s in self.segmentations(w))
                # add (p, s)
                if (w, p, s) not in new_tables:
                    new_tables[w, p, s] = []
                    new_ncustomers[w, p, s] = 0
                new_tables[w, p, s].append(c)
                new_ncustomers[w, p, s] += c
                new_analyses[w][p, s] += c
                self.base.increment(p, s)
        self.analyses = new_analyses
        self.tables = new_tables
        self.ncustomers = new_ncustomers

    def decode(self, w):
        """Compute the most likely segmentation (p, s) of word #w"""
        return max(self.segmentations(w), key=lambda ps: self.base.marginal_prob(*ps))

    def log_likelihood(self):
        """
        LL = \frac{\prod_{t \in tables} \alpha p_0(l_t) \times 1 \times \dots (c_t - 1)}{\prod_{k=0}^{N-1} (\alpha + k)}
           = p_{\text{base}} \times \frac{\Gamma(\alpha)}{\Gamma(\alpha + N)}\prod_{t \in tables} \alpha \times c_t!
        """
        return (math.lgamma(self.alpha)
                - math.lgamma(self.alpha + self.total_customers)
                + self.ntables * math.log(self.alpha)
                + sum(math.lgamma(c) for tables in self.tables.itervalues()
                    for c in tables)
                + self.base.log_likelihood())

    def __repr__(self):
        return ('SegmentationModel(alpha={self.alpha}, base={self.base}, '
                '#customers={self.total_customers}, #tables={self.ntables}, '
                '#dishes={V})').format(self=self, V=len(self.tables))

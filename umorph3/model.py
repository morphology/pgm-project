import logging
import math
import multiprocessing
import numpy as np
import random
from collections import Counter
from segment import segmentation_mapping
from slave import CRPSlave


class Multinomial(object):
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
        self.counts = [0]*self.K
        self.N = 0

    def resample(self):
        self.theta = self.prior.sample(self.counts)

    def log_likelihood(self):
        return (math.lgamma(self.N + 1) + sum(c * math.log(self.theta[k]) - math.lgamma(c + 1)
            for k, c in enumerate(self.counts) if c > 0)
            + self.prior.log_likelihood(self.theta))


class Dirichlet(object):
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


class MultinomialProduct(object):
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


class ParallelSegmentationModel(object):
    def __init__(self, alpha, alpha_p, alpha_s, corpus, w_vocabulary, p_vocabulary, s_vocabulary, n_processors):
        self.alpha = float(alpha)
        self.alpha_p = alpha_p
        self.alpha_s = alpha_s
        self.base = MultinomialProduct(len(p_vocabulary), alpha_p, len(s_vocabulary), alpha_s)
        self.base.resample()
        self.corpus = corpus
        self.word_vocabulary = w_vocabulary
        self.prefix_vocabulary = p_vocabulary
        self.suffix_vocabulary = s_vocabulary
        self.seg_mappings = segmentation_mapping(w_vocabulary, p_vocabulary, s_vocabulary)
        self.n_processors = n_processors

        self._slaves = []
        self._processor_indicators = [random.randrange(self.n_processors) for _ in self.corpus]
        for gid in xrange(self.n_processors):
            iq, oq = multiprocessing.Queue(), multiprocessing.Queue()
            s = CRPSlave(self.alpha/self.n_processors, self.base, self.seg_mappings, gid, iq, oq)
            self._slaves.append((s, iq, oq))
            s.start()
            words = [w for i, w in enumerate(self.corpus) if self._processor_indicators[i] == gid]
            iq.put(words)

    def resample(self, processors=False):
        """Run the sampler for the parallelized model."""
        for p, iq, _ in self._slaves:
            iq.put(self.base)

        self.base.reset()

        for p, _, oq in self._slaves:
            p_counts, s_counts = oq.get()
            self.base.update(p_counts, s_counts)

        self.base.resample()

        if processors:
            slave_tables = []
            for _, iq, oq in self._slaves:
                iq.put('send_tables')
                slave_tables.append(oq.get())

            new_tables = [[] for _ in slave_tables]
            for tables in slave_tables:
                for table in tables:
                    pi = random.randrange(self.n_processors)
                    new_tables[pi].append(table)

            new_ccs = [self._counts_of_counts(tables) for tables in new_tables]
            old_ccs = [self._counts_of_counts(tables) for tables in slave_tables]
            numer = sum(math.lgamma(v+1) for ccs in new_ccs for v in ccs.itervalues())
            denom = sum(math.lgamma(v+1) for ccs in old_ccs for v in ccs.itervalues())
            logging.info('%f - %f = %f', numer, denom, numer - denom)
            ratio = math.exp(numer - denom)
            logging.info('%f', ratio)
            accept_prob = min(1.0, ratio)
    
            accept = random.random() < accept_prob
            if accept:
                logging.info('Resampling processor indicators')

            for i, (_, iq, _) in enumerate(self._slaves):
                iq.put(accept)
                iq.put(new_tables[i])

    def _counts_of_counts(self, tables):
        return Counter(c for _, c in tables)

    def shutdown(self):
        """Shut down any resources used."""
        for p, iq, _ in self._slaves:
            iq.put(None)
            p.join()

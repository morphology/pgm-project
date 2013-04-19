import random
import logging
from vpyp.pyp import CRP
from vpyp.prob import mult_sample

class Multinomial:
    def __init__(self, K, prior):
        self.prior = prior
        self.theta = prior.sample([0]*K)

    def prob(self, k):
        return self.theta[k]

    def resample(self, counts):
        self.theta = self.prior.sample(counts)

class Dirichlet:
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, counts):
        params = [self.alpha + c for c in counts]
        sample = [random.gammavariate(a, 1) for a in params]
        norm = sum(sample)
        return [v/norm for v in sample]

def segmentations(word):
    for k in range(0, len(word)+1):
        yield word[:k], word[k:]

class LexiconModel:
    def __init__(self, n_prefixes, n_suffixes, alpha_s, alpha_p):
        self.theta_s = Multinomial(n_prefixes, Dirichlet(alpha_s))
        self.theta_p = Multinomial(n_suffixes, Dirichlet(alpha_p))

    def prob(self, k):
        return -1

    def resample(self, counts_s, counts_p):
        self.theta_s.update(counts_s)
        self.theta_p.update(counts_p)

class DP(CRP):
    def __init__(self, alpha, base, word_vocabulary, stem_vocabulary, postfix_vocabulary):
        self.alpha = alpha
        self.base = base
        self.word_vocabulary = word_vocabulary
        self.stem_vocabulary = stem_vocabulary
        self.postfix_vocabulary = postfix_vocabulary
        super(DP, self).__init__()


    def init(self, k):
        (seat, s, p) = mult_sample(self.seating_probs(k, True))
        self._seat_to(k, seat)
        return seat

    def resample(self, k, seat):
        self._unseat_from(k, seat)
        (new_seat, s, p) = mult_sample(self.seating_probs(k))
        self._seat_to(k, new_seat)
        return new_seat

    def seating_probs(self, k, init=False):
        for (stem, postfix) in segmentations(self.word_vocabulary[k]):
            s = self.stem_vocabulary[stem]
            p = self.postfix_vocabulary[postfix]
            yield (-1, s, p), (1 if init else self.alpha*self.base.prob((s, p)))
            if not (s, p) in self.tables: continue
            for seat, count in enumerate(self.tables[(s, p)]):
                yield (seat, s, p), (1 if init else count)

def run_sampler(model, n_iter, words):
    # initialize
    logging.info('Initializing')
    seatings = [model.init(word) for word in words]
    # loop
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        # 1. resample seat assignments and table labels given H
        for w in xrange(len(words)):
            seatings[w] = model.resample(words[w], seatings[w])
        # 2. resample H given table tables
        counts_s = [0]*len(model.stem_vocabulary)
        counts_p = [0]*len(model.postfix_vocabulary)
        for (s, p), tables in model.tables.iteritems():
            counts_s[s] += len(tables)
            counts_p[p] += len(tables)
        model.base.resample(counts_s, counts_p)

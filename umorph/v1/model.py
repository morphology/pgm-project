import random
from vpyp.prob import mult_sample, remove_random, SparseDirichletMultinomial
from vpyp.prior import GammaPrior
from vpyp.pyp import DP
from umorph.segment import segmentations

dirichlet_multinomial = lambda K, alpha: SparseDirichletMultinomial(K, GammaPrior(1, 1, alpha))
dirichlet_process = lambda base, strength: DP(base, GammaPrior(1, 1, strength))

class LexiconModel:
    def __init__(self, alpha_p, alpha_s,
            word_vocabulary, prefix_vocabulary, suffix_vocabulary):
        self.prefix_model = dirichlet_multinomial(len(prefix_vocabulary), alpha_p)
        self.suffix_model = dirichlet_multinomial(len(suffix_vocabulary), alpha_s)
        self.word_vocabulary = word_vocabulary
        self.prefix_vocabulary = prefix_vocabulary
        self.suffix_vocabulary = suffix_vocabulary
        self.analyses = [[] for _ in xrange(len(word_vocabulary))] # word -> [(p, s)]
    
    def increment(self, word, initialize=False):
        if initialize:
            p, s = random.choice([a for a, _ in self._analysis_probs(word)])
        else:
            p, s = mult_sample(self._analysis_probs(word))
        self.prefix_model.increment(p)
        self.suffix_model.increment(s)
        self.analyses[word].append((p, s))
    
    def decrement(self, word):
        p, s = remove_random(self.analyses[word])
        self.prefix_model.decrement(p)
        self.suffix_model.decrement(s)

    def prob(self, word):
        return sum(p for _, p in self._analysis_probs(word))

    def log_likelihood(self, full=False):
        return self.prefix_model.log_likelihood() + self.suffix_model.log_likelihood()

    def _analysis_probs(self, word):
        for prefix, suffix in segmentations(self.word_vocabulary[word]):
            p, s = self.prefix_vocabulary[prefix], self.suffix_vocabulary[suffix]
            p_split = self.prefix_model.prob(p) * self.suffix_model.prob(s)
            yield (p, s), p_split

    def decode_word(self, word ):
        return max(self._analysis_probs(word), key=lambda t: t[1])[0]

    def __repr__(self):
        return 'LexiconModel(prefix ~ {self.prefix_model}, suffix ~ {self.suffix_model})'.format(self=self)

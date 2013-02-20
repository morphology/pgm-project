from vpyp.prob import mult_sample, remove_random, DirichletMultinomial
from vpyp.prior import GammaPrior, PYPPrior
from vpyp.pyp import PYP

dirichlet_multinomial = lambda K, alpha: DirichletMultinomial(K, GammaPrior(1, 1, alpha))
pyp = lambda base, discount, strength: PYP(base, PYPPrior(1, 1, 1, 1, discount, strength))

def word_splits(word):
    # allow NULL suffix but not prefix
    for k in range(1, len(word)+1):
        yield word[:k], word[k:]

class LexiconModel:
    def __init__(self, n_classes, alpha_c, alpha_t, alpha_f,
            word_vocabulary, stem_vocabulary, suffix_vocabulary):
        self.class_model = dirichlet_multinomial(n_classes, alpha_c)
        self.stem_models = [dirichlet_multinomial(len(stem_vocabulary), alpha_t)
                for _ in xrange(n_classes)]
        self.suffix_models = [dirichlet_multinomial(len(suffix_vocabulary), alpha_t)
                for _ in xrange(n_classes)]
        self.word_vocabulary = word_vocabulary
        self.stem_vocabulary = stem_vocabulary
        self.suffix_vocabulary = suffix_vocabulary
        self.analyses = [[] for _ in range(len(word_vocabulary))] # word -> [(c, t, f)]
    
    def increment(self, word):
        c, t, f = mult_sample(self._analysis_probs(word))
        self.class_model.increment(c)
        self.stem_models[c].increment(t)
        self.suffix_models[c].increment(f)
        self.analyses[word].append((c, t, f))
    
    def decrement(self, word):
        c, t, f = remove_random(self.analyses[word])
        self.class_model.decrement(c)
        self.stem_models[c].decrement(t)
        self.suffix_models[c].decrement(f)

    def prob(self, word):
        return sum(p for _, p in self._analysis_probs(word))

    def log_likelihood(self, full=False):
        return (self.class_model.log_likelihood()
                + sum(m.log_likelihood() for m in self.stem_models)
                + sum(m.log_likelihood() for m in self.suffix_models))

    def _analysis_probs(self, word):
        for c in range(self.class_model.K):
            p_class = self.class_model.prob(c)
            # note: could replace (stem, suffix) by the split position
            for stem, suffix in word_splits(self.word_vocabulary[word]):
                t, f = self.stem_vocabulary[stem], self.suffix_vocabulary[suffix]
                p_split = (p_class * self.stem_models[c].prob(t)
                        * self.suffix_models[c].prob(f))
                yield (c, t, f), p_split

    def decode_word(self, word):
        return max(self._analysis_probs(word), key=lambda t: t[1])[0]

    def __repr__(self):
        return ('LexiconModel(c ~ {self.class_model}); '
                't, f | c ~ Mult ~ Dir').format(self=self, n_classes=len(self.stem_models))

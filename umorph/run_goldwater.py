#!/usr/bin/env python
import logging
import math
import sys
import time
from vpyp.corpus import Vocabulary
from golwater import pyp, LexiconModel, word_splits

n_classes = 6
alpha_c = 0.5
alpha_t = 0.001
alpha_f = 0.001
d, theta = 0.1, 1e-6
n_iter = 1000

def run_sampler(model, corpus, n_iter):
    for it in xrange(n_iter):
        for w in corpus:
            if it > 0: model.decrement(w)
            model.increment(w)
        if it % 10 == 9:
            logging.info('Iteration %d/%d', it+1, n_iter)
            ll = model.log_likelihood(full=True)
            ppl = math.exp(-ll/len(corpus))
            logging.info('LL=%.0f ppl=%.0f', ll, ppl)
            logging.info('Model: %s', model)

def show_analyses(model):
    for w, word in enumerate(model.word_vocabulary):
        c, t, f = model.decode_word(w)
        stem = model.stem_vocabulary[t]
        suffix = model.suffix_vocabulary[f]
        print(u'{}\t{}\t{}\t{}'.format(word, c, stem, suffix).encode('utf8'))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Read the training corpus
    word_vocabulary = Vocabulary(start_stop=False)
    corpus = [word_vocabulary[line.decode('utf8').strip()] for line in sys.stdin]

    # Compute all the possible stems
    stems = set(stem for word in word_vocabulary for stem, suffix in word_splits(word))
    stem_vocabulary = Vocabulary(start_stop=False, init=stems)

    # Compute all the possible suffixes
    suffixes = set(suffix for word in word_vocabulary for stem, suffix in word_splits(word))
    suffix_vocabulary = Vocabulary(start_stop=False, init=suffixes)

    logging.info('%d tokens / %d types / %d stems / %d suffixes',
            len(corpus), len(word_vocabulary), len(stem_vocabulary), len(suffix_vocabulary))

    model = pyp(LexiconModel(n_classes, alpha_c, alpha_t, alpha_f,
            word_vocabulary, stem_vocabulary, suffix_vocabulary), d, theta)

    # Run the Gibbs sampler
    t_start = time.time()
    run_sampler(model, corpus, n_iter)
    t_end = time.time()
    runtime = t_end - t_start
    print('Sampler ran for {:.3f} seconds'.format(runtime))

    # Print out the most likely analysis for each word in the vocabulary
    show_analyses(model.base)

if __name__ == '__main__':
    main()

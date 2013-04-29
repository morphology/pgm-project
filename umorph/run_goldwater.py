#!/usr/bin/env python
import argparse
import logging
import math
import sys
import time
from vpyp.corpus import Vocabulary
from golwater import dirichlet_process, LexiconModel, word_splits

def run_sampler(model, corpus, n_iter):
    for w in corpus:
        model.increment(w, initialize=True)
    for it in xrange(n_iter):
        # PYP sampling
        for w in corpus:
            model.decrement(w)
            model.increment(w)
        # base sampling
        try:
            model.resample_base()
        except AttributeError:
            pass
        if it % 10 == 9:
            logging.info('Iteration %d/%d', it+1, n_iter)
            ll = model.log_likelihood(full=True)
            ppl = math.exp(-ll/len(corpus))
            logging.info('LL=%.0f ppl=%.0f', ll, ppl)
            logging.info('Model: %s', model)

def show_analyses(model, marginalize):
    for w, word in enumerate(model.word_vocabulary):
        if marginalize:
            t, f = model.decode_word(w, marginalize=True)
            c = '_'
        else:
            c, t, f = model.decode_word(w)
        stem = model.stem_vocabulary[t]
        suffix = model.suffix_vocabulary[f]
        print(u'{}\t{}\t{}\t{}'.format(word, c, stem, suffix).encode('utf8'))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Run Golwater model')
    parser.add_argument('-i', '--n_iter', type=int, required=True,
            help='Number of iterations')
    parser.add_argument('-k', '--n_classes', type=int, required=True,
            help='Number of latent classes')
    parser.add_argument('--alpha_c', '-ac', type=float, default=0.5,
            help='Smoothing parameter for class Dirichlet prior')
    parser.add_argument('--alpha_t', '-at', type=float, default=0.001,
            help='Smoothing parameter for stem Dirichlet prior')
    parser.add_argument('--alpha_f', '-af', type=float, default=0.001,
            help='Smoothing parameter for suffix Dirichlet prior')
    parser.add_argument('--strength', '-t', type=float, default=1e-6,
            help='DP prior stength')
    parser.add_argument('--types', action='store_true',
            help='Run model on types instead of tokens')
    parser.add_argument('--marginalize', action='store_true',
            help='Marginalize latent class when decoding')
    args = parser.parse_args()

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

    lexicon_model = LexiconModel(args.n_classes, args.alpha_c, args.alpha_t, args.alpha_f,
            word_vocabulary, stem_vocabulary, suffix_vocabulary) # generator
    if args.types:
        model = lexicon_model
        corpus = range(len(word_vocabulary)) # corpus = lexicon
    else:
        model = dirichlet_process(lexicon_model, args.strength) # adaptor

    # Run the Gibbs sampler
    t_start = time.time()
    run_sampler(model, corpus, args.n_iter)
    t_end = time.time()
    runtime = t_end - t_start
    logging.info('Sampler ran for {:.3f} seconds'.format(runtime))

    # Print out the most likely analysis for each word in the vocabulary
    show_analyses(lexicon_model, args.marginalize)

if __name__ == '__main__':
    main()

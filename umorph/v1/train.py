#!/usr/bin/env python
import argparse
import logging
import heapq
import sys
from vpyp.corpus import Vocabulary
from model import dirichlet_process, LexiconModel
from umorph.segment import affixes

def show_top(model):
    top_prefixes = heapq.nlargest(10, model.prefix_model.count.iteritems(), key=lambda t:t[1])
    n_prefixes = sum(1 for c in model.prefix_model.count.itervalues() if c > 0)
    logging.info('Top prefixes (10/%d): %s', n_prefixes, ' '.join(model.prefix_vocabulary[prefix]+':'+str(c)
        for prefix, c in top_prefixes))
    top_suffixes = heapq.nlargest(10, model.suffix_model.count.iteritems(), key=lambda t:t[1])
    n_suffixes = sum(1 for c in model.suffix_model.count.itervalues() if c > 0)
    logging.info('Top suffixes (10/%d): %s', n_suffixes, ' '.join(model.suffix_vocabulary[suffix]+':'+str(c)
        for suffix, c in top_suffixes))

def run_sampler(model, corpus, n_iter):
    for w in corpus:
        model.increment(w, initialize=True)
    for it in xrange(n_iter):
        # PYP sampling
        for w in corpus:
            model.decrement(w)
            model.increment(w)
        # base sampling
        model.resample_base()
        if it % 10 == 9:
            logging.info('Iteration %d/%d', it+1, n_iter)
            crp_ll = model.log_likelihood()
            base_ll = model.base.log_likelihood()
            logging.info('LL=%.0f\tCRPLL=%.0f\tBaseLL=%.0f', crp_ll+base_ll, crp_ll, base_ll)
            logging.info('Model: %s', model)
            show_top(model.base)

def show_analyses(model):
    for w, word in enumerate(model.word_vocabulary):
        p, s = model.decode_word(w)
        prefix = model.prefix_vocabulary[p]
        suffix = model.suffix_vocabulary[s]
        print(u'{}\t{}\t{}'.format(word, prefix, suffix).encode('utf8'))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Run Golwater model')
    parser.add_argument('-i', '--n_iter', type=int, required=True,
            help='Number of iterations')
    parser.add_argument('--alpha_p', type=float, default=0.001,
            help='Smoothing parameter for prefix Dirichlet prior')
    parser.add_argument('--alpha_s', type=float, default=0.001,
            help='Smoothing parameter for suffix Dirichlet prior')
    parser.add_argument('--strength', type=float, default=1e-6,
            help='DP prior stength')
    args = parser.parse_args()

    # Read the training corpus
    word_vocabulary = Vocabulary(start_stop=False)
    corpus = [word_vocabulary[line.decode('utf8').strip()] for line in sys.stdin]

    # Compute all the possible prefixes, suffixes
    prefix_vocabulary, suffix_vocabulary = affixes(word_vocabulary)

    logging.info('%d tokens / %d types / %d prefixes / %d suffixes',
            len(corpus), len(word_vocabulary), len(prefix_vocabulary), len(suffix_vocabulary))

    lexicon_model = LexiconModel(args.alpha_p, args.alpha_s,
            word_vocabulary, prefix_vocabulary, suffix_vocabulary) # generator
    model = dirichlet_process(lexicon_model, args.strength) # adaptor

    # Run the Gibbs sampler
    run_sampler(model, corpus, args.n_iter)

    # Print out the most likely analysis for each word in the vocabulary
    show_analyses(lexicon_model)

if __name__ == '__main__':
    main()

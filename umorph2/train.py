import sys
import argparse
import logging
import math
import heapq
import time
from itertools import izip
from vpyp.corpus import Vocabulary
from model import segmentations, SegmentationModel

def show_top(model):
    top_prefixes = heapq.nlargest(10, izip(model.base.theta_p.counts, model.prefix_vocabulary))
    n_prefixes = sum(1 for c in model.base.theta_p.counts if c > 0)
    logging.info('Top prefixes (10/%d): %s', n_prefixes, ' '.join(prefix+':'+str(c) for c, prefix in top_prefixes))
    top_suffixes = heapq.nlargest(10, izip(model.base.theta_s.counts, model.suffix_vocabulary))
    n_suffixes = sum(1 for c in model.base.theta_s.counts if c > 0)
    logging.info('Top suffixes (10/%d): %s', n_suffixes, ' '.join(suffix+':'+str(c) for c, suffix in top_suffixes))

def run_sampler(model, n_iter, words):
    logging.info('Initializing')
    # Initialize H
    model.base.resample()
    # Initialize G (labels & assignments)
    for w in words:
        model.increment(w, initialize=True)
    for it in xrange(n_iter):
        if it % 10 == 0:
            logging.info('Iteration %d/%d', it+1, n_iter)
            LL = model.log_likelihood()
            ppl = math.exp(-LL/len(words))
            logging.info('LL=%.0f ppl=%.0f', LL, ppl)
            show_top(model)
        # 1. resample seat assignments given labels, H
        for w in words:
            model.decrement(w)
            model.increment(w)
        # 2. resample table labels given H
        model.resample_labels()
        # 3. resample H given table labels
        model.base.resample()


def show_analyses(model):
    for w, word in enumerate(model.word_vocabulary):
        p, s = model.decode(w)
        prefix = model.prefix_vocabulary[p]
        suffix = model.suffix_vocabulary[s]
        print(u'{}\t_\t{}\t{}'.format(word, prefix, suffix).encode('utf8'))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('-i', '--n_iter', type=int, required=True,
            help='Number of iterations')
    parser.add_argument('--alpha_p', type=float, default=0.001,
            help='Smoothing parameter for prefix Dirichlet prior')
    parser.add_argument('--alpha_s', type=float, default=0.001,
            help='Smoothing parameter for suffix Dirichlet prior')
    parser.add_argument('--strength', '-t', type=float, default=1e-6,
            help='DP prior stength')
    args = parser.parse_args()

    word_vocabulary = Vocabulary(start_stop=False)
    corpus = [word_vocabulary[line.decode('utf8').strip()] for line in sys.stdin]

    # Compute all the possible prefixes
    prefixes = set(prefix for word in word_vocabulary for prefix, suffix in segmentations(word))
    prefix_vocabulary = Vocabulary(start_stop=False, init=prefixes)

    # Compute all the possible suffixes
    suffixes = set(suffix for word in word_vocabulary for prefix, suffix in segmentations(word))
    suffix_vocabulary = Vocabulary(start_stop=False, init=suffixes)

    logging.info('%d tokens / %d types / %d prefixes / %d suffixes',
            len(corpus), len(word_vocabulary), len(prefix_vocabulary), len(suffix_vocabulary))

    model = SegmentationModel(args.strength, args.alpha_p, args.alpha_s,
            word_vocabulary, prefix_vocabulary, suffix_vocabulary)

    t_start = time.time()
    run_sampler(model, args.n_iter, corpus)
    runtime = time.time() - t_start
    logging.info('Sampler ran for {:.3f} seconds', runtime)

    show_analyses(model)

if __name__ == '__main__':
    main()

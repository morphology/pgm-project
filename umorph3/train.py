import argparse
import heapq
import logging
import multiprocessing as mp
import random
import segment
import sys
import time
from model import ParallelSegmentationModel
from itertools import izip
from vpyp.corpus import Vocabulary


def show_top(model):
    top_prefixes = heapq.nlargest(10, izip(model.base.theta_p.counts, model.prefix_vocabulary))
    n_prefixes = sum(1 for c in model.base.theta_p.counts if c > 0)
    logging.info('Top prefixes (10/%d): %s', n_prefixes, ' '.join(prefix+':'+str(c) for c, prefix in top_prefixes))
    top_suffixes = heapq.nlargest(10, izip(model.base.theta_s.counts, model.suffix_vocabulary))
    n_suffixes = sum(1 for c in model.base.theta_s.counts if c > 0)
    logging.info('Top suffixes (10/%d): %s', n_suffixes, ' '.join(suffix+':'+str(c) for c, suffix in top_suffixes))


def run_sampler(model, n_iter):
    for it in xrange(n_iter):

        processors = True if it % 10 == 0 else False
        model.resample(processors)
        if processors:
            logging.info('MH Acceptance Rate= %f', model.acceptance_rate())

        if it % 10 == 0:
            logging.info('Iteration %d/%d', it+1, n_iter)
            show_top(model)


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
    parser.add_argument('--processors', '-p', type=int, default=mp.cpu_count(),
                        help='Number of processors to use')
    args = parser.parse_args()

    word_vocabulary = Vocabulary(start_stop=False)
    corpus = [word_vocabulary[line.decode('utf8').strip()] for line in sys.stdin if len(line.decode('utf8').strip()) > 1]
    prefix_vocabulary, suffix_vocabulary = segment.affixes(word_vocabulary)

    logging.info('%d tokens / %d types / %d prefixes / %d suffixes',
                 len(corpus), len(word_vocabulary), len(prefix_vocabulary), len(suffix_vocabulary))

    logging.info('Starting %d processes', args.processors)
    model = ParallelSegmentationModel(args.strength, args.alpha_p, args.alpha_s, corpus, word_vocabulary,
                                      prefix_vocabulary, suffix_vocabulary, args.processors)

    t_start = time.time()
    run_sampler(model, args.n_iter)
    runtime = time.time() - t_start
    logging.info('Sampler ran for {:.3f} seconds', runtime)

    model.shutdown()

    for word in word_vocabulary:
        p, s = model.decode(word)
        print(u'{}\t_\t{}\t{}'.format(word, p, s).encode('utf8'))


if __name__ == '__main__':
    main()

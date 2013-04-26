import argparse
import heapq
import logging
import multiprocessing as mp
import random
import segment
import sys
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

        if it % 10 == 0:
            logging.info('Iteration %d/%d', it+1, n_iter)
            show_top(model)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--processors', help='number of slaves to use', type=int, default=4)
    args = parser.parse_args()
    n_processors = args.processors if args.processors else mp.cpu_count()

    word_vocabulary = Vocabulary(start_stop=False)
    corpus = [word_vocabulary[line.decode('utf8').strip()] for line in sys.stdin
              if len(line.decode('utf8').strip()) > 1]
    prefix_vocabulary, suffix_vocabulary = segment.affixes(word_vocabulary)

    logging.info('%d tokens / %d types / %d prefixes / %d suffixes',
                 len(corpus), len(word_vocabulary), len(prefix_vocabulary), len(suffix_vocabulary))

    logging.info('Starting %d processes', n_processors)
    model = ParallelSegmentationModel(0.1, 1e-3, 1e-3, corpus, word_vocabulary,
                                      prefix_vocabulary, suffix_vocabulary, n_processors)

    run_sampler(model, 1000)

    model.shutdown()

    for word in word_vocabulary:
        p, s = model.decode(word)
        print(u'{}\t_\t{}\t{}'.format(word, p, s))


if __name__ == '__main__':
    main()

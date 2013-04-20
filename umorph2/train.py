import sys
import logging
import math
import heapq
from itertools import izip
from vpyp.corpus import Vocabulary
from model import segmentations, SegmentationModel

def show_top(model):
    top_prefixes = heapq.nlargest(10, izip(model.base.theta_p.counts, model.prefix_vocabulary))
    n_prefixes = sum(1 for c in model.base.theta_p.counts if c > 0)
    logging.info('Top prefixes (10/%d): %s', n_prefixes, ' '.join(prefix for _, prefix in top_prefixes))
    top_suffixes = heapq.nlargest(10, izip(model.base.theta_s.counts, model.suffix_vocabulary))
    n_suffixes = sum(1 for c in model.base.theta_s.counts if c > 0)
    logging.info('Top suffixes (10/%d): %s', n_suffixes, ' '.join(suffix for _, suffix in top_suffixes))

def run_sampler(model, n_iter, words):
    logging.info('Initializing')
    model.base.resample() # Initialize H
    analyses = [model.increment(word, initialize=True) for word in words] # Initialize G, (p, s)
    for it in xrange(n_iter):
        if it % 10 == 0:
            logging.info('Iteration %d/%d', it+1, n_iter)
            LL = model.log_likelihood()
            ppl = math.exp(-LL/len(words))
            logging.info('LL=%.0f ppl=%.0f', LL, ppl)
            show_top(model)
        # 1. resample seat assignments and table labels given H
        for w in xrange(len(words)):
            model.decrement(*analyses[w])
            analyses[w] = model.increment(words[w])
        # 2. resample H given table labels
        model.base.resample()


def show_analyses(model):
    for w, word in enumerate(model.word_vocabulary):
        p, s = model.decode(w)
        prefix = model.prefix_vocabulary[p]
        suffix = model.suffix_vocabulary[s]
        print(u'{}\t_\t{}\t{}'.format(word, prefix, suffix).encode('utf8'))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

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

    model = SegmentationModel(0.1, 1e-6, 1e-6, word_vocabulary,
            prefix_vocabulary, suffix_vocabulary)

    run_sampler(model, 1000, corpus)

    show_analyses(model)

if __name__ == '__main__':
    main()

import sys
import logging
from vpyp.corpus import Vocabulary
from model import segmentations, SegmentationModel

def run_sampler(model, n_iter, words):
    logging.info('Initializing')
    analyses = [model.increment(word, initialize=True) for word in words]
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        # 1. resample seat assignments and table labels given H
        for w in xrange(len(words)):
            model.decrement(*analyses[w])
            analyses[w] = model.increment(words[w])
        # 2. resample H given table tables
        counts_p = [0]*len(model.prefix_vocabulary)
        counts_s = [0]*len(model.suffix_vocabulary)
        for (p, s), tables in model.tables.iteritems():
            counts_p[p] += len(tables)
            counts_s[s] += len(tables)
        model.base.resample(counts_p, counts_s)

def main():
    logging.basicConfig(level=logging.INFO)

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

if __name__ == '__main__':
    main()

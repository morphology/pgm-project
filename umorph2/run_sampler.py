import sys
import logging
from vpyp.corpus import Vocabulary
from train import segmentations, SegmentationModel, run_sampler

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

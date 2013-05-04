import sys
import logging
from collections import Counter
from vpyp.corpus import Vocabulary
from umorph.segment import affixes
from umorph.distributions import MultinomialProduct

def main():
    logging.basicConfig(level=logging.INFO)
    # Read the training corpus
    word_vocabulary = Vocabulary(start_stop=False)
    analyses = {}
    for line in sys.stdin:
        word, analysis, _ = line.decode('utf8').split('\t')
        morphemes = analysis.split('+')
        if len(morphemes) not in (1, 2):
            raise Exception('wtf?')
        prefix = morphemes[0]
        suffix = '' if len(morphemes) == 1 else morphemes[1]
        word_vocabulary[word]
        analyses[word] = (prefix, suffix)

    # Compute all the possible prefixes, suffixes
    prefix_vocabulary, suffix_vocabulary = affixes(word_vocabulary)

    logging.info('%d types / %d prefixes / %d suffixes',
            len(word_vocabulary), len(prefix_vocabulary), len(suffix_vocabulary))

    prefix_counts = Counter()
    suffix_counts = Counter()
    for word, (prefix, suffix) in analyses.iteritems():
        prefix_counts[prefix_vocabulary[prefix]] += 1
        suffix_counts[suffix_vocabulary[suffix]] += 1

    ## The base
    base = MultinomialProduct(len(prefix_vocabulary), 0.001, len(suffix_vocabulary), 0.001)
    ## Updating the counts
    base.update(prefix_counts, suffix_counts)

    print base.log_likelihood()

if __name__ == "__main__":
    main()

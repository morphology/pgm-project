from vpyp.corpus import Vocabulary

def segmentations(word):
    for k in range(1, len(word)+1):
        yield word[:k], word[k:]

def affixes(words):
    prefixes, suffixes = zip(*[seg for w in words for seg in segmentations(w)])
    prefixes = Vocabulary(start_stop=False, init=set(prefixes))
    suffixes = Vocabulary(start_stop=False, init=set(suffixes))
    return prefixes, suffixes

def segmentation_mapping(vocab, prefixes, suffixes):
    mapping = {}
    for w in vocab:
        segs = segmentations(w)
        segs = [(prefixes[p], suffixes[s]) for p, s in segs]
        mapping[vocab[w]] = set(segs)
    return mapping

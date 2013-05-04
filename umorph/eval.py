#!/usr/bin/env python
import argparse

special = {
        ('be', 'ed', 'was'): ('was', ''),
        ('be', 'ed', 'were'): ('were', ''),
        ('be', '', 'are'): ('are', ''),
        ('be', '', 'am'): ('am', ''),
        ('be', 's', 'is'): ('is', ''),
        ('have', 's', 'has'): ('has', ''),
        ('have', 'ed', 'had'): ('had', ''),
        ('do', 'en', 'done'): ('done', ''),
        ('do', 'ed', 'did'): ('did', ''),
        ('do', 's', 'does'): ('doe', 's'),
        ('go', 's', 'goes'): ('goe', 's'),
        ('go', 'en', 'gone'): ('gone', ''),
        ('go', 'ed', 'went'): ('went', ''),
}

def make_split(stem, suffix, word):
    if stem+suffix == word: # support+ed
        return stem, suffix
    m = special.get((stem, suffix, word))
    if m: return m
    if stem[-1] == 'e' and suffix in ('ing', 'ed', 'en'): # wrestle+ing
        return stem[:-1], suffix
    if (suffix == 's' and (stem[-1] in ('s', 'x', 'z') # address+s
            or stem.endswith('ch') or stem.endswith('sh'))):
        return stem, 'e'+suffix
    if stem[-1] == 'y':
        if suffix == 'ed':
            return stem[:-1]+'i', 'ed'
        elif suffix == 's':
            return stem[:-1]+'i','es'
    if suffix in ('ed', 'ing') and stem+stem[-1]+suffix == word: # spot+ed
        return (stem+stem[-1], suffix)
    return (stem, suffix)

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation.')
    parser.add_argument('analysis')
    parser.add_argument('gold_standard')
    args = parser.parse_args()

    analysis = {}
    with open(args.analysis) as f:
        for line in f:
            word, stem, suffix = line.decode('utf8')[:-1].split('\t')
            analysis[word] = (stem, suffix)

    types = set()
    n_tokens = 0
    token_errors = 0
    type_errors = 0
    with open(args.gold_standard) as f:
        for line in f:
            word, morph, pos = line.decode('utf8')[:-1].split('\t')
            morphemes = morph.split('+')
            if len(morphemes) not in (1, 2): continue # should not happen
            stem = morphemes[0]
            suffix = '' if len(morphemes) == 1 else morphemes[1]
            stem, suffix = make_split(stem, suffix, word)
            if stem+suffix != word: # irregular forms (said, rose, made...)
                stem, suffix = word, ''
            error = (1 if analysis[word] != (stem, suffix) else 0)
            n_tokens += 1
            token_errors += error
            if word not in types:
                types.add(word)
                type_errors += error
    
    n_types = len(types)
    print('{} tokens'.format(n_tokens))
    print('{:.2%} correct'.format(1-token_errors/float(n_tokens)))
    print('{} types'.format(n_types))
    print('{:.2%} correct'.format(1-type_errors/float(n_types)))

if __name__ == '__main__':
    main()

#!/usr/bin/env python
import sys
from vpyp.corpus import Vocabulary

def main():
    word_vocabulary = Vocabulary()
    for word, pos in (line.split() for line in sys.stdin):
        word_vocabulary[word]
    for word in word_vocabulary:
        print('{}\t{}\t{}\t{}'.format(word, '-1', word, ''))

if __name__ == '__main__':
    main()


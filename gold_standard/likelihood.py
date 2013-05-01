import sys
from dist import MultinomialProduct
from collections import Counter

def type_log_likelihood(corpus):
  analyses = [tuple(line.split('\t')[1].split('+')) if line.find('+')>0 else (line.split('\t')[1],'') for line in corpus]
  ## Preparing base
  prefixes, suffixes = zip(*analyses)
  ## Taking types
  prefixes = set(prefixes)
  suffixes = set(suffixes)
  ## The base
  base = MultinomialProduct(len(prefixes), 0.001, len(suffixes), 0.001)
  base.resample()
  ## Updating the counts
  base.update(Counter(range(len(prefixes))), Counter(range(len(suffixes))))
  print base.log_likelihood()

def main():
  corpus = [line.decode('utf8').strip() for line in sys.stdin if len(line.decode('utf8').strip()) > 1]
  type_log_likelihood(corpus)

if __name__ == "__main__":
  main()

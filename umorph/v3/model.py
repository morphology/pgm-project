import logging
import math
import multiprocessing
import random
from collections import Counter
from umorph.segment import segmentation_mapping
from umorph.distributions import MultinomialProduct
from slave import CRPSlave

class Message:
    def __init__(self, name, *args):
        logging.debug('Message: %s', name)
        self.name = name
        self.args = args

class ParallelSegmentationModel(object):
    def __init__(self, alpha, alpha_p, alpha_s, corpus, w_vocabulary, p_vocabulary, s_vocabulary, n_processors, n_mh, collapsed):
        self.alpha = float(alpha)
        self.base = MultinomialProduct(len(p_vocabulary), alpha_p,
                len(s_vocabulary), alpha_s, collapsed)
        self.corpus = corpus
        self.word_vocabulary = w_vocabulary
        self.prefix_vocabulary = p_vocabulary
        self.suffix_vocabulary = s_vocabulary
        self.seg_mappings = segmentation_mapping(w_vocabulary, p_vocabulary, s_vocabulary)
        self.n_processors = n_processors
        self.n_mh = n_mh

        self._slaves = []
        for gid in xrange(self.n_processors):
            iq, oq = multiprocessing.Queue(), multiprocessing.Queue()
            s = CRPSlave(self.alpha/self.n_processors, self.seg_mappings, gid, iq, oq)
            self._slaves.append((s, iq, oq))
            s.start()

    def initialize(self):
        # Initialize H
        self.base.initialize()

        # Send tokens to processors (initialize G)
        processor_indicators = [random.randrange(self.n_processors) for _ in self.corpus]
        for gid, (_, iq, _) in enumerate(self._slaves):
            words = [w for i, w in enumerate(self.corpus) if processor_indicators[i] == gid]
            iq.put(Message('init_tokens', words))

        # Update H
        self.update_base()

    def update_base(self):
        # Receive and aggregate counts
        total_p_counts = Counter()
        total_s_counts = Counter()
        for _, iq, oq in self._slaves:
            iq.put(Message('send_counts'))
            p_counts, s_counts = oq.get()
            total_p_counts += p_counts
            total_s_counts += s_counts

        # Update the base counts
        self.base.update(total_p_counts, total_s_counts)

        # Resample the base
        self.base.resample()

    def resample(self, processors=False):
        """Run the sampler for the parallelized model."""
        # Send H to slaves
        for _, iq, _ in self._slaves:
            iq.put(Message('update_base', self.base))

        # Each slave: resample CRP
        for _, iq, _ in self._slaves:
            iq.put(Message('resample'))

        # Update H
        self.update_base()

        if processors:
            self.resample_assignments()

    def mh_step(self, old_tables):
        new_tables = [[] for _ in old_tables]
        for tables in old_tables:
            for table in tables:
                pi = random.randrange(self.n_processors)
                new_tables[pi].append(table)

        new_ccs = [self._counts_of_counts(tables) for tables in new_tables]
        old_ccs = [self._counts_of_counts(tables) for tables in old_tables]
        numer = sum(math.lgamma(v+1) for ccs in new_ccs for v in ccs.itervalues())
        denom = sum(math.lgamma(v+1) for ccs in old_ccs for v in ccs.itervalues())

        ratio = math.exp(numer - denom)
        accept_prob = min(1.0, ratio)

        accept = random.random() < accept_prob

        return accept, (new_tables if accept else old_tables)

    def resample_assignments(self):
        # 1. Collect tables
        tables = [] # [[(dish, count), ...], ...]
        for _, iq, oq in self._slaves:
            iq.put(Message('send_tables'))
            tables.append(oq.get())

        # 2. Resample processor assignments
        mh_steps = 0.0
        mh_accepts = 0.0
        for mh_step in xrange(self.n_mh):
            accept, tables = self.mh_step(tables)
            mh_steps += 1
            mh_accepts += accept

        acceptance_rate = mh_accepts/mh_steps

        # 3. Send new table assignments to slaves
        for i, (_, iq, _) in enumerate(self._slaves):
            iq.put(Message('receive_tables', tables[i]))

        # Write log-likelihood
        total_customers = sum(sum(c for _, c in tables) for tables in tables)
        n_tables = sum(len(tables) for tables in tables)
        n_dishes = len(set(dish for tables in tables for dish, _ in tables))

        logging.info('MH Acceptance Rate: %f', acceptance_rate)
        logging.info('LL= %.0f\tCRPLL= %.0f\tBaseLL= %.0f', *self._log_likelihood(*tables))
        logging.info(('ParallelSegmentationModel(alpha={self.alpha}, base={self.base}, '
                '#customers={total_customers}, #tables={n_tables}, '
                '#dishes={n_dishes})').format(self=self, total_customers=total_customers,
                        n_tables=n_tables, n_dishes=n_dishes))

    def _log_likelihood(self, *tables):
        tables = [t for ts in tables for t in ts]
        ntables = len(tables)
        ncustomers = sum(c for _, c in tables)
        crp_ll = (math.lgamma(self.alpha) - math.lgamma(self.alpha + ncustomers)
              + sum(math.lgamma(c) for _, c in tables)
              + ntables * math.log(self.alpha))
        base_ll = self.base.log_likelihood()
        return crp_ll+base_ll, crp_ll, base_ll

    def _counts_of_counts(self, tables):
        return Counter(c for _, c in tables)

    def shutdown(self):
        """Shut down any resources used."""
        for p, iq, _ in self._slaves:
            iq.put(Message('shutdown'))
            p.join()

    def decode(self, word):
        analyses = self.seg_mappings[self.word_vocabulary[word]]
        probs = [self.base.marginal_prob(*a) for a in analyses]
        _, (p, s) = max(zip(probs, analyses))
        return self.prefix_vocabulary[p], self.suffix_vocabulary[s]

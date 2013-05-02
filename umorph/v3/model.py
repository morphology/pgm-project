import logging
import math
import time
import multiprocessing
import random
from collections import Counter
from umorph.segment import segmentation_mapping
from slave import CRPSlave
from distributions import MultinomialProduct

class ParallelSegmentationModel(object):
    def __init__(self, alpha, alpha_p, alpha_s, corpus, w_vocabulary, p_vocabulary, s_vocabulary, n_processors, n_mh):
        self.alpha = float(alpha)
        self.base = MultinomialProduct(len(p_vocabulary), alpha_p, len(s_vocabulary), alpha_s)
        self.base.resample()
        self.corpus = corpus
        self.word_vocabulary = w_vocabulary
        self.prefix_vocabulary = p_vocabulary
        self.suffix_vocabulary = s_vocabulary
        self.seg_mappings = segmentation_mapping(w_vocabulary, p_vocabulary, s_vocabulary)
        self.n_processors = n_processors
        self.n_mh = n_mh

        self._slaves = []
        self._processor_indicators = [random.randrange(self.n_processors) for _ in self.corpus]
        for gid in xrange(self.n_processors):
            iq, oq = multiprocessing.Queue(), multiprocessing.Queue()
            s = CRPSlave(self.alpha/self.n_processors, self.seg_mappings, gid, iq, oq)
            self._slaves.append((s, iq, oq))
            s.start()
            words = [w for i, w in enumerate(self.corpus) if self._processor_indicators[i] == gid]
            iq.put(words)

    def decode(self, word):
        analyses = self.seg_mappings[self.word_vocabulary[word]]
        probs = [self.base.marginal_prob(*a) for a in analyses]
        _, (p, s) = max(zip(probs, analyses))
        return self.prefix_vocabulary[p], self.suffix_vocabulary[s]

    def resample(self, processors=False):
        """Run the sampler for the parallelized model."""
        # Send the base to slaves
        for p, iq, _ in self._slaves:
            iq.put(self.base)

        logging.info('Local step')
        local_start = time.time()

        # Receive and aggregate counts
        total_p_counts = Counter()
        total_s_counts = Counter()
        for p, _, oq in self._slaves:
            p_counts, s_counts = oq.get()
            total_p_counts += p_counts
            total_s_counts += s_counts

        local_end = time.time()
        local_time = local_end - local_start
        logging.info('Local time: %f seconds', local_time)

        # Update the base
        self.base.update(total_p_counts, total_s_counts)

        logging.info('Global step')
        global_start = time.time()

        # Resample the base
        self.base.resample()

        if processors:
            mh_steps = 0.0
            mh_accepts = 0.0
            old_tables = []
            for _, iq, oq in self._slaves:
                iq.put('send_tables')
                old_tables.append(oq.get())

            for mh_step in xrange(self.n_mh):
                mh_steps += 1
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
                if accept:
                    mh_accepts += 1
                    old_tables = new_tables

            ## End Global

            acceptance_rate = mh_accepts/mh_steps

            total_customers = sum(sum(c for _, c in tables) for tables in old_tables)
            n_tables = sum(len(tables) for tables in old_tables)
            n_dishes = len(set(dish for tables in old_tables for dish, _ in tables))

            for i, (_, iq, _) in enumerate(self._slaves):
                iq.put(old_tables[i])

            global_end = time.time()
            global_time = global_end - global_start
            logging.info('Global time_proc: %f seconds', global_time)

            logging.info('MH Acceptance Rate: %f', acceptance_rate)
            logging.info('LL= %.0f\tCRPLL= %.0f\tBaseLL= %.0f', *self._log_likelihood(*new_tables))
            logging.info(('ParallelSegmentationModel(alpha={self.alpha}, base={self.base}, '
                    '#customers={total_customers}, #tables={n_tables}, '
                    '#dishes={n_dishes})').format(self=self, total_customers=total_customers,
                            n_tables=n_tables, n_dishes=n_dishes))
        
        if not processors:
            global_end = time.time()
            global_time = global_end - global_start
            logging.info('Global time: %f seconds', global_time)


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
            iq.put(None)
            p.join()

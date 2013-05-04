import multiprocessing
import random
from collections import Counter
from vpyp.pyp import CRP
from vpyp.prob import mult_sample


class CRPSlave(CRP, multiprocessing.Process):
    def __init__(self, alpha, seg_mappings, gid, iq, oq):
        CRP.__init__(self) # TODO fix hardcoded initialization
        multiprocessing.Process.__init__(self)

        self.alpha = alpha
        self.seg_mappings = seg_mappings
        self.gid = gid
        self.iq = iq
        self.oq = oq
        self.analyses = []
        self.p_counts = Counter()
        self.s_counts = Counter()

    def _random_table(self, k):
        """Pick a table with dish k randomly"""
        n = random.randrange(0, self.ncustomers[k])
        tables = self.tables[k]
        for i, c in enumerate(tables):
            if n < c: return i
            n -= c

    def seating_probs(self, w, initialize=False):
        """Joint probabilities of all possible (segmentation, table assignments) of word #w"""
        for p, s in self.seg_mappings[w]:
            yield (w, p, s, -1), (1 if initialize else self.alpha * self.base.prob(p, s))
            if not (w, p, s) in self.tables: continue
            for seat, count in enumerate(self.tables[(w, p, s)]):
                yield (w, p, s, seat), (1 if initialize else count)

    def increment(self, w, initialize=False):
        (w, p, s, seat) = mult_sample(self.seating_probs(w, initialize))
        if self._seat_to((w, p, s), seat):
            self.p_counts[p] += 1
            self.s_counts[s] += 1
        return (w, p, s)

    def decrement(self, w, p, s):
        seat = self._random_table((w, p, s))
        if self._unseat_from((w, p, s), seat):
            self.p_counts[p] -= 1
            self.s_counts[s] -= 1

    def init_tokens(self, words):
        # Initialize G
        for w in words:
            w, p, s = self.increment(w, initialize=True)
            self.analyses.append((w, p, s))

    def send_counts(self):
        # Send initialized counts to master
        self.oq.put((self.p_counts, self.s_counts))

    def update_base(self, base):
        # Update H
        self.base = base

    def send_tables(self):
        # Send tables to master
        my_tables = [(dish, c) for dish in self.tables for c in self.tables[dish]]
        self.oq.put(my_tables)

    def receive_tables(self, new_tables):
        self.analyses = []
        self.ntables = len(new_tables)
        self.total_customers = 0
        self.tables = {}
        self.ncustomers = {}
        for dish, c in new_tables:
            self.total_customers += c
            for i in xrange(c):
                self.analyses.append(dish)
            if dish not in self.tables:
                self.tables[dish] = []
            self.tables[dish].append(c)
            self.ncustomers[dish] = self.ncustomers.get(dish, 0) + c

    def resample(self):
        # 1. Resample table assignments
        for i, (w, p, s) in enumerate(self.analyses):
            self.decrement(w, p, s)
            w, p, s = self.increment(w)
            self.analyses[i] = (w, p, s)

        # 2. Resample table dishes
        new_analyses = []
        new_tables = {}
        new_ncustomers = {}
        for (w, old_p, old_s), tables in self.tables.iteritems():
            for c in tables:
                self.p_counts[old_p] -= 1
                self.s_counts[old_s] -= 1
                
                p, s = mult_sample(((p, s), self.base.prob(p, s))
                        for p, s in self.seg_mappings[w])
                self.p_counts[p] += 1
                self.s_counts[s] += 1

                if (w, p, s) not in new_tables:
                    new_tables[w, p, s] = []
                    new_ncustomers[w, p, s] = 0
                new_tables[w, p, s].append(c)
                new_ncustomers[w, p, s] += c
                new_analyses.extend([(w, p, s)] * c)

        self.analyses = new_analyses
        self.tables = new_tables
        self.ncustomers = new_ncustomers

    def run(self):
        while True:
            msg = self.iq.get()
            if msg.name == 'shutdown': break
            getattr(self, msg.name)(*msg.args)

    def __repr__(self):
        return 'CRPSlave(alpha={self.alpha}, gid={self.gid})'.format(self=self)

import multiprocessing
import random
from collections import Counter
from vpyp.pyp import CRP
from vpyp.prob import mult_sample


class CRPSlave(CRP, multiprocessing.Process):
    def __init__(self, alpha, base, seg_mappings, gid, iq, oq):
        CRP.__init__(self) # TODO fix hardcoded initialization
        multiprocessing.Process.__init__(self)

        self.alpha = alpha
        self.base = base
        self.seg_mappings = seg_mappings
        self.gid = gid
        self.iq = iq
        self.oq = oq
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

    def run(self):
        analyses = []
        words = self.iq.get()
        for w in words:
            w, p, s = self.increment(w, initialize=True)
            analyses.append((w, p, s))

        while True:
            parcel = self.iq.get()
            if parcel is None: # poison pill
                return
            elif parcel == 'send_tables':
                my_tables = [(dish, c) for dish in self.tables for c in self.tables[dish]]
                self.oq.put(my_tables)

                accept = self.iq.get()
                new_tables = self.iq.get()

                if accept:
                    analyses = []
                    self.ntables = len(new_tables)
                    self.total_customers = 0
                    self.tables = {}
                    self.ncustomers = {}
                    for dish, c in new_tables:
                        self.total_customers += c
                        for i in xrange(c):
                            analyses.append(dish)
                        if dish not in self.tables:
                            self.tables[dish] = []
                        self.tables[dish].append(c)
                        self.ncustomers[dish] = self.ncustomers.get(dish, 0) + c

            else:
                base = parcel
                self.base = base
                for i, (w, p, s) in enumerate(analyses):
                    self.decrement(w, p, s)
                    w, p, s = self.increment(w)
                    analyses[i] = (w, p, s)
                self.oq.put((self.p_counts, self.s_counts))

    def __repr__(self):
        return 'CRPSlave(alpha={self.alpha}, gid={self.gid})'.\
            format(self=self)


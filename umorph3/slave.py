import multiprocessing
import random
from collections import Counter
from vpyp.pyp import CRP
from vpyp.prob import mult_sample


class CRPSlave(CRP, multiprocessing.Process):
    def __init__(self, alpha, base, corpus, seg_mappings, gid, iq, oq):
        self.alpha = alpha
        self.base = base
        self.corpus = corpus
        self.seg_mappings = seg_mappings
        self.gid = gid
        self.iq = iq
        self.oq = oq
        self.p_counts = Counter()
        self.s_counts = Counter()
        CRP.__init__(self) # these two lines are bad, but okay for now
        multiprocessing.Process.__init__(self)

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
            yield (p, s, -1), (1 if initialize else self.alpha * self.base.prob(p, s))
            if not (p, s) in self.tables: continue
            for seat, count in enumerate(self.tables[(p, s)]):
                yield (p, s, seat), (1 if initialize else count)

    def increment(self, w, initialize=False):
        (p, s, seat) = mult_sample(self.seating_probs(w, initialize))
        if self._seat_to((p, s), seat):
            self.p_counts[p] += 1
            self.s_counts[s] += 1
        seat = seat if seat >= 0 else len(self.tables[(p, s)])-1 # get seat number if we created new table
        return (p, s), seat

    def decrement(self, p, s):
        seat = self._random_table((p, s))
        if self._unseat_from((p, s), seat):
            self.p_counts[p] -= 1
            self.s_counts[s] -= 1

    def run(self):
        processor_indicators = self.iq.get()
        analyses = {}
        seats = {}
        for i in (i for i, pi in enumerate(processor_indicators) if pi == self.gid):
            analysis, seat = self.increment(self.corpus[i], initialize=True)
            analyses[i] = analysis
            seats[i] = seat

        while True:
            parcel = self.iq.get()
            if parcel is None: # poison pill
                return
            if parcel == 'send_assignments': # prepare for processor resample
                assignments = {}
                for i in (i for i, pi in enumerate(processor_indicators) if pi == self.gid):
                    assignments[i] = (self.gid, analyses[i], seats[i])
                self.oq.put(assignments)
                new_tables = self.iq.get()
            else:
                base, processor_indicators = parcel
                self.base = base
                for i in (i for i, pi in enumerate(processor_indicators) if pi == self.gid):
                    self.decrement(*analyses[i])
                    analysis, seat = self.increment(self.corpus[i])
                    analyses[i] = analysis
                    seats[i] = seat
                self.oq.put((self.p_counts, self.s_counts))

    def __repr__(self):
        return 'CRPSlave(alpha={self.alpha}, gid={self.gid})'.\
            format(self=self)


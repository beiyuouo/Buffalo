import re

class Acc(object):
    def __init__(self):
        self.acc = 0
        self.n = 0

    def clear(self, ans):
        cleaner = lambda t: re.sub(
            "[.,?;*!%^&_+():-\[\]{}]", "", t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower()
        )

        return cleaner(ans)

    def compute_score(self, refs, hypos):
        # compute acc
        for k, ref in refs.items():
            hypo = hypos[k]
            # print(ref, hypo)
            for rr, hh in zip(ref, hypo):
                if self.clear(rr) in self.clear(hh):
                    self.acc += 1
                self.n += 1
        return self.acc / self.n, self.acc

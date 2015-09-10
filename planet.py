import numpy as np


class Planet:
    def __init__(self, r=None, v=None):
        self.r = r or [0, 0]
        self.v = v or [0, 0]

    def _numpy_row(self):
        r = []
        r.extend(self.r)
        r.extend(self.v)
        return r

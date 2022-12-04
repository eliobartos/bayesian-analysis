import pandas as pd
import numpy as np
from collections.abc import Iterable
import copy

class DiscreteRV:
    """
    Class representing discrete random variable.
    """
    def __init__(self, values=None, probs=None):
        """
        Args:
            values: values that RV can take
            probs: probs or weights of taking specific value
        """
        if values is None or probs is None:
            self.values = []
            self.probs = np.array([])
            return
        
        self.values = values

        if isinstance(probs, float) | isinstance(probs, int):
            self.probs = np.repeat(probs, len(self.values))
        else:
            self.probs = np.array(probs)

    def _repr_html_(self):
        """
        Returns an HTML representation of the series.
        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(data = self.probs, index = self.values)
        df.columns = ['probs']
        return df._repr_html_()

    def __setitem__(self, key, new_prob):
        if key in self.values:
            idx = self.values.index(key)
            self.probs[idx] = new_prob
        else:
            self.values.append(key)
            self.probs = np.append(self.probs, new_prob)
    
    def __getitem__(self, key):
        if key not in self.values:
            return 0
        idx = self.values.index(key)
        return self.probs[idx]
        
    def __mul__(self, other):
        new = copy.deepcopy(self)
        new.probs *= other
        new.normalize()
        return new
    
    def __len__(self):
        return len(self.values)

    def get_df(self):
        df = pd.DataFrame(data = self.probs, index = self.values)
        df.columns = ['probs']
        return df

    def mean(self):
        return np.sum(self.values * self.probs)

    def normalize(self):
        self.probs = self.probs / np.sum(self.probs)

    def get_marginal_distributions(self):
        """
        If RV is multidimensional, get all marginal distributions.
        """
        n_dim = len(self.values[0])
        dists = [DiscreteRV() for i in range(n_dim)]

        for j, v in enumerate(self.values):
            for i, xi in enumerate(v):
                dists[i][xi] += self.probs[j]
        
        return dists

    @staticmethod
    def from_independent_distributions(dists):
        """
        Create new RV from list of independent distributions. 
        """
        out = dists[0]
        for i in range(1, len(dists)):
            out = out._join_with_dist(dists[i])
        return out

    def _join_with_dist(self, other):
        """
        Create multidimensinal distribution from current and other as independent
        of each other.
        """
        new = DiscreteRV()

        for i, v1 in enumerate(self.values):
            for j, v2 in enumerate(other.values):
                if isinstance(v1, Iterable) and isinstance(v2, Iterable):
                    new[(*v1, *v2)] = self.probs[i] * other.probs[j]
                elif isinstance(v1, Iterable):
                    new[(*v1, v2)] = self.probs[i] * other.probs[j]
                elif isinstance(v2, Iterable):
                    new[(v1, *v2)] = self.probs[i] * other.probs[j]
                else:
                    new[(v1, v2)] = self.probs[i] * other.probs[j]
        return new

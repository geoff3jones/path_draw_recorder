import copy
import time
import numpy as np
import pandas as pd

class PathsList():
    def __init__(self):
        self._pathlist = []

    def get_list(self):
        return self._pathlist

    def get_dataframe(self):
        return pd.DataFrame.from_dict([path.as_dict() for path in self._pathlist
                                       ])

    def __str__(self):
        return f"({self._pathlist})"

    def __repr__(self):
        return str(self)


class DrawnPath():
    """
    The collected user drawn path and target path settings. Use the factory
    method to generate DrawnPaths correctly
    """

    def __init__(self, path_list, target_settings):
        self.drawn_data = np.array(path_list)
        self.target_settings = copy.deepcopy(target_settings)

    def __str__(self):
        return f"({self.target_settings}, {self.drawn_data})"

    def __repr__(self):
        return str(self)

    def as_dict(self):
        d = copy.deepcopy(self.target_settings)
        d['x pos'] = self.drawn_data[:,0].copy()
        d['y pos'] = self.drawn_data[:,0].copy()
        d['t'] = self.drawn_data[:,0].copy()
        return d

    @property
    def size(self):
        return self.drawn_data.shape[0]


def get_drawnpath_factory(x, y, settings, pathlist=None):
    """
    Returns a factory for generating DrawnPath objects:
    Usage:
        factory = get_drawnpath_factory(x, y, settings,...)
        factory(x, y)
        factory.finalise()
    """

    def _time_sec(): return time.time_ns() / (10**9)
    _start = _time_sec()
    _temp_p = [(x, y, 0)]
    _settings = copy.deepcopy(settings)

    def _factory(x, y):
        _temp_p.append((x, y, _time_sec()-_start))

    def _finalise():
        path = DrawnPath(_temp_p, _settings)
        if pathlist is not None:
            pathlist.append(path)
        else:
            return path
    _factory.finalise = _finalise
    return _factory

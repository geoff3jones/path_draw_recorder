import copy
import time
import numpy as np

class PathsList():
    def __init__(self, seed=None):
        self._seed = seed
        self._pathlist = []
        if seed is not None:
            np.random.seed(seed)
        self._MT_STATE = np.random.get_state()

    def get_list(self):
        return self._pathlist

    def serialise(self):
        pass

    def __str__(self):
        return f"({self._MT_STATE}, {self._pathlist})"

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

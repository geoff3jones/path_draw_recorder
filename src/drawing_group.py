import time
import functools
import numpy as np
from hashlib import md5

def newDrawingGroup(GroupType,cached=False):
    """
    Get a drawing group to manage the drawing of paths, extends the
    given GroupType adds storing of the drawn path data
    """
    if not cached:
        return _newDrawingGroup_uncached(GroupType)
    else:
        return _newDrawingGroup_cached(GroupType)

@functools.lru_cache(maxsize=None)
def _newDrawingGroup_cached(GroupType):
    return _newDrawingGroup_uncached(GroupType)

def _newDrawingGroup_uncached(GroupType):
    class DrawingGroup(GroupType):
        
        _pathslist = []

        def exportlist():
            return [dg.to_dict() for dg in DrawingGroup._pathslist]

        def resetlist():
            DrawingGroup._pathslist = []

        def __init__(self, *args, **kwargs):
            self._rnd_state = np.random.get_state()
            super(DrawingGroup, self).__init__(*args, **kwargs)
            self._drawn_paths = tuple(list() for _ in range(super(DrawingGroup,self).n_subpaths()))
            self._active_subpath = 0
            self._start = 0
            DrawingGroup._pathslist.append(self)

        def to_dict(self):
            d = {k: v for k, v in zip(['rnd_engine',
                                       'rnd_keys',
                                       'rnd_pos',
                                       'has_gauss',
                                       'cached_gauss'], self._rnd_state)}
            for i, p in enumerate(self._drawn_paths):
                path = np.array(self._drawn_paths[i])
                d[f"path[{i}].drawn_x"] = path[:,0]
                d[f"path[{i}].drawn_y"] = path[:,1]
                d[f"path[{i}].drawn_t"] = path[:,2]
            return {'state_hash': md5(self._rnd_state).hexdigest(),
                    **super(DrawingGroup, self).to_dict(),
                    **d}

        def _time_sec(self):
            return time.time_ns() / (10**9)

        def reset_active_path(self):
            self._drawn_paths[self._active_subpath][:] = []
            self._start = self._time_sec()

        def append_coord(self, x: int, y: int):
            self._drawn_paths[self._active_subpath].append((x,y,self._time_sec()-self._start))

        def set_active_subpath(self, active_subpath: int):
            assert active_subpath < len(self._drawn_paths), \
                f"active_subpath ({active_subpath}) must be a valid index: in the range [0,{super(DrawingGroup,self).n_subpaths() - 1 })"
            self._active_subpath = active_subpath


    return DrawingGroup

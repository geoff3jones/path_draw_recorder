import argparse
import uuid 
import sys
import os
import hashlib
import re

import cv2
import numpy as np
import pandas as pd

from ellipse import EllipseGroup
from bspline import BSplineGroup
from drawing_group import newDrawingGroup

#np.set_printoptions(threshold=50)

# Create a black image, a window and bind the function to window
# mouse callback function

DBG_LVL = 0

class PathCaptureSettings():
    def __init__(self, settings):
        """
        this should sponge up all that spare state out of global namespace
        """
        self._itmax = settings.itmax
        self._npaths = settings.npaths
        self._current_iteration = 0 if settings.current_iter is None \
                                    else settings.current_iter
        self._current_path = 0 if settings.current_path is None \
                               else settings.current_path
        self._paths_remaining = True
        self._mode = "bspline"
        self._img = np.zeros((512, 512, 3), np.uint8)
        self._img_drawlayer = np.zeros((512, 512, 3), np.uint8)

        if settings.seed_state is not None:
            np.random.set_state(settings.seed_state)
        elif settings.seed is not None:
            np.random.seed(settings.seed)
        # store the MT19937 initial sate
        (self._rnd_mt_str,
         self._rnd_mt_keys,
         self._rnd_mt_pos,
         self._rnd_mt_has_gauss,
         self._rnd_mt_gauss_cached) = np.random.get_state()

    def get_img(self, layer='base'):
        if layer=='base':
            return self._img
        if layer=='draw':
            return self._img_drawlayer
        if layer=='combined':
            return np.maximum(self._img,self._img_drawlayer)


    def to_dict(self):
        return {
                "iterations":   self._itmax,
                "n_paths":      self._npaths,
                "current_iter": self._current_iteration,
                "current_path": self._current_path,
                "rnd_state":    self.get_seed_state_tuple(),
                }


    def get_seed_state_tuple(self):
        return (self._rnd_mt_str, self._rnd_mt_keys, self._rnd_mt_pos,
                self._rnd_mt_has_gauss, self._rnd_mt_gauss_cached)

    @property
    def paths_remaining(self):
        return self._paths_remaining

    def terminate(self):
        self._paths_remaining = False

    def _reset_random(self, n=0):
        np.random.set_state((self._rnd_mt_str,
                             self._rnd_mt_keys,
                             self._rnd_mt_pos + n,
                             self._rnd_mt_has_gauss,
                             self._rnd_mt_gauss_cached))

    def get_next_path_group(self):
        """
        get the next path group.
        this will fast forward to the current path and iteration, generating
        all the same random states along the way
        """
        for i in range(self._current_iteration, self._itmax):
            if self._current_iteration < i:
                self._current_iteration = i
            self._reset_random()
            for p in range(self._current_path, self._npaths):
                if self._current_iteration == i and self._current_path < p:
                    self._current_path = p
                if DBG_LVL > 0:
                    print(f"np.MT: key# {hashlib.md5(np.random.get_state()[1]).hexdigest()} | pos {np.random.get_state()[2]}")
                pathgroup = self._get_next_path_group()
                if self._current_iteration == i and self._current_path == p:
                    yield (pathgroup, i, p)
            if self._current_iteration == i:
                self._current_path = 0

        self._current_iteration = None
        self._current_path = None

    def _get_next_path_group(self):
        if self._mode == "bspline":
            return newDrawingGroup(BSplineGroup, cached=True)(samples_per_n=50)
        elif self._mode == "ellipse":
            return newDrawingGroup(EllipseGroup, cached=True)()


def get_callbacks(path_capture_settings ):
    
    group_gen = path_capture_settings.get_next_path_group()
    path_group = None
    def new_path_group():
        """
        Gets a new group object and returns the co-routine that draws
        each path highlighted iteratively
        """
        nonlocal path_group
        try:
            path_group = next(group_gen)
        except StopIteration as e:
            path_capture_settings.terminate()
        return path_group[0].draw_iterative_highlight_ends(path_capture_settings.get_img())

    path_group_draw  = iter([])
    def cycle_path_group():
        """
        cycles through the paths iteratively highlighting individual ones
        """
        nonlocal path_group_draw
        path_capture_settings.get_img('base')[:] = 0
        path_capture_settings.get_img('draw')[:] = 0
        try:
            iteration_info = next(path_group_draw)
            cv2.putText(path_capture_settings.get_img(),
                       f"{path_group[1]+1}/{path_capture_settings._itmax} "
                       f"{path_group[2]+1}/{path_capture_settings._npaths} "
                       f"{iteration_info+1}/3",
                        (50, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        except StopIteration as e:
            path_group_draw = new_path_group()
            iteration_info = cycle_path_group()
        return iteration_info

    def keyboard_callback(key):
        if key == 32: # spacebar
            i = cycle_path_group()
            path_group[0].set_active_subpath(i)

    def draw_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and path_group is not None:
            path_capture_settings.get_img('draw')[:] = 0
            path_group[0].reset_active_path()
            path_group[0].append_coord(x, y)
            cv2.circle(path_capture_settings.get_img('draw'),
                       (x, y), 4, (255, 0, 0))
        if event == cv2.EVENT_LBUTTONUP and path_group is not None:
            pass
        if event == cv2.EVENT_MOUSEMOVE and path_group is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            # draw circle and add point to path
            path_group[0].append_coord(x, y)
            cv2.circle(path_capture_settings.get_img('draw'),(x,y),4,(255,0,0))
    return {'draw':draw_callback,
            'keyboard':keyboard_callback}

class FileIOHelper():
    def __init__(self, arg_settings):
        self._data_file_root = os.path.abspath(arg_settings.outfile)
        self._temp_file_root = os.path.abspath(f".temp.{uuid.uuid4().hex}")
        self._data_file_ok   = True
        self._out_type       = arg_settings.outtype

        if arg_settings.outtype == "pickle":
            self._writer = pd.DataFrame.to_pickle
            self._writer_args = dict(compression="bz2")
            self._reader = pd.read_pickle
            self._reader_args = dict(compression="bz2")
            self._file_ext = ".bz2.pkl"
        elif arg_settings.outtype == "parquet":
            self._writer = pd.DataFrame.to_parquet
            self._writer_args = dict()
            self._reader = pd.read_parquet
            self._reader_args = dict()
            self._file_ext = ".parquet"
        elif arg_settings.outtype == "csv":
            self._writer = pd.DataFrame.to_csv
            self._writer_args = dict()
            self._reader = pd.read_csv
            self._reader_args = dict()
            self._file_ext = ".csv"
        
    def _get_file_root(self):
        return self._data_file_root if self._data_file_ok else self._temp_file_root

    @property
    def settings_filename(self):
        return f"{self._get_file_root()}.settings.pkl"

    @property
    def data_filename(self):
        return f"{self._get_file_root()}{self._file_ext}"

    def read(self, target="settings"):
        if target == "settings":
            return pd.read_pickle(self.settings_filename)
        elif target == "data":
            return self._reader(self.data_filename, **self._reader_args)
        raise ValueError("unknown target type")

    def write(self, data_frame):
        try:
            self._writer(data_frame, self.data_filename, **self._writer_args)
        except FileNotFoundError:
            self._data_file_ok = False
            print( "The target output file could not be written to; output will try to be written to:\n"
                  f"{os.path.abspath(self.data_filename)}")
            self._writer(data_frame, self.data_filename, **self._writer_args)


def parse_args():
    global DBG_LVL

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,  # type=String,
                        help="output file name [less extension] to save the"
                        " results data frame to")
    parser.add_argument("-t", "--outtype", dest="outtype", default="pickle",
                        choices=["pickle", "parquet", "csv"], 
                        help="output file name to save the results data frame to")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None,
                        help="set a seed value for the random shape generator for repeatability",
                        )
    parser.add_argument("-S", "--settingsfile", dest="settingsfile", default=None,
                        help="load settings from file this would have been "
                        "created in a previous run")
    parser.add_argument("-n", "--npaths", dest="npaths", type=int, default=25,
                        help="the number of paths to record")
    parser.add_argument("-i", "--itmax", dest="itmax", type=int, default=5,
                        help="the number times to record each path")
    parser.add_argument( "--_dbg_level", dest="DBG_LVL", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument( "--start_iter", dest="current_iter", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument( "--start_path", dest="current_path", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    io_helper = FileIOHelper(args)

    # run is a continutation from previous and should be appended to the same file 
    args.continuation = False
    # check for existing settings file
    if args.settingsfile is None and os.path.exists(io_helper.settings_filename):
        msg = "Matching settings file found\n" \
              "Would you like to load these settings? [Y/n]"
        loadsettings = input(msg)
        if re.match(r'[nN][oO]|[nN]', loadsettings) is None:
            args.settingsfile = io_helper.settings_filename

    if args.settingsfile is not None:
        print("loading pickled settings file")
        state = pd.read_pickle(args.settingsfile)
        args.seed_state = state.rnd_state[0]
        if state.current_iter[0] is not None or state.current_path[0] is not None \
            and os.path.exists(io_helper.data_filename):
            msg = "It looks like you were half way through this, shall we pick up were you left of? [Y/n]"
            load_last_position = input(msg)
        if re.match(r'[nN][oO]|[nN]', load_last_position ) is None:
            args.continuation = True
            args.current_path = state.current_path[0]
            args.current_iter = state.current_iter[0]
            args.itmax  = state.iterations[0]
            args.npaths = state.n_paths[0]

    else:
        args.seed_state = None

    DBG_LVL = args.DBG_LVL

    return args, io_helper

def main():
    args, io_helper = parse_args()
    path_capture_settings = PathCaptureSettings(args)
    cv2.putText(path_capture_settings.get_img(),
                "press [space] to begin", (160, 250),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    call_backs = get_callbacks(path_capture_settings)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', call_backs['draw'])

    while(path_capture_settings.paths_remaining):
        cv2.imshow('image', path_capture_settings.get_img('combined'))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        call_backs['keyboard'](key)

    state_d = path_capture_settings.to_dict()
    state_df = pd.DataFrame.from_dict([state_d])
    out_df = pd.DataFrame.from_dict(newDrawingGroup(BSplineGroup,
                                                cached=True).exportlist())

    if args.continuation:
        old_df = io_helper.read("data")
        out_df = pd.concat([old_df, out_df]).dropna()

    io_helper.write(out_df)
    state_df.to_pickle(io_helper.settings_filename)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

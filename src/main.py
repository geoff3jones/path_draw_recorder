import argparse
import uuid 
import sys

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
    def __init__(self, parsed_args):
        """
        this should sponge up all that spare state out of global namespace
        """
        self._itmax  = parsed_args.itmax
        self._npaths = parsed_args.npaths
        self._paths_remaining = True
        self._mode   = "bspline"
        self._img    = np.zeros((512, 512, 3), np.uint8)
        self._img_drawlayer = np.zeros((512, 512, 3), np.uint8)

        if parsed_args.seed is not None:
            np.random.seed(parsed_args.seed)
        if parsed_args.seed_state is not None:
            np.random.set_state(parsed_args.seed_state)
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

    def get_seed_state_tuple(self):
        return (self._rnd_mt_str, self._rnd_mt_keys, self._rnd_mt_pos,
                self._rnd_mt_has_gauss, self._rnd_mt_gauss_cached)

    def get_seed_state_dict(self):
        return {"ENGINE":       self._rnd_mt_str,
                "KEYS":         self._rnd_mt_keys,
                "POS":          self._rnd_mt_pos,
                "HAS_GAUSS":    self._rnd_mt_has_gauss,
                "CACHED_GAUSS": self._rnd_mt_gauss_cached}

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
        for i in range(self._itmax):
            for p in range(self._npaths):
                self._reset_random(p*64)
                if DBG_LVL > 0:
                    print(f"dgb: current mt_pos {self._rnd_mt_pos + (p*257)}")
                yield (self._get_next_path_group(), i, p)

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
                        (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
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
    parser.add_argument("-S", "--seedfile", dest="seedfile", default=None,
                        help="set a seed state from pickle file this would have"
                        " been created in a previous run",
                        )
    parser.add_argument("-n", "--npaths", dest="npaths", type=int, default=25,
                        help="the number of paths to record")
    parser.add_argument("-i", "--itmax", dest="itmax", type=int, default=5,
                        help="the number times to record each path")
    parser.add_argument( "--_dbg_level", dest="DBG_LVL", type=int, default=0,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.seedfile is not None:
        print("loading pickled seed state")
        state = pd.read_pickle(args.seedfile)
        args.seed_state = tuple(v for v in state.to_dict()[0].values())
    else:
        args.seed_state = None

    DBG_LVL = args.DBG_LVL

    return args

if __name__ == "__main__":

    args = parse_args()
    path_capture_settings = PathCaptureSettings(args)

    callbacks = get_callbacks(path_capture_settings)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', callbacks['draw'])

    while(path_capture_settings.paths_remaining):
        cv2.imshow('image', path_capture_settings.get_img('combined'))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        callbacks['keyboard'](key)


 #   cv2.putText(img, "Done! Thankyou, now saving...",
 #              (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
#    cv2.imshow('image', img)


    state_d = path_capture_settings.get_seed_state_tuple()
    state_df = pd.DataFrame.from_dict(state_d)
    df = pd.DataFrame.from_dict(newDrawingGroup(BSplineGroup,
                                                cached=True).exportlist())
    try:
        if args.outtype == "pickle":
            df.to_pickle(f"{args.outfile}.bz2.pkl", compression="bz2")
        elif args.outtype == "parquet":
            df.to_parquet(f"{args.outfile}.parquet")
        elif args.outtype == "csv":
            df.to_csv(f"{args.outfile}.csv")
        else:
            print("unknown export file type")
        # save rndm state
        state_df.to_pickle(f"{args.outfile}.seedstate.pkl")

    except FileNotFoundError:
        outfile = f".temp.{uuid.uuid4().hex}"
        if args.outtype == "pickle":
            outfile = f"{outfile}.bz2.pkl"
            df.to_pickle(outfile, compression="bz2")
        elif args.outtype == "parquet":
            outfile = f"{outfile}.parquet"
            df.to_parquet(outfile)
        elif args.outtype == "csv":
            outfile = f"{outfile}.csv"
            df.to_csv(outfile)
        # save rndm state
        state_df.to_pickle(f"{outfile}.seedstate.pkl")

        print(f"""couldn't write to the path:
        \"{args.outfile}\"
        instead writing to th temporary file
        {outfile}""")
    
    


    cv2.destroyAllWindows()

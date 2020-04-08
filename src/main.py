from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd

from drawnpath import get_drawnpath_factory
from drawnpath import PathsList

np.set_printoptions(threshold=50)

# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
# mouse callback function

pathslist = PathsList()

class PathCaptureSettings():
    def __init__(self, parsed_args):
        self._itmax  = parsed_args.itmax
        self._npaths = parsed_args.npaths
        self._paths_remaining = True

        if parsed_args.seed is not None:
            np.random.seed(parsed_args.seed)
        # store the MT19937 initial sate
        (self._rnd_mt_str,
         self._rnd_mt_keys,
         self._rnd_mt_pos,
         self._rnd_mt_has_gauss,
         self._rnd_mt_gauss_cached) = np.random.get_state()

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

    def get_next_settings(self):
        for i in range(self._itmax):
            for p in range(self._npaths):
                self._reset_random(p*256)
                yield (self._get_next_settings(), i, p)

    def _get_next_settings(self):
        # retrieve the state position before
        state_pos  = np.random.get_state()[2]

        startAngle = np.random.rand()*360.0
        arclength  = 40 + np.sum(np.random.rand(2))*150.0
        direction  = np.random.choice([-1, 1])
        settings = {'MT_STATE_POS': state_pos,
                    'shape':        'ellipse',
                    'centre':       np.random.randint(200, 300, (2,)).tolist(),
                    'radii':        np.random.randint(10, 100, (2,)).tolist(),
                    'angle':        np.random.rand()*360.0,
                    'startAngle':   startAngle,
                    'endAngle':     startAngle + (direction*arclength),
                    }
        return settings

    def draw_next_shape(self, settings, i=None, p=None):
        # zero the image
        img[:] = 0
        if i is not None and p is not None:
            cv2.putText(img, f"{i}/{self._itmax} {p}/{self._npaths}",
                        (100, 100), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

        cv2.ellipse(img,
                    center=tuple(settings['centre']),
                    axes=tuple(settings['radii']),
                    angle=settings['angle'],
                    startAngle=settings['startAngle'],
                    endAngle=settings['endAngle'],
                    color=(255,255,255),
                    thickness=2)
        cv2.ellipse(img,
                    center=tuple(settings['centre']),
                    axes=tuple(settings['radii']),
                    angle=settings['angle'],
                    startAngle=settings['startAngle'],
                    endAngle=settings['startAngle']+5,
                    color=(0,255,0), # b g r
                    thickness=2)
        cv2.ellipse(img,
                    center=tuple(settings['centre']),
                    axes=tuple(settings['radii']),
                    angle=settings['angle'],
                    startAngle=settings['endAngle']-5,
                    endAngle=settings['endAngle'],
                    color=(0,0,255), # b g r
                    thickness=2)


def get_draw_callback(path_capture_settings):
    factory = None
    settings_gen = path_capture_settings.get_next_settings()
    settings = None
    def draw_next():
        nonlocal settings
        try:
            settings = next(settings_gen)
            path_capture_settings.draw_next_shape(*settings)
        except StopIteration:
            path_capture_settings.terminate()
        return settings
    draw_next()
    def draw_callback(event, x, y, flags, param):
        nonlocal factory, settings
        if event == cv2.EVENT_LBUTTONDOWN:
            factory = get_drawnpath_factory(x, y, settings[0], pathslist.get_list())
            cv2.circle(img,(x,y),4,(255,0,0))
        if event == cv2.EVENT_LBUTTONUP:
            # finalise the path
            if factory is not None:
                factory.finalise()
            factory = None
            # draw new target path
            draw_next()
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img,(x,y),4,(255,0,0))
            if factory is not None:
                factory(x, y)
    return draw_callback

def parse_args():
    global pathslist
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,  # type=String,
                        help="output file name to save the results data frame to")
    parser.add_argument("-t", "--outtype", dest="outtype", default="parquet",
                        choices=["parquet", "csv"], 
                        help="output file name to save the results data frame to")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None,
                        help="set a seed value for the random shape generator for repeatability",
                        )
    parser.add_argument("-n", "--npaths", dest="npaths", type=int, default=25,
                        help="the number of paths to record")
    parser.add_argument("-i", "--itmax", dest="itmax", type=int, default=5,
                        help="the number times to record each path")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    path_capture_settings = PathCaptureSettings(args)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_draw_callback(path_capture_settings))

    while(path_capture_settings.paths_remaining):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.putText(img, "Done! Thankyou, now saving...",
                (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    cv2.imshow('image', img)


    df = pathslist.get_dataframe()
    if args.outtype == "parquet":
        df.to_parquet(args.outfile)
    elif args.filetype == "csv":
        df.to_csv(args.outfile)
    else:
        print("unknown export file type")

    cv2.destroyAllWindows()

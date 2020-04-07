from optparse import OptionParser

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

def get_next_settings():
    # retrieve the state position before
    state_pos  = np.random.get_state()[2]

    startAngle = np.random.rand()*360.0
    arclength  = 40 + np.sum(np.random.rand(2))*150.0
    direction  = np.random.choice([-1, 1])
    settings = {'MT_STATE_POS': state_pos,
                'shape':        'ellipse',
                'centre':       tuple(np.random.randint(200, 300, (2,)).tolist()),
                'radii':        tuple(np.random.randint(10, 100, (2,)).tolist()),
                'angle':        np.random.rand()*360.0,
                'startAngle':   startAngle,
                'endAngle':     startAngle + (direction*arclength),
                }
    return settings

def draw_next_shape(settings):
    # zero the image
    img[:] = 0
    cv2.ellipse(img,
                center=settings['centre'],
                axes=settings['radii'],
                angle=settings['angle'],
                startAngle=settings['startAngle'],
                endAngle=settings['endAngle'],
                color=(255,255,255),
                thickness=2)
    cv2.ellipse(img,
                center=settings['centre'],
                axes=settings['radii'],
                angle=settings['angle'],
                startAngle=settings['startAngle'],
                endAngle=settings['startAngle']+5,
                color=(0,255,0), # b g r
                thickness=2)
    cv2.ellipse(img,
                center=settings['centre'],
                axes=settings['radii'],
                angle=settings['angle'],
                startAngle=settings['endAngle']-5,
                endAngle=settings['endAngle'],
                color=(0,0,255), # b g r
                thickness=2)

def get_draw_callback():
    factory = None
    settings = get_next_settings()
    draw_next_shape(settings)
    def draw_callback(event, x, y, flags, param):
        nonlocal factory, settings
        if event == cv2.EVENT_LBUTTONDOWN:
            factory = get_drawnpath_factory(x, y, settings, pathslist.get_list())
            cv2.circle(img,(x,y),4,(255,0,0))
        if event == cv2.EVENT_LBUTTONUP:
            # finalise the path
            if factory is not None:
                factory.finalise()
            factory = None
            # draw new target path
            settings = get_next_settings()
            draw_next_shape(settings)
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img,(x,y),4,(255,0,0))
            if factory is not None:
                factory(x, y)
    return draw_callback


def parse_args():
    global pathslist
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", type='int', default=None,
                      help="set a seed value for the random shape generator for repeatability",
                      )
    (options, _) = parser.parse_args()
    if options.seed is not None:
        pathslist = PathsList(options.seed)


if __name__ == "__main__":

    parse_args()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_draw_callback())

    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print(pathslist)

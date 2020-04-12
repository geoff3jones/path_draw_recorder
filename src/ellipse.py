import numpy as np
import cv2


class Ellipse():

    def __init__(self):
        """
        A partial ellipse
        """
        # this must happen before anyother calls to random are made
        self._mt_pos = np.random.get_state()[3]

        startAngle = np.random.rand()*360.0
        arclength = 60 + np.sum(np.random.rand(2))*140.0
        direction = np.random.choice([-1, 1])
        self._shape      = 'ellipse'
        self._centre     = np.random.randint(200, 300, (2,)).tolist()
        self._radii      = np.random.randint(10, 100, (2,)).tolist()
        self._angle      = np.random.rand()*360.0
        self._startAngle = startAngle
        self._endAngle   = startAngle + (direction*arclength)

    def draw(self, img, highlight_ends=False):
        cv2.ellipse(img,
                    center=tuple(self._centre),
                    axes=tuple(self._radii),
                    angle=self._angle,
                    startAngle=self._startAngle,
                    endAngle=self._endAngle,
                    color=(255,255,255),
                    thickness=2)
        if highlight_ends:
            cv2.ellipse(img,
                        center=tuple(self._centre),
                        axes=tuple(self._radii),
                        angle=self._angle,
                        startAngle=self._startAngle,
                        endAngle=self._startAngle+5,
                        color=(0,255,0), # b g r
                        thickness=2)
            cv2.ellipse(img,
                        center=tuple(self._centre),
                        axes=tuple(self._radii),
                        angle=self._angle,
                        startAngle=self._endAngle-5,
                        endAngle=self._endAngle,
                        color=(0,0,255), # b g r
                        thickness=2)
        
    def to_dict(self):
        return {'mt_pos':          self._mt_pos,
                'shape':           'ellipse',
                'centre_x':        self._centre[0],
                'centre_y':        self._centre[1],
                'radii_primary':   self._radii[0],
                'radii_secondary': self._radii[1],
                'angle':           self._angle,
                'startAngle':      self._startAngle,
                'endAngle':        self._endAngle,
                }

class EllipseGroup():

    def __init__(self, n_subpaths=1):
        """
        initialise and store random sate prior
        """
        self._n_subpaths = n_subpaths
        # this must happen before anyother calls to random are made
        self._mt_pos = np.random.get_state()[3]
        
        self._paths = [Ellipse() for _ in range(self._n_subpaths)]

    def to_dict(self):
        """
        export to a flat dictionary
        """
        settings = {'mt_pos': self._mt_pos,
                    'n_subpaths': len(self._paths)}
        for i,p in enumerate(self._paths):
            for k, v in p.to_dict().items():
                settings[f"path[{i}].{k}"] = v
        
        return settings

    def draw(self, img, highlight_ends=False):
        """
        draw everything and highlight start and end of path
        """
        for p in self._paths:
            p.draw(img)

    def draw_iterative_highlight_ends(self):
        """
        this should return a generator that will cycle
        """
        for i in range(self._n_subpaths):
            for I,p in enumerate(self._paths):
                p.draw(img,i==I)
            yield i
    

if __name__ == "__main__":
    import cv2

    img=np.zeros((500, 500, 3), np.uint8)

    def new_char():
        e_char = EllipseGroup(2)
        #bs_char.draw_bspline(img,'centre')
        print(e_char.to_dict())
        return e_char.draw_iterative_highlight_ends()

    char_cycle = iter([])
    def cycle_char():
        global char_cycle
        img[:] = 0
        try:
            next(char_cycle)
        except StopIteration as e:
            char_cycle = new_char()
            cycle_char()
        


    while(True):
        cv2.imshow('image', img)
        key=cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key == 32:
            cycle_char()

    cv2.destroyAllWindows()

import numpy as np
import cv2


class Bspline:
    """
    Create a bspline curve for given control points and knotvector
    """

    def __init__(self, control_points=None, order="quadratic", knotvector=None,
                 boundingbox=None, max_complexity=None):
        self._mt_pos = np.random.get_state()[2]

        self._order = order
        if self._order == "linear":
            self._k = 2
        if self._order == "quadratic":
            self._k = 3
        if self._order == "cubic":
            self._k = 4

        if control_points is None:
            if boundingbox is None:
                boundingbox = [[10, 490], [10, 250]]
            if max_complexity is None:
                max_complexity = 5
            # gen = np.random.default_rng()
            shape = (np.random.randint(self._k,
                                               self._k+max_complexity),)
            self._control_pts = np.stack(
                [np.random.choice(boundingbox[0][1] - boundingbox[0][0], shape) + boundingbox[0][0],
                 np.random.choice(boundingbox[1][1] - boundingbox[1][0], shape) + boundingbox[1][0]])

        self._n = self._control_pts.shape[1] - 1

        if self._order == "linear":
            self._knotvector = [1]*self._k + \
                list(range(2, self._n+1)) + [self._n+1]*self._k
        elif self._order == "quadratic":
            self._knotvector = [1]*self._k + \
                list(range(2, self._n)) + [self._n]*self._k
        elif self._order == "cubic":
            self._knotvector = [1]*self._k + \
                list(range(2, self._n-1)) + [self._n-1]*self._k

        # self._control_pts = np.array([[0,0,1,0,0]]).T

        if self._k + self._control_pts.shape[1] != len(self._knotvector):
            raise ValueError(
                "knot vector length should equal number of control points + order")

        self._boundingbox_xy = None
        self._boundingbox_aex = None
        self._sample_points = None

    def gen_sample_points(self,samples_per_n):

        # this doesn't have to be a uniform range
        t = np.linspace(start=self._knotvector[self._k - 1],
                        stop=self._knotvector[-(self._k)],
                        num=samples_per_n*self._n, endpoint=False)

        self._sample_points = np.zeros((self._control_pts.shape[0], len(t)))
        for n, _t in enumerate(t):
            for i in range(1, self._n + 2):
                N_ik = self._Nik(i, self._k, _t)
                self._sample_points[:, n] += N_ik * self._control_pts[:, i-1]

    def draw_sample_points(self, img, shift=8):
        shift_scale = 1 << shift
        for x, y in (0.5 + (self._sample_points.T*shift_scale)).astype('int64'):
            cv2.circle(img, (x, y), 12, (255, 255, 255),
                       thickness=-1, lineType=cv2.LINE_AA, shift=shift)

    def draw_endpoints(self, img):
        cv2.circle(img, tuple(
            self._control_pts[:, 0]), 4, (0, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(
            self._control_pts[:, -1]), 4, (0, 0, 255), lineType=cv2.LINE_AA)

    def draw(self, img, samples_per_n, show_ends=True):
        """
        draws the curve on a given image
        $$
          P(t) = \sum^{n+1}_{i=1}N_{i,k}P_i
        $$
        given $ t_{min} \lte t \lt t_{max} $
        """
        if self._sample_points is None or \
            samples_per_n*self._n != self._control_pts.shape[1]:
            self.gen_sample_points(samples_per_n)
        self.draw_sample_points(img)
        if show_ends:
            self.draw_endpoints(img)

    # this function is potentially cacheable for monotonic knotvectors
    # in this case N_ik (t) == N_ik(mod(t,k))
    def _Nik(self, i: int, k: int, t):
        """computes the knotting function see https://www.cl.cam.ac.uk/teaching/2000/AGraphHCI/SMEG/node4.html
        eq 89"""
        if k == 1:
            if self._knotvector[i-1] <= t and \
                    self._knotvector[i] > t:
                return 1
            else:
                return 0
        else:
            _i_km = self._Nik(i, k-1, t)
            if _i_km != 0:
                _i_km *= (
                    t - self._knotvector[i-1])/(self._knotvector[i + k - 2] - self._knotvector[i-1])
            _ip_km = self._Nik(i+1, k-1, t)
            if _ip_km != 0:
                _ip_km *= (self._knotvector[i + k - 1] - t) / \
                    (self._knotvector[i + k - 1] - self._knotvector[i])

            return _i_km + _ip_km

    def __str__(self):
        return f"{self._k} {self._n } {self._knotvector}"

    def to_dict(self):
        return dict(
            mt_pos=self._mt_pos,
            shape='bspline',
            order=self._order,
            k=self._k,
            knotvector=self._knotvector,
            controlpoints_x=self._control_pts[0,:],
            controlpoints_y=self._control_pts[1,:],
        )

    def boundingbox(self):
        if self._boundingbox_xy is None:
            # of the form [[x_min x_max][y_min y_max]]
            self._boundingbox_xy = np.stack([np.min(self._control_pts, axis=1),
                                             np.max(self._control_pts, axis=1)],
                                            axis=1)
        if self._boundingbox_aex is None:
            # of the form [[x_min x_width][y_min y_width]]
            self._boundingbox_aex = np.stack([self._boundingbox_xy[:, 0],
                                              self._boundingbox_xy[:, 1] - self._boundingbox_xy[:, 0]],
                                             axis=1)

        return (self._boundingbox_xy, self._boundingbox_aex)

class BSplineGroup():

    def __init__(self, samples_per_n):
        self._mt_pos = np.random.get_state()[2]
        self._samples_per_n = samples_per_n

        self._chr_centre = Bspline(order="cubic",
                                   boundingbox=[[100, 400], [100, 400]],
                                   max_complexity=6)
        self._chr_centre.gen_sample_points(self._samples_per_n)
        _, bb = self._chr_centre.boundingbox()
        bb_ul = [[bb[0, 0], bb[0, 0] + max(bb[1, 0]//2,50)],
                 [bb[0, 1], bb[0, 1]+100]]
        bb_ur = [[bb[0, 0]+bb[1, 0]//2, bb[0, 0] + max(bb[1, 0],50)],
                 [bb[0, 1], bb[0, 1]+100]]
        self._chr_accent_ul = Bspline(order="quadratic",
                                      boundingbox=bb_ul,
                                      max_complexity=1)
        self._chr_accent_ur = Bspline(order="quadratic",
                                      boundingbox=bb_ur,
                                      max_complexity=1)
    def n_subpaths(self):
        return 3

    def to_dict(self):
        d = {'mt_pos': self._mt_pos,
             'n_subpaths': 3}
        for k, v in self._chr_centre.to_dict().items():
            d[f"path[0].{k}"] = v
        for k, v in self._chr_accent_ul.to_dict().items():
            d[f"path[1].{k}"] = v
        for k,v in self._chr_accent_ur.to_dict().items():
            d[f"path[2].{k}"] = v
        return d    

    def draw(self, img, highlight_ends=False):
        self._chr_centre.draw(img, self._samples_per_n,
                             show_ends=highlight_ends)
        self._chr_accent_ul.draw(img, self._samples_per_n,
                                show_ends=highlight_ends)
        self._chr_accent_ur.draw(img, self._samples_per_n,
                             show_ends=highlight_ends)
    
    def draw_iterative_highlight_ends(self, img):
        for i in range(self.n_subpaths()):
            self._chr_centre.draw(img, self._samples_per_n,
                                show_ends=i==0)
            self._chr_accent_ul.draw(img, self._samples_per_n//3,
                                    show_ends=i==1)
            self._chr_accent_ur.draw(img, self._samples_per_n//3,
                                show_ends=i==2)
            yield i



if __name__ == "__main__":
    import cv2

    img=np.zeros((500, 500, 3), np.uint8)

    def new_char():
        bs_char = BSplineGroup(300)
        #bs_char.draw_bspline(img,'centre')
        print(bs_char.to_dict())
        return bs_char.draw_iterative_highlight_ends(img)

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

# Module that contains the Lane class. that facilitates all the data associated
# with one Lane.
"""This module only contains the Lane Class."""

from collections import deque
from itertools import chain
from functools import reduce

import numpy as np
import cv2

from scipy.interpolate import UnivariateSpline

class Lane(object):
    """Lane object that contains all data (mutable) for lane calculations."""

    Y_VALS = 101
    WEIGHT_HYPERPARAMETER = 0.8
    LINE_THICKNESS = 100
    # TODO: Change to dictionary
    Y_MTS_PER_PIX = 0.04
    X_MTS_PER_PIX = 0.005

    def __init__(self, base_pt, img_size, cache_length):
        # Principal attributes
        self._detected = False
        self._last_xfitted = deque(maxlen=cache_length)
        self._last_yfitted = deque(maxlen=cache_length)

        # Outside presentations
        self._current_fit = [np.array([False])]
        self._radius_curvature = None
        self._vehicle_center = 0.0

        self._current_xfit = None
        
        self._img_size = img_size
        self._base_pt = int(base_pt)
        
        self._yvals = np.linspace(0, self._img_size[0], Lane.Y_VALS)
        self._mask = np.zeros(self._img_size, dtype=np.uint8)
        self._dropped_frames = 0

    @staticmethod
    def curvature_splines(x, y, error=0.1):
        
        x, y = x*Lane.X_MTS_PER_PIX, y*Lane.Y_MTS_PER_PIX

        t = np.arange(x.shape[0])
        std = error * np.ones_like(x)

        fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
        fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

        xˈ = fx.derivative(1)(t)
        xˈˈ = fx.derivative(2)(t)
        yˈ = fy.derivative(1)(t)
        yˈˈ = fy.derivative(2)(t)
        curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
        # return np.amax(curvature)
        return np.mean(curvature)

    @staticmethod
    def compute_rad_curv(xvals, yvals):
        fit_cr = np.polyfit(yvals*Lane.Y_MTS_PER_PIX, xvals*Lane.X_MTS_PER_PIX, 2)
        y_eval = np.max(yvals)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad

    @property
    def detected(self):
        return self._detected

    @property
    def current_xfit(self):
        return self._current_xfit

    @property
    def yvals(self,):
        return self._yvals
    
    @property
    def radius_of_curvature(self):
        return self._radius_curvature

    @radius_of_curvature.setter
    def radius_of_curvature(self, value):
        self._radius_curvature = value

    @property
    def dropped_frames(self):
        return self._dropped_frames

    def detect_from_mask(self, image):
        mask_lanes = cv2.bitwise_and(image, self._mask)
        all_pts = cv2.findNonZero(mask_lanes)
        if all_pts is not None:
            all_pts = all_pts.reshape((-1,2))
            self.add_lane_pixels(all_pts[:,0], all_pts[:,1])
        else:
            self._detected = False

    def add_lane_pixels(self, x, y):

        weights = np.ones(len(self._last_xfitted))
        if len(weights) > 1:
            weights[0] = Lane.WEIGHT_HYPERPARAMETER
            weights[1:] = (1-Lane.WEIGHT_HYPERPARAMETER)/(len(weights) - 1)

            def _weight_product(intercept, container_list):
                return intercept + container_list[0]*container_list[1]

            grid = np.zeros(len(self._yvals))
            w_x = reduce(_weight_product, zip(weights, self._last_xfitted), grid)
            w_y = reduce(_weight_product, zip(weights, self._last_yfitted), grid)

        else:
            w_x, w_y = [], []

        x_hist = np.fromiter(chain(w_x, x), dtype=np.int32)
        y_hist = np.fromiter(chain(w_y, y), dtype=np.int32)

        quadratic_lane = np.polyfit(y_hist, x_hist, 2)
        # rad_curvature = self.curvature_splines(x_hist, y_hist)
        rad_curvature = self.compute_rad_curv(x_hist, y_hist)

        if self._radius_curvature is None:
            self._detected = True
        else:
            self._vehicle_center = abs(rad_curvature - self._radius_curvature)/self._radius_curvature
            self._detected = self._vehicle_center < 0.3


        if self._detected:
            x_axis_fit = np.polyval(quadratic_lane, self._yvals)
            self._current_xfit = x_axis_fit

            self._last_xfitted.append(x_axis_fit)
            self._last_yfitted.append(self._yvals)

            self._radius_curvature = rad_curvature
            self._current_fit = quadratic_lane
            self._dropped_frames = 0

        else:
            quadratic_lane = self._current_fit
            rad_curvature = self._radius_curvature
            x_axis_fit = np.polyval(quadratic_lane, self._yvals)
            self._dropped_frames += 1

        map_pts = np.transpose(np.vstack([x_axis_fit, self._yvals])).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(self._mask, map_pts, -1, (255, 255, 255), thickness=Lane.LINE_THICKNESS)

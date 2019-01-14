# Long due solution to the finding lanes on an image.
import pickle
import os

import numpy as np
import cv2

from collections import deque
from itertools import chain
from functools import reduce

from fire import Fire
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks_cwt

# PARAMETERS:
LEFT_LANE_BASE_PT = 0.15
RIGHT_LANE_BASE_PT = 0.60
Y_PERCENTAGE = 0.075

class Lane(object):

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
    def compute_radius_curv(xvalues, yvalues):
        # TODO: add curvature compute formula
        cr = np.polyfit(yvalues*Lane.Y_MTS_PER_PIX, xvalues*Lane.X_MTS_PER_PIX, 2)
        return (1 + (2*cr[0]*(np.max(yvalues)) + cr[1]**2)**1.5) / np.absolute(2*cr[0])

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
                # assert not container_list, "container has only one element"
                return intercept + container_list[0]*container_list[1]

            grid = np.zeros(len(self._yvals))
            w_x = reduce(_weight_product, zip(weights, self._last_xfitted), grid)
            w_y = reduce(_weight_product, zip(weights, self._last_yfitted), grid)

        else:
            w_x, w_y = [], []

        x_hist = np.fromiter(chain(w_x, x), dtype=np.int32)
        y_hist = np.fromiter(chain(w_y, y), dtype=np.int32)

        quadratic_lane = np.polyfit(y_hist, x_hist, 2)
        rad_curvature = self.compute_radius_curv(x_hist, y_hist)

        if self._radius_curvature is None:
            self._detected = True
        else:
            self._vehicle_center = abs(rad_curvature - self._radius_curvature)/self._radius_curvature
            self._detected = self._vehicle_center < 0.5


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

def reject_outliers(x_list, y_list):
    if not x_list or not y_list:
        return x_list, y_list
    mu_x, mu_y = np.mean(x_list), np.mean(y_list)
    sig_x, sig_y = np.std(x_list), np.std(y_list)
    new_x, new_y = zip(*[(x, y) for (x,y) in zip(x_list, y_list)
                                 if abs(x - mu_x) < 2*sig_x and abs(y - mu_y) < 2*sig_y])
    return new_x, new_y


def sliding_window(image, left_lane, right_lane, base_pts, num_bands = 10, window_width = 0.2):
    """Uses histogram and sliding window to detect lanes from scratch"""

    height = image.shape[0]
    band_height = int(1./num_bands * height)   # Divide image into horizontal bands
    band_width = int(window_width*image.shape[1])

    l_x, l_y, r_x, r_y = [], [], [], []

    base_left, base_right = base_pts

    idx_left, idx_right = base_pts
    for i in reversed(range(num_bands)):
        w_left = image[i*band_height:(i+1)*band_height,base_left-band_width//2:base_left+band_width//2]
        w_right = image[i*band_height:(i+1)*band_height,base_right-band_width//2:base_right+band_width//2]

        left_y_pt, left_x_pt = np.nonzero(w_left)
        right_y_pt, right_x_pt = np.nonzero(w_right)

        l_x.extend(left_x_pt + base_left-band_width//2)
        l_y.extend(left_y_pt + i*band_height)
        r_x.extend(right_x_pt+ base_right-band_width//2)
        r_y.extend(right_y_pt+ i*band_height)

        # Find 'x' with maximum nonzero elements as baseline for next window
        s_left = np.sum(w_left, axis=0)
        s_right = np.sum(w_right, axis=0)
        if np.any(s_left > 0):
            base_left = np.argmax(s_left) + base_left-band_width//2
        if np.any(s_right > 0):
            base_right = np.argmax(s_right) + base_right-band_width//2

    l_x, l_y = reject_outliers(l_x, l_y)
    r_x, r_y = reject_outliers(r_x, r_y)

    left_lane.add_lane_pixels(l_x, l_y)
    right_lane.add_lane_pixels(r_x, r_y)

    return left_lane, right_lane



def find_edges(image, ):
    # Remove noise by blurring with a Gaussian filter
    image = cv2.GaussianBlur(image, (5, 5), 0)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    #gray = (0.5*image[:,:,0] + 0.4*image[:,:,1] + 0.1*image[:,:,2]).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    # _, gray_binary = cv2.threshold(gray.astype('uint8'), 130, 255, cv2.THRESH_BINARY)
    
    # switch to gray image for laplacian if 's' doesn't give enough details
    total_px = image.shape[0]*image.shape[1]
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    cv2.convertScaleAbs(laplacian, laplacian)
    #mask_one = (laplacian < 0.15*np.min(laplacian)).astype(np.uint8)
    
    mask_one = (laplacian < 0.27*np.min(laplacian)).astype(np.uint8)
    
    #_, s_binary = cv2.threshold(s.astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, s_binary = cv2.threshold(s.astype('uint8'), 190, 255, cv2.THRESH_BINARY)
    mask_two = s_binary


    combined_binary = np.clip(cv2.bitwise_and(gray_binary,
                        cv2.bitwise_or(mask_one, mask_two)), 0, 1).astype('uint8')

    return combined_binary

def find_perspective_points(image):
    edges = find_edges(image)

    # Computing perspective points automatically
    rho = 2              # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 100       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 25    # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, min_line_length, max_line_gap)

    def _calculate_lane_markers(lines, angle_min=20*np.pi/180, angle_max=65*np.pi/180):
        lane_markers_x = [[], []]
        lane_markers_y = [[], []]
        for line in lines:
            for x1,y1,x2,y2 in line:
                theta = np.arctan2(y1-y2, x2-x1)
                if all([abs(theta) >= angle_min, abs(theta) <= angle_max]):
                    # TODO: 3 way operator
                    if theta > 0: 
                        i = 0 
                    else:
                        i = 1 # Right lane marker
                    lane_markers_x[i].append(x1)
                    lane_markers_x[i].append(x2)
                    lane_markers_y[i].append(y1)
                    lane_markers_y[i].append(y2)

        return lane_markers_x, lane_markers_y

    lane_mark_x, lane_mark_y = _calculate_lane_markers(lines)

    
    if any([not lane_mark_x[0], not lane_mark_x[1]]):
        return None

    p_right = np.polyfit(lane_mark_y[1], lane_mark_x[1], 1)
    p_left = np.polyfit(lane_mark_y[0], lane_mark_x[0], 1)

    def _compute_src_apex(p_left, p_right, edges):

        apex_pt = np.linalg.solve([[p_left[0], -1], [p_right[0], -1]], [-p_left[1], -p_right[1]])

        y_size = edges.shape[0]
        
        top_y = np.ceil(apex_pt[0] + Y_PERCENTAGE*y_size)

        bl_pt = np.ceil(np.polyval(p_left, y_size))
        tl_pt = np.ceil(np.polyval(p_left, top_y))

        br_pt = np.ceil(np.polyval(p_right, y_size))
        tr_pt = np.ceil(np.polyval(p_right, top_y))

        src = np.array([[tl_pt, top_y],
                        [tr_pt, top_y],
                        [br_pt, y_size],
                        [bl_pt, y_size]], np.float32)

        return src

    src = _compute_src_apex(p_left, p_right, edges)

    get_perspective_transform(edges, src_in = src, dst_in = None, display=False)
    return src



def get_perspective_transform(image, src_in = None, dst_in = None, display=False):
    img_size = image.shape
    if src_in is None:
        src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
                        [705. /1280.*img_size[1], 455./720.*img_size[0]],
                        [1130./1280.*img_size[1], 720./720.*img_size[0]],
                        [190. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
    else:
        src = src_in

    if dst_in is None:
        dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 720./720.*img_size[0]],
                        [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
    else:
        dst = dst_in

    warp_m = cv2.getPerspectiveTransform(src, dst)
    warp_minv = cv2.getPerspectiveTransform(dst, src)

    return warp_m, warp_minv



def histogram_base_points(lanes, min_peak = 25.0):
    """Uses histogram to find possible base points for lane lines"""
    hist = np.sum(lanes[int(lanes.shape[0]*0.5):,:], axis=0)

    widths = [100]
    idx = find_peaks_cwt(hist, widths, max_distances=widths, noise_perc=50)
    if len(idx) < 2:
        return None

    # Avoid edges
    idx = [i for i in idx if i > lanes.shape[1]*0.1
                             and i < lanes.shape[1]*0.9
                             and max(hist[i-50:i+50]) > min_peak]

    return [min(idx), max(idx)]





def frame_pipeline(frame,  key_frame_interval=20, cache_length=20,):

    # TODO: if camera properties does not exist on disk, run and create the file
    with open("camera_properties.pkl", "rb") as fd:
        camera_properties = pickle.load(fd)

    # if frame_pipeline.cache is None:
    # if true:
    if frame_pipeline.cache is None:

        left_lane = Lane(LEFT_LANE_BASE_PT*frame.shape[0], frame.shape[:2], cache_length=cache_length)

        right_lane = Lane(RIGHT_LANE_BASE_PT*frame.shape[0], frame.shape[:2], cache_length=cache_length)

        cache = dict(
            cam_mtx = camera_properties["mtx"],
            cam_dist = camera_properties["dist"],
            warp_m = None,
            frame_ctr = 0,
            left = left_lane,
            right = right_lane,
            base_pts = None
        )

    else:
        cache = frame_pipeline.cache

    left_lane, right_lane = cache['left'], cache['right']

    mtx = camera_properties["mtx"]
    dist = camera_properties["dist"]
    undist = cv2.undistort(frame, mtx, dist, None, mtx)

    if cache['warp_m'] is None:
        src = find_perspective_points(undist)
        warp_m, warp_minv = get_perspective_transform(frame, src_in = src)

        if src is not None:
            # Save only if customized perspective transform is found
            cache['warp_m'] = warp_m
            cache['warp_minv'] = warp_minv
    else:
        warp_m, warp_minv = cache['warp_m'], cache['warp_minv']

    edges = find_edges(undist)
    warp_edges = cv2.warpPerspective(edges, warp_m, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    base_pts = cache['base_pts']
    if base_pts is None: #or cache['frame_ctr'] % key_frame_interval == 0:
        new_base_pts = histogram_base_points(warp_edges)

        if new_base_pts is not None:
            base_pts = new_base_pts
        else:
            # Could not find new base points
            # Re-use previous data if base points could not be found
            cache['frame_ctr'] = cache['frame_ctr'] - 1 # Make sure we try again in the next frame
            return undist

    if ((left_lane.current_xfit is None or left_lane.dropped_frames > 16)
            or (right_lane.current_xfit is None or right_lane.dropped_frames > 16)):
        # Detect from scratch
        left_lane.radius_of_curvature = None
        right_lane.radius_of_curvature = None
        sliding_window(warp_edges, left_lane, right_lane, base_pts)
    else:
        left_lane.detect_from_mask(warp_edges)
        right_lane.detect_from_mask(warp_edges)

    cache['frame_ctr'] = cache['frame_ctr'] + 1
    cache['base_pts'] = base_pts
    frame_pipeline.cache = cache

    # Create an image to draw the lines on
    color_warp = np.zeros_like(frame).astype(np.uint8)

    yvals = left_lane.yvals
    left_fitx = left_lane.current_xfit
    right_fitx = right_lane.current_xfit

    # Create an image to draw the lines on
    color_warp = np.zeros_like(frame).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # TODO: Change
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Draw lane markers
    pts = np.transpose(np.vstack([left_lane.current_xfit, left_lane.yvals])).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (255,0,0), thickness=30)
    pts = np.transpose(np.vstack([right_lane.current_xfit, right_lane.yvals])).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (0,0,255), thickness=30)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, warp_minv, (frame.shape[1], frame.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_r = left_lane.radius_of_curvature
    right_r = right_lane.radius_of_curvature
    middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = frame.shape[1]//2

    dx = (veh_pos - middle)*Lane.X_MTS_PER_PIX # Positive if on right, Negative on left

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Left radius of curvature  = %.2f m'%(left_r),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Right radius of curvature = %.2f m'%(right_r),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110),
                        font, 1,(255,255,255),2,cv2.LINE_AA)

    is_tracking = left_lane.detected or right_lane.detected
    cv2.putText(result,'Tracking Locked' if is_tracking else 'Tracking Lost',(50,140),
            font, 1,(0,255,0) if is_tracking else (255,0,0), 3,cv2.LINE_AA)

    cache['left'] = left_lane
    cache['right'] = right_lane

    return result


def _clear_cache():
    frame_pipeline.cache = None

def main(filepath_source, filepath_destination):
    # TODO: if camera properties does not exist on disk, run and create the file
    # with open("camera_properties.pkl", "rb") as fd:
    #     cam_propeties = pickle.load(fd)


    

    _clear_cache()
    print("Processing video {}".format(filepath_source))
    assert os.path.isfile(filepath_source), "{} is not a file".format(filepath_source)
    original_vclip = VideoFileClip(filepath_source)
    processed_clip = original_vclip.fl_image(frame_pipeline)
    processed_clip.write_videofile(filepath_destination, audio=False)

if __name__ == "__main__":
    Fire(main)

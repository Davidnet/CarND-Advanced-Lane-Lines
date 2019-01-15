# Long due solution to the finding lanes on an image.
import pickle
import os

import numpy as np
import cv2

from fire import Fire
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks_cwt

from utils import get_perspective_transform, reject_outliers, find_edges

from lane import Lane

# PARAMETERS:
LEFT_LANE_BASE_PT = 0.15
RIGHT_LANE_BASE_PT = 0.60
Y_PERCENTAGE = 0.075

def sliding_window(image, left_lane, right_lane, base_pts, num_bands=10, window_width=0.2):

    height = image.shape[0]
    band_height = int(1./num_bands * height)   # Divide image into horizontal bands
    band_width = int(window_width*image.shape[1])//2 # We just care about half


    def _generate_candidates(base_pts, band_height, band_width, num_bands):

        l_x, l_y, r_x, r_y = [], [], [], []

        base_left, base_right = base_pts

        for i in range(num_bands, -1, -1):
            w_left = image[i*band_height:(i+1)*band_height, base_left-band_width:base_left+band_width]
            w_right = image[i*band_height:(i+1)*band_height, base_right-band_width:base_right+band_width]

            left_y_pt, left_x_pt = np.nonzero(w_left)
            right_y_pt, right_x_pt = np.nonzero(w_right)

            l_y.extend(left_y_pt + i*band_height)
            l_x.extend(left_x_pt + base_left-band_width)
            r_y.extend(right_y_pt+ i*band_height)
            r_x.extend(right_x_pt+ base_right-band_width)

            s_left = np.sum(w_left, axis=0)
            s_right = np.sum(w_right, axis=0)
            if np.any(s_left > 0):
                base_left = np.argmax(s_left) + base_left-band_width
            if np.any(s_right > 0):
                base_right = np.argmax(s_right) + base_right-band_width

        return l_x, l_y, r_x, r_y

    l_x, l_y, r_x, r_y = _generate_candidates(base_pts, band_height, band_width, num_bands)

    
    l_x, l_y = reject_outliers(l_x, l_y)
    r_x, r_y = reject_outliers(r_x, r_y)

    left_lane.add_lane_pixels(l_x, l_y)
    right_lane.add_lane_pixels(r_x, r_y)

    return left_lane, right_lane




def find_perspective_points(image):
    edges = find_edges(image)

    # Taken from the tutorial
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

    def _calculate_lane_markers(lines, angle_min=20*np.pi/180, angle_max=65*np.pi/180):
        lane_markers_x = [[], []]
        lane_markers_y = [[], []]
        for line in lines:
            for x1, y1, x2, y2 in line:
                theta = np.arctan2(y1-y2, x2-x1)
                if all([abs(theta) >= angle_min, abs(theta) <= angle_max]):

                    i = 0 if theta > 0 else 1
                    [lane_markers_x[i].append(coor) for coor in [x1, x2]]
                    [lane_markers_y[i].append(coor) for coor in [y1, y2]]

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

    get_perspective_transform(edges, src_in=src)
    return src

def histogram_base_points(lanes, min_peak=25.0, edge_percentage=0.9):
    hist = np.sum(lanes[int(lanes.shape[0]*0.5):, :], axis=0)

    idx = find_peaks_cwt(hist, [100], max_distances=[100], noise_perc=50)

    # Doesn't make sense if there are less than two lanes
    if len(idx) < 2:
        return None

    # Avoid edges
    idx = [i for i in idx if i > lanes.shape[1]*(1-edge_percentage)
           and i < lanes.shape[1]*edge_percentage
           and max(hist[i-50:i+50]) > min_peak]

    return [min(idx), max(idx)]


def frame_pipeline(frame, camera_properties, cache_length=20):


    if frame_pipeline.cache is None:

        left_lane = Lane(LEFT_LANE_BASE_PT*frame.shape[0], 
                         frame.shape[:2], cache_length=cache_length)

        right_lane = Lane(RIGHT_LANE_BASE_PT*frame.shape[0], 
                          frame.shape[:2], cache_length=cache_length)

        cache = dict(
            cam_mtx=camera_properties["mtx"],
            cam_dist=camera_properties["dist"],
            warp_m=None,
            left=left_lane,
            right=right_lane,
            base_pts=None
        )

    else:
        cache = frame_pipeline.cache

    left_lane, right_lane = cache['left'], cache['right']

    mtx = camera_properties["mtx"]
    dist = camera_properties["dist"]
    undist = cv2.undistort(frame, mtx, dist, None, mtx)

    if cache['warp_m'] is None:
        src = find_perspective_points(undist)
        warp_m, warp_minv = get_perspective_transform(frame, src_in=src)

        if src is not None:
            # Save only if customized perspective transform is found
            cache['warp_m'] = warp_m
            cache['warp_minv'] = warp_minv
    else:
        warp_m, warp_minv = cache['warp_m'], cache['warp_minv']

    edges = find_edges(undist)
    warp_edges = cv2.warpPerspective(edges, warp_m, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    base_pts = cache['base_pts']
    if base_pts is None:
        new_base_pts = histogram_base_points(warp_edges)

        if new_base_pts is not None:
            base_pts = new_base_pts
        else:
            # Could not find new base points
            return undist

    if any((left_lane.current_xfit is None or left_lane.dropped_frames > 16, 
            right_lane.current_xfit is None or right_lane.dropped_frames > 16)):
        left_lane.radius_of_curvature = None
        right_lane.radius_of_curvature = None
        sliding_window(warp_edges, left_lane, right_lane, base_pts)
    else:
        left_lane.detect_from_mask(warp_edges)
        right_lane.detect_from_mask(warp_edges)

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

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))


    # Draw lane markers
    pts = np.transpose(np.vstack([left_lane.current_xfit, left_lane.yvals])).reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (255, 0, 0), thickness=30)
    pts = np.transpose(np.vstack([right_lane.current_xfit, right_lane.yvals])).reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (0, 0, 255), thickness=30)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, warp_minv, (frame.shape[1], frame.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_r = left_lane.radius_of_curvature
    right_r = right_lane.radius_of_curvature
    middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = frame.shape[1]//2

    dx = (veh_pos - middle)*Lane.X_MTS_PER_PIX # Positive if on right, Negative on left

    results_str = '{} radii of curvature = {:.2f}'

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(result, results_str.format("left", left_r), (80, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, results_str.format("right", right_r), (80, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(result, "Vehicle postion: {:.2f} m {} of center".format(abs(dx), 'left' if dx < 0 else 'right'), (80, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    is_tracking = left_lane.detected or right_lane.detected

    cv2.putText(result, 'Tracking Locked' if is_tracking else 'Tracking Lost', (80, 140), font, 1, (0, 255, 0) if is_tracking else (255, 0, 0), 3, cv2.LINE_AA)

    cache['left'] = left_lane
    cache['right'] = right_lane

    return result


def main(filepath_source, filepath_destination):

    while True:
        try:
            with open("camera_properties.pkl", "rb") as fd:
                camera_properties = pickle.load(fd)

            break

        except FileNotFoundError:
            from camera_calibration import create_camera_pickle
            create_camera_pickle()

    frame_pipeline_cam = lambda x: frame_pipeline(x, camera_properties)
    frame_pipeline.cache = None

    print("Processing video {}".format(filepath_source))
    assert os.path.isfile(filepath_source), "{} is not a file".format(filepath_source)
    original_vclip = VideoFileClip(filepath_source)
    processed_clip = original_vclip.fl_image(frame_pipeline_cam)
    processed_clip.write_videofile(filepath_destination, audio=False)

if __name__ == "__main__":
    Fire(main)

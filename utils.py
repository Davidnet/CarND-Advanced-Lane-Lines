"""Collections of functions that helps to build the lane finder package."""

import cv2
import numpy as np

def get_perspective_transform(image, src_in=None, dst_in=None):
    if src_in is None:
        src = np.array([[585. , 455],
                        [705. , 455],
                        [1130., 720],
                        [190. , 720]], np.float32)
    else:
        src = src_in

    if dst_in is None:
        dst = np.array([[300., 100.],
                        [1000, 100.],
                        [1000, 720.],
                        [300., 720.]], np.float32)
    else:
        dst = dst_in

    warp_m = cv2.getPerspectiveTransform(src, dst)
    warp_m_inv = cv2.getPerspectiveTransform(dst, src)

    return warp_m, warp_m_inv

def reject_outliers(x_list, y_list):
    if not x_list or not y_list:
        return x_list, y_list
    mu_x, mu_y = np.mean(x_list), np.mean(y_list)
    sig_x, sig_y = np.std(x_list), np.std(y_list)
    new_x, new_y = zip(*[(x, y) for (x ,y) in zip(x_list, y_list)
                         if abs(x - mu_x) < 2*sig_x and abs(y - mu_y) < 2*sig_y])
    return new_x, new_y

def find_edges(image):
    # Remove noise by blurring with a Gaussian filter
    image = cv2.GaussianBlur(image, (5, 5), 0)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    #gray = (0.5*image[:,:,0] + 0.4*image[:,:,1] + 0.1*image[:,:,2]).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    # _, gray_binary = cv2.threshold(gray.astype('uint8'), 130, 255, cv2.THRESH_BINARY)
    
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

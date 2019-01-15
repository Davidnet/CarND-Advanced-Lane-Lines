# Module that contains the loop and main definitions for calibrate cameras.
# Usage example $ python camera_calibration.py '["calibration1.jpg"]'
# the above is a list of file to save in the example
"""Calibration module for images stored in memory.
"""
import os
import pickle
import numpy as np
import cv2
from fire import Fire

# Folder parameters
CALIBRATION_FOLDER = './camera_cal'
DESTINATION_FOLDER = './output_images/calibration_pipeline'
EXAMPLE_FOLDER = './output_images'


# Chessboard parameters
NX = 9
NY = 6

# Camera properties
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
CAMERA_PROPERTIES = ["mtx", "dist"]
CAMERA_IMAGE_SHAPE = (720, 1280)
CAMERA_PICKLE_FILEPATH = './camera_properties.pkl'



def draw_image_and_save(filepath,
                        nx,
                        ny,
                        corners,
                        ret,
                        destination_folder=DESTINATION_FOLDER):
    """Function that draws and save image to a destination folder."""
    img = cv2.imread(filepath)
    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    filename = os.path.basename(filepath)
    file_info = os.path.splitext(filename)
    write_name = f"{file_info[0]}_with_chessboard.{file_info[1]}"
    cv2.imwrite(os.path.join(destination_folder, write_name), img)
    # breakpoint()
    cv2.imshow(write_name, img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
    return None


def process_single_image(filepath, nx, ny):
    """Function that process a filepath to check wheter there is a valid chessboard
    pattern.
    """
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        return ret, corners
    else:
        return None

def calculate_obj_img_points(calibration_folder=CALIBRATION_FOLDER,
                            destination_folder=DESTINATION_FOLDER,
                            nx=NX,
                            ny=NY):
    """Function that contains the main loop for camera calibration."""

    # Prepare object points
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store tobject points and image points from all images.
    objpoints = []
    imgpoints = []

    for dir_entry in os.scandir(calibration_folder):
        if all([dir_entry.is_file, not dir_entry.name.startswith('.')]):
            try:
                ret, corners = process_single_image(dir_entry.path, nx, ny)
            except TypeError:
                print(f"Chessboard not detected at {dir_entry.path}")
            else:
                objpoints.append(objp)
                imgpoints.append(corners)
                draw_image_and_save(dir_entry.path, nx, ny, corners, ret)

    return objpoints, imgpoints



def create_camera_pickle(calibration_folder=CALIBRATION_FOLDER,
                        destination_folder=DESTINATION_FOLDER,
                        nx=NX,
                        ny=NY,
                        camera_properties=CAMERA_PROPERTIES,
                        camera_image_shape=CAMERA_IMAGE_SHAPE,
                        camera_pickle_filepath=CAMERA_PICKLE_FILEPATH,
                        ):
    """Create pickle that contains the camera pickle properties"""

    objpoints, imgpoints = calculate_obj_img_points(calibration_folder,
                                                     destination_folder,
                                                     nx,
                                                     ny,
    )

    # Be consistent with exercises
    img_size = (camera_image_shape[1], camera_image_shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    calibration_params = dict(
        ret=ret,
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    camera_pickle = { key: calibration_params[key] for key in camera_properties }
    with open(camera_pickle_filepath, "wb") as fd:
        pickle.dump(camera_pickle, fd)
    return None

def load_camera_pickle(camera_pickle_filepath=CAMERA_PICKLE_FILEPATH,):
    "load camera pickle"
    with open(camera_pickle_filepath, 'rb') as fd:
        return pickle.load(fd)



def main(image_filenames,
        calibration_folder=CALIBRATION_FOLDER,
        destination_folder=DESTINATION_FOLDER,
        nx=NX,
        ny=NY,
        camera_properties=CAMERA_PROPERTIES,
        camera_image_shape=CAMERA_IMAGE_SHAPE,
        camera_pickle_filepath=CAMERA_PICKLE_FILEPATH,
        example_folder=EXAMPLE_FOLDER,):

        """Main loop for calibration."""
        create_camera_pickle(calibration_folder,
                            destination_folder,
                            nx,
                            ny,
                            camera_properties,
                            camera_image_shape,)
        camera_pickle = load_camera_pickle(camera_pickle_filepath)
        for image_fname in image_filenames:
            img = cv2.imread(os.path.join(calibration_folder, image_fname))
            # breakpoint()
            cv2.imshow("img", img)
            cv2.waitKey(150)

            mtx = camera_pickle["mtx"]
            dist = camera_pickle["dist"]

            dst = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imshow("dst", dst)
            cv2.waitKey(150)
            # cv2.imwrite(os.path.join(example_folder, image_fname), dst)
            cv2.imwrite(os.path.join(example_folder, os.path.splitext(image_fname)[0] + "undistorted" + os.path.splitext(image_fname)[1]), dst)
            cv2.destroyAllWindows()

        return None



if __name__ == "__main__":
    Fire(main)


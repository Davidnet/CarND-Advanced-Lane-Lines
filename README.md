**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1undistorted.jpg "Undistorted"
[image2]: ./camera_cal/calibration1.jpg "Original"
[image3]: ./output_images/HSV.png "HSV Channels"
[image4]: ./output_images/Laplacians.png "Laplace"
[image5]: ./output_images/Perspective "Perspectives"
<!-- [image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video" -->

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in a camera_calibration CLI located at the root of the project.

At first I create a function `calculate_ob_img_points` that cretes `objp` points which are just the x, y and z coordinates of the chessboard corners in the world. It is important to remark that I do certain verifications that the `calibration_folder` and the `destination_folder` exists and are can be written. For each image I create a simple pipeline that is contained on the lines 50 to 60 of the `camera_calibration.py`. So that for each image I bring the image as a BGR array in memory, I transformed this image to gray using `cv2.COLOR_BGR2GRAY` for retriving the corners of the chessboard (which if you call the script I show using `cv2.imshow`), this results for each image to be chainned in `objpoints` and `imgpoints`, to compute the camera calibration and distortion coefficients using `cv2.calubrateCamera()` function. I applied this distortion correction to  

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

#### 1. Provide an example of a distortion-corrected image.

### Undistorted and Distorted
![generated image][image1]

* Original Image
![original untocuhed image][image2]

For all test images there is a notebook with the images called [camera-corrections.ipynb](./notebooks/camera_corrections)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The main module `find_lane_lines.py` and `utilis.py` contains the main loop that have the full pipeline for each frame. A previous exploration on the test images done in [Colorspace_thresholds.ipynb](./notebooks/Colorspace_thresholds.ipynb) showed that the best channel for find a thresholed binary was to use the `s` channel, and (if you follow the notebook) it is shown that a Laplacian computation is much better at finding the lanes. so in lines 37 to 63 the procedure `find_edges` is choosen to have the s channel and compute the laplacian, I decied to go for a second-order derivate since I wanted to explore if having the information of a second derivate makes the detection much better. 

#### Examples:


Test image channels:

![multiple test images][image3]

Laplacian test:

![Laplacian image][image4]

As you can see it seems to be a good combination!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included ina `utils.py` module
I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 455      | 300, 100      | 
| 705, 455      | 1000, 100      |
| 1130, 720     | 1000, 720      |
| 190, 720      | 300, 720      |

These are bgr images so pltshow shows this: The notebook actually uses imshow.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two main pieces that I want to remark on which I used `np.polyfit`. The first one is to fit the lanes and is in a public method `add_lane_pixels` on the `Lane` class located in `lane.py`. I try to fit a quadratic function on the points that are detected. The second one is to calculate the curvature, I decided to try to make splines (which could be an overkill for lanes in traffic), e.g is an staticmethod on the lane class:

```python
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
```
I take the mean of these curvatures and I report this as the curvature of the figure in general


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Lines 45 to 61 in the `Lane` class in `lane.py`

**Review 2**: See now lines 64 to 69 in the `Lane` class in `lane.py`. My approach of calculating curvature via splines was not estable nor accurate. My udacity review pinpoint at my mistake, so I decided to take the old routine of assuming a quadratic curve and calculating the curvature.


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


## First Attempt
Here's a [link to my video result](https://youtu.be/EqHch__5o7A)
Youtube link

## Second attempt
Here's a [link to my video result](https://youtu.be/InOlf8-j8-Q)
Youtube link




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project was extremly hard for me, at first the sobel (first dervitives) were taken me nowhere, I decided to search in dsp books (MIT Course) and it was a suggestion of taking infomration of the second derivative (laplacian). The whole pipeline could be resumed as:

```
    check_cache -> find_edges -> compute splines -> histogram -> find_peaks
```

I try to use different ways on finding peaks within the histogram points, and I was suprised to found a method that uses a wavelet approach (which I really want to study in the future). it works mainly well but there are too many parameters to tweak for this project that I found that this might not be a scalable solution.

In retrospective I found that I spent a lot of time doing feature-engineering and manual engineering for solving this. I might wonder if a deep learning approach could be more robust.

In the future I will try to make a more optimal way of finding all the parameters, but right now is very tiresome to do manual search of parameters.

I tried on the hardest challange, and to be honest is better that I expected:

Here's a [link to my video](https://youtu.be/KuqcBA48pKM)

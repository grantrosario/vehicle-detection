#Vehicle Detection
---


[//]: # (Image References)
[detect]: output_images/basic_detect.png
[carhog]: output_images/car_hog.png
[carnotcar]: output_images/car_notcar.png
[grayhog]: output_images/grayscale_hog.png
[heatmap]: output_images/heatmap.png
[notcarhog]: output_images/notcar_hog.png
[labels]: output_images/labels.png
[final]: output_images/final.png
[test1]: output_images/result1.png
[test2]: output_images/result2.png
[test3]: output_images/result3.png
[video1]: project_attempt.mp4


###Histogram of Oriented Gradients (HOG)

####1. HOG Features

The code for this step is contained in the first code cell of the IPython notebook (or in lines 59 through 71 of the file called `util.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][carnotcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on a car and not_car image:
![alt text][carhog]
![alt text][notcarhog]

####2. HOG Parameters

I tried various combinations of parameters and after numerous and long-winded trial and error attempts, I decided on using all of the HOG-channels with an orientation of 9, 8 pixels per cell, and 2 cells per block.

####3. Description

I trained a linear SVM (lines 9 through 34 of `pipeline.py`) using 3 main features, spatial binning, color histogram, and HOG features extraction. I then normalized the features and split them into test and training data before running them trough the linear SVM.

###Sliding Window Search

####1. Description

I decided to implement a sliding window approach via hog sub-sampling in lines 127 - 199 of `util.py`. I landed on using two scales. One scale of 1 across the middle of the image to scan for far away cars and another scale of 1.5 across the bottom portion to scan for closer cars. I also used 2 cells per step which translates to an overlap of 75%. These gave me the best results of my trials.

####2. Test images

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][test1]
![alt text][test2]
![alt text][test3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](project_attempt.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (lines 191 - 198 of `util.py`).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions using `apply_threshold` (lines 211 - 215).  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (line 50 of `pipeline.py`). I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame and its corresponding heatmaps:

![alt text][heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][labels]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][final]



---

###Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The approach I took consisted of running the training images through a linear SVM and scanning several windows across the bottom half of the image generated via hog sub-sampling. The most significant issues I faced was inconsistency in regards to the bounding boxes around detected vehicles (shaky boxes) and boxes disappearing. This was dramatically alleviated by averaging the heatmaps across 10 frames and drawing boxes around that average as well as generating windows across 2 scales, one for far away cars and one for close cars.

Overall, I don't believe this pipeline is very robust. It seems like it gets shaky around the pavement change and signficant shadows.

I would be very interested to learning how to apply a deep learning approach to this project to alleviate CV shortcomings.


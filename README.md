# Curling with Computer Vision
***Extending non-learning (mostly) computer vision techniques to more advanced concepts like Color Space Transformations, Hough Circles, CSRT Tracking***

## Task 1: Counting Red and Yellow Stones in Curling

Task1 - this directory contains 25 training images showing the red and yellow stones on the ice surface in a constrained scenario.

In this scenario, each image shows The House from above. The Task1 consists in correctly counting how many stones are on the ice surface and how many of these stones are red and yellow.

The format that you need to follow is the one used in the ground-truth files (located in the subdirectory ground-truth) with the first line containing the total number of stones and the second and third lines containing the number of red and yellow stones.

## Proposed Solution

For each image that we have:
  - We apply a conversion to grayscale and a median blur with a kernel size of 3
  - On the obtained image we apply Hough Circles for detecting all the stones with the risk of detecting false positives
  - For reducing false positives and validating circles we apply to each:
      - a conversion to HSV Space for color detection
      - we apply a red and yellow mask over the image
      - compute the mean color over the pixels in a circle smaller contained by the original detection
      - we apply a threshold over the obtained mean to predict red or yellow circles
  - After this step we count the remaining detections

## Examples

![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-1-image-16.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-1-image-19.png)
:-------------------------:|:-------------------------:
![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-1-image-23.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-1-image-24.png)

## Task 2: Computing the relative score based on videos

Task2 - this directory contains 15 training videos in the constrained scenario showing the ice surface from above. The videos usually end when all stones have stopped from moving on the ice surface. 

### At this point we would like to know what is the potential score based on the configuration of stones on the ice.



We will follow the curling scoring rules adapted to a single video.

For our purpose we will consider that only one team (Red or Yellow) can score during a video. The Team Red plays with the red stones and the Team Yellow plays with the yellow stones. The team with the most stones in The House closest to the curling bullseye (the button) is awarded points. A stone which touches the edge of The House is considered to be inside The House, so it could potentially score a point. So if, after some stones are thrown, Team Red has a stone right on the button, and Team Yellow has a stone a few feet off the button, Team Red scores a point, so the score is 1-0.

If Team Red has one stone on the button and a stone a few feet off the button, while Team Yellow has a stone on the outer edge of the house, Team Red scores two points, so the score is 2-0. If Team Red has three stones outside The House and Team Yellow has two stones inside The House, Team Yellow scores two points, so the score is 0-2.

If no team has stones inside The House the score is 0-0.

The format that you need to follow is the one used in the ground-truth files (located in the subdirectory ground-truth) with the first line containing the total number of points for Team Red and the second line containing the number of points for Team Yellow.

Please pay attention to the scoring rules as you will have to output similar scores in the test phase.

## Proposed Solution

For each video that we have:
  - We get the last frame of the video
  - We apply a conversion from BGR to HSV
  - We generate a blue mask that will be later used for detecting the House circle
  - For smoothing the house circle we apply in this order
      - 5 iterations of dilation with a kernel of 7x7
      - median blur with a kernel of 51x51
      - adaptive thresholding
      - median blur with a kernel of 11x11
  - On the generated image we apply Hough Circles for detecting the House circle
      - If we detect only one circle we use it
      - If we detect multiple circles we use the blue mask for computing the mean intensity of the pixels for each detected circle, selecting the one 'with the most blue pixels'
      - From the selected house circle we reduce the radius with a threshold (on the preprocessed image we used multiple iterations of dilation which can increase the size of the detected house and to match the original image house circle we need to reduce the radius)
      - We use the approach from Task-1 to detect the red and yellow stones
      - We find the stones inside the house (if exists) based on the distance between the house circle center and a detected circle center (should be smaller then the sum of the house radius with the detected circle radius)
      - We sort the distances in ascending order
      - We iterate through distance computing the score until we have stones with different colors
      - Finally the score is computed

## Examples

![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-2-image-12.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-2-image-13.png)
:-------------------------:|:-------------------------:
![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-2-image-15.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-2-image-8.png)

## Task 3: Tracking Stones in a constrained scenario

Task3 - this directory contains 15 training videos in the constrained scenario showing the ice surface from above. The task is to track the stone thrown by a player. You should track the stone from the initial frame to the final frame of the video. The stone is either red or yellow.

The initial bounding box of the stone to be tracked is provided for the first frame (the annotation follows the format [xmin ymin xmax ymax], where (xmin,ymin) is the top left corner and (xmax,ymax) is the bottom right corner of the initial bounding-box).

In each video we will consider that your algorithm correctly tracks the stone if in more (greater or equal) than 80% of the video frames your algorithm correctly localizes the stone to be tracked.

We consider that your algorithmcorrectly localizes the stone to be tracked in a specific frame if the value of the IOU (intersection over union) beetween the window provided by your algorithm and the ground-truth window is more than 20%. The format that you need to follow is the one used in the ground-truth files (located in the subdirectory ground-truth) with the first line containing the number of frames N of the video, and each line having the format [frame index xmin ymin xmax ymax].

The first frame for which we provide the bounding box initialization has frame index 0, the last frame of a video with N frames has frame index N − 1. Please note that the first line of the annotation file has the format [N -1 -1 -1 -1] as it is easy to load the entire matrix ( N + 1 ) × 5 to assess the correctness of your algorithm. Notice that for this task the stone to be tracked appears in each frame of the video.

## Proposed Solution

For each frame in each video that we have:
  - We apply a conversion from BGR to HSV
  - We generate a red, yellow and gray mask
  - We apply the combined mask over the original image
  - We use CSRT tracker to detect the initial bounding box in the masked frames

## Examples

Masked Image (where we track objects)  |  Original Image
:-------------------------:|:-------------------------:
![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-3-mask-1.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-3-image-1.png)
![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-3-mask-9.png)  |  ![](https://github.com/AdrianIordache/Curling-with-Computer-Vision/blob/main/images/task-3-image-9.png)


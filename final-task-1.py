import os
import glob
import math
import cv2 as cv
import numpy as np

def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    total_correct_number_stones_all = 0
    total_correct_number_red_and_yellow = 0
    for i in range(1,26):
        filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")        
        filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")
        
        correct_number_stones_all = 1        
        #read the first line - number of total stones
        p_line = p.readline()
        gt_line = gt.readline()
        if (p_line[:1] != gt_line[:1]):
                correct_number_stones_all = 0                
        
        correct_number_red_and_yellow = 1
        #read the second and third lines - number of red and yellow stones        
        p_line = p.readline()
        gt_line = gt.readline()
        if (p_line[:1] != gt_line[:1]):
                correct_number_red_and_yellow = 0
        p_line = p.readline()
        gt_line = gt.readline()
        if (p_line[:1] != gt_line[:1]):
                correct_number_red_and_yellow = 0                
        p.close()
        gt.close()
        
        if verbose:
            print("Task 1 -Counting stones + their color: for test example number ", str(i), " the prediction is :", (1-correct_number_stones_all) * "in" + "correct for number of total stones and ",(1-correct_number_red_and_yellow) * "in" + "correct for number of red and yellow stones" + "\n")
               
        total_correct_number_stones_all = total_correct_number_stones_all + correct_number_stones_all
        total_correct_number_red_and_yellow = total_correct_number_red_and_yellow + correct_number_red_and_yellow
        points = total_correct_number_stones_all * 0.03 + total_correct_number_red_and_yellow * 0.03
        
    return total_correct_number_stones_all, total_correct_number_red_and_yellow, points

def extract_pixels_values(image, x_center, y_center, radius):
    red_pixels, yellow_pixels = [], []
    is_red, is_yellow = False, False

    threshold = 0
    x_start = x_center - radius
    y_start = y_center - radius
    
    x_end = x_start + 2 * radius
    y_end = y_start + 2 * radius

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # red bounds
    # lower = np.array([161, 155, 84])
    # upper = np.array([179, 255, 255])

    lower = np.array([175, 70, 70])
    upper = np.array([255, 255, 160])
    
    mask = cv.inRange(hsv, lower, upper)
    red  = cv.bitwise_and(image, image, mask = mask)

    # kernel = np.ones((3, 3), np.uint8)
    # red    = cv.erode(red, kernel, iterations = 3)

    # kernel = np.ones((3, 3), np.uint8)
    # red    = cv.dilate(red, kernel, iterations = 8)

    # check detection is red
    for y in range(y_start + threshold, y_end - threshold):
        for x in range(x_start + threshold, x_end - threshold):
            if y >= 720 or x >= 1280: continue
            if np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) < radius - 5:
                red_pixels.append(red[y, x, :].mean())

    #print("Red Means: ", np.mean(red_pixels))
    if np.mean(red_pixels) > 10: is_red = True

    # yellow bounds
    # lower = np.array([22, 93, 0])
    # upper = np.array([45, 255, 255])

    lower = np.array([20, 100, 135])
    upper = np.array([50, 255, 190])

    mask   = cv.inRange(hsv, lower, upper)
    yellow = cv.bitwise_and(image, image, mask = mask)

    # kernel = np.ones((3, 3), np.uint8)
    # yellow = cv.erode(yellow, kernel, iterations = 3)

    # kernel = np.ones((3, 3), np.uint8)
    # yellow = cv.dilate(yellow, kernel, iterations = 10)

    # check detection is yellow
    for y in range(y_start + threshold, y_end - threshold):
        for x in range(x_start + threshold, x_end - threshold):
            if y >= 720 or x >= 1280: continue
            if np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) < radius - 5:
                yellow_pixels.append(yellow[y, x, :].mean())

    #print("Yellow Means: ", np.mean(yellow_pixels))
    if np.mean(yellow_pixels) > 27: is_yellow = True

    is_valid = is_yellow | is_red

    return is_valid, is_red, is_yellow

TASK = 1
PATH_TO_DATA = 'data/'
PATH_TO_TRAIN = 'data/test/test/Task{}/'.format(TASK)

# PATH_TO_GT = 'data/test/test/Task{}/ground-truth/'.format(TASK)
# PATH_TO_PREDICTIONS   = 'data/oof/Task{}/'.format(TASK)

rs = []
for i, path_to_image in enumerate(sorted(glob.glob(PATH_TO_TRAIN + '*.png'))):
    print(path_to_image)
    #path_to_image = 'data/train/Task1/5.png'
    # file_name = path_to_image.split(os.sep)[-1].split(".")[0]
    # path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
    # prediction_file = open(path_to_prediction_file, 'w')

    image = cv.imread(path_to_image)
    output = image.copy()
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)

    # minDist   = 35 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    # param1    = 20 # 500
    # param2    = 20 # 200 #smaller value -> more false circles
    # minRadius = 15
    # maxRadius = 25 # 10

    minDist   = 35 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    param1    = 20 # 500
    param2    = 17 # 200 #smaller value -> more false circles
    minRadius = 15
    maxRadius = 25 # 10

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    # ensure at least some circles were found
    
    reds, yellows = [], []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            is_valid, is_red, is_yellow = extract_pixels_values(output, x, y, r)

            # print(is_red, is_yellow)
            if is_valid:
                #print('R: ', r)
                if is_red: 
                    reds.append(1)
                    image = cv.circle(image, (x,y), r, (0, 0, 255), 3)
                    rs.append(r)

                if is_yellow: 
                    yellows.append(1)
                    image = cv.circle(image, (x,y), r, (0, 255, 255), 3)
                    rs.append(r)

                
    print(reds)
    print(yellows)
    # prediction_file.write(str(sum(reds) + sum(yellows)))
    # prediction_file.write('\n')
    # prediction_file.write(str(sum(reds)))
    # prediction_file.write('\n')
    # prediction_file.write(str(sum(yellows)))

    # prediction_file.close()

    cv.imshow('output', image)
    cv.waitKey(0)

    #break
    if i == 26: break

# print(np.min(rs))
# print(np.max(rs))

# print(*evaluate_results_task1(PATH_TO_PREDICTIONS, PATH_TO_GT, verbose = 1))
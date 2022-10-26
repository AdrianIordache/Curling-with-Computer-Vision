import os
import glob
import math
import cv2 as cv
import numpy as np

def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    total_correct_scores = 0
    for i in range(1,16):
        filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")        
        filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")
        correct_scores = 1
        
        #read the first and second lines - giving the score
        p_line = p.readline()
        gt_line = gt.readline()
        if (p_line[:1] != gt_line[:1]):
                correct_scores = 0
        p_line = p.readline()
        gt_line = gt.readline()
        if (p_line[:1] != gt_line[:1]):
                correct_scores = 0                
        p.close()
        gt.close()
        
        
        if verbose:
            print("Task 2 -Assessing correct score: for test example number ", str(i), " the prediction is :", (1-correct_scores) * "in" + "correct" + "\n")
               
        total_correct_scores = total_correct_scores + + correct_scores        
        points = total_correct_scores * 0.1
        
    return total_correct_scores, points

def preprocessing(image):
    # image  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image  = cv.medianBlur(image, 3)
    # kernel = np.ones((3, 3), np.uint8)
    # image  = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations = 1)


    kernel = np.ones((3, 3), np.uint8)
    image  = cv.dilate(image, kernel, iterations = 1)

    # image  = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations = 1)

    return image

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


def extract_pixels(image, x_center, y_center, radius):
    threshold = 0

    x_start = x_center - radius
    y_start = y_center - radius
    
    x_end = x_start + 2 * radius
    y_end = y_start + 2 * radius

    blue_pixels = []
    for y in range(y_start + threshold, y_end - threshold):
        for x in range(x_start + threshold, x_end - threshold):
            if y >= 720 or x > 1280: continue
            if np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) < radius - 3:
                blue_pixels.append(image[y, x, :].mean())

    return np.mean(blue_pixels)

TASK = 2
PATH_TO_DATA = 'data/'
PATH_TO_TRAIN = 'data/test/test/Task{}/'.format(TASK)

PATH_TO_GT = 'data/train/Task{}/ground-truth/'.format(TASK)
PATH_TO_PREDICTIONS   = 'data/oof/Task{}/'.format(TASK)

rs = []
for i, path_to_video in enumerate(sorted(glob.glob(PATH_TO_TRAIN + '*.mp4'))):
    print(path_to_video)
    #path_to_video = 'data/train/Task2/5.mp4'

    # file_name = path_to_video.split(os.sep)[-1].split(".")[0]
    # path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
    # prediction_file = open(path_to_prediction_file, 'w')

    vs = cv.VideoCapture(path_to_video)

    last_frame_num = vs.get(cv.CAP_PROP_FRAME_COUNT)
    vs.set(cv.CAP_PROP_POS_FRAMES, last_frame_num - 1)

    _, image = vs.read()

    output = image.copy()
    
    prep  = preprocessing(image)
    gray  = cv.cvtColor(prep, cv.COLOR_BGR2GRAY)

    gray_balls = gray.copy()

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # lower_blue = np.array([85,150,50])
    # upper_blue = np.array([95,255,255])

    lower_blue = np.array([70, 45, 60])
    upper_blue = np.array([110, 255, 135])

    mask = cv.inRange(hsv, lower_blue, upper_blue)
    blue = cv.bitwise_and(image, image, mask = mask)

    blue = cv.cvtColor(blue, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)

    black_pixels_mask     = np.all(blue == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(blue != [0, 0, 0], axis=-1)  

    gray[black_pixels_mask] = 0
    gray[non_black_pixels_mask] = 255

    kernel = np.ones((7, 7), np.uint8)
    gray   = cv.dilate(gray, kernel, iterations = 5)
    gray   = cv.medianBlur(gray, 51)

    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 31)
    gray = cv.medianBlur(gray, 11)

    # cv.imshow('blur', gray)
    # cv.waitKey(0)

    # minDist   = 120 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    # param1    = 80 # 500
    # param2    = 80 # 200 #smaller value -> more false circles
    # minRadius = 150
    # maxRadius = 380 # 10

    minDist   = 120 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    param1    = 80 # 500
    param2    = 100 # 200 #smaller value -> more false circles
    minRadius = 150
    maxRadius = 380 # 10

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2.0, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    house = None
    # ensure at least some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # for (x, y, r) in circles:
        #     image = cv.circle(image, (x, y), r - 23, (0, 255, 0), 3)

        if len(circles) > 1:
            means = [extract_pixels(blue, x, y, r) for (x, y, r) in circles]
            max_means = np.max(means)

            for (x, y, r) in circles:
                mean = extract_pixels(blue, x, y, r)
                if mean == max_means:
                    image = cv.circle(image, (x, y), r - 23, (0, 255, 0), 3)
                    house = (x, y, r - 23)

        else:
            for (x, y, r) in circles:
                image = cv.circle(image, (x, y), r - 23, (0, 255, 0), 3)
                house = (x, y, r - 23)
    

    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)

    minDist   = 35 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    param1    = 20 # 500
    param2    = 17 # 200 #smaller value -> more false circles
    minRadius = 15
    maxRadius = 25 # 10

    # minDist   = 22 # if the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    # param1    = 20 # 500
    # param2    = 20 # 200 #smaller value -> more false circles
    # minRadius = 14
    # maxRadius = 25 # 10


    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    stones = []
    # ensure at least some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            is_valid, is_red, is_yellow = extract_pixels_values(output, x, y, r)

            if is_valid:
                #print('R: ', r)
                image = cv.circle(image, (x,y), r, (0, 255, 0), 3)
                if is_red and is_yellow: print("RAU")

                if is_red: 
                    stones.append((x, y, r, 'red'))
                    rs.append(r)
                else:
                    stones.append((x, y, r, 'yellow'))
                    rs.append(r)

    # print(house)
    # print(stones)    

    distances = {}
    for stone in stones:
        distance = np.sqrt((stone[0] - house[0]) ** 2 + (stone[1] - house[1]) ** 2)
        if distance < stone[2] + house[2]:
            #print(distance)
            distances[np.round(distance, 3)] = stone[3]

    score = {
        'red': 0,
        'yellow': 0,
    }

    if len(distances) != 0:
        last = distances[sorted(distances.keys())[0]]
        for key in sorted(distances.keys()):
            if distances[key] == last:
                last = distances[key]
                score[distances[key]] += 1
            else:
                break

    print(score)

    cv.imshow('output', image)
    cv.waitKey(0)

    # prediction_file.write('{}\n{}'.format(score['red'], score['yellow']))
    # prediction_file.close()

    # if i == 0: break

# print(np.min(rs))
# print(np.max(rs))
# print(*evaluate_results_task2(PATH_TO_PREDICTIONS, PATH_TO_GT, 1))
import os
import glob
import math
import cv2 as cv
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames):
    """
    This function compute the percentage of detected bounding boxes based on the ground-truth bboxes and the predicted ones.
    :param gt_bboxes. The ground-truth bboxes with the format: frame_idx, x_min, y_min, x_max, y_max.
    :param predicted_bboxes. The predicted bboxes with the format: frame_idx, x_min, y_min, x_max, y_max
    :param num_frames. The total number of frames in the video.
    """
    
    num_frames = int(num_frames)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    gt_dict = {}
    for gt_box in gt_bboxes:
        gt_dict[gt_box[0]] = gt_box[1:]
    
    pred_dict = {}
    for pred_bbox in predicted_bboxes:
        pred_dict[pred_bbox[0]] = pred_bbox[1:]
        
    for i in range(num_frames):
        if gt_dict.get(i, None) is None and pred_dict.get(i, None) is None: # the stone is not on the ice surface
            tn += 1 
        
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is None: # the stone is not detected
            fn += 1
            
        elif gt_dict.get(i, None) is None and pred_dict.get(i, None) is not None: # the stone is not on the ice surface, but it is 'detected'
            fp += 1
            
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is not None: # the stone is on the ice surface and it is detected
            
            iou = bb_intersection_over_union(gt_dict[i], pred_dict[i])
            if iou >= 0.2:
                tp += 1
            else:
                fp += 1 
                         
    print(f'tp = {tp}, tn = {tn}, fp = {fp},fn = {fn}')
    assert tn + fn + tp + fp == num_frames
    perc = (tp + tn) / (tp + fp + tn + fn)
    
    return perc

def evaluate_results_task3(predictions_path,ground_truth_path, verbose = 0):
    total_correct_tracked_videos = 0
    for i in range(1,16):
        filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"        
        filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
        
        p = np.loadtxt(filename_predictions)
        num_frames = p[0][0]
        predicted_bboxes = p[1:]
        
        gt = np.loadtxt(filename_ground_truth)
        gt_bboxes = gt[1:]
        
        percentage = compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames)
        
        correct_tracked_videos = 1
        if percentage < 0.8:
             correct_tracked_videos = 0
        
        print("percentage = ", percentage)
        if verbose:
            print("Task 3 - Tracking a stone in constrained scenario: for test example number ", str(i), " the prediction is :", (1-correct_tracked_videos) * "in" + "correct", "\n")
        
        total_correct_tracked_videos = total_correct_tracked_videos + correct_tracked_videos
    
    points = total_correct_tracked_videos * 0.1
        
    return total_correct_tracked_videos,points 


# red_lower = np.array([161, 155, 84])
# red_upper = np.array([179, 255, 255])

# yellow_lower = np.array([22, 93, 0])
# yellow_upper = np.array([45, 255, 255])

# lower_gray = np.array([0, 5, 60])
# upper_gray = np.array([179, 50, 100])

if __name__ == '__main__' :

    tracker = cv.TrackerCSRT_create()

    TASK = 3
    PATH_TO_DATA = 'data/'
    PATH_TO_TRAIN = 'data/test/test/Task{}/'.format(TASK)

    PATH_TO_GT = 'data/train/Task{}/ground-truth/'.format(TASK)
    PATH_TO_PREDICTIONS   = 'data/oof/Task{}/'.format(TASK)

    for i, path_to_video in enumerate(sorted(glob.glob(PATH_TO_TRAIN + '*.mp4'))):
        print(path_to_video)
        #path_to_video = 'data/train/Task3/14.mp4'

        # file_name = path_to_video.split(os.sep)[-1].split(".")[0]
        # path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
        # prediction_file = open(path_to_prediction_file, 'w')

        box_file_path = path_to_video.split('/')[-1].split('.')[0]
        box_file = open(os.path.join(PATH_TO_TRAIN, box_file_path + '.txt'), 'r')

        lines = box_file.readlines()
        frames = int(lines[0].split(' ')[0])
        initial_frame = int(lines[1].split(' ')[0])

        xmin = int(lines[1].split(' ')[1])
        ymin = int(lines[1].split(' ')[2])
        xmax = int(lines[1].split(' ')[3])
        ymax = int(lines[1].split(' ')[4])

        first_line = "{} -1 -1 -1 -1\n".format(frames)
        # prediction_file.write(first_line)

        vs = cv.VideoCapture(path_to_video)
        vs.set(cv.CAP_PROP_POS_FRAMES, initial_frame)
        _, frame = vs.read()
        copy = frame.copy()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # red_lower = np.array([175, 155, 84])
        # red_upper = np.array([255, 200, 200])

        # yellow_lower = np.array([22, 93, 0])
        # yellow_upper = np.array([45, 255, 255])

        # lower_gray = np.array([0, 20, 60])
        # upper_gray = np.array([255, 90, 90])

        red_lower = np.array([175, 70, 70])
        red_upper = np.array([255, 255, 160])

        yellow_lower = np.array([20, 100, 135])
        yellow_upper = np.array([50, 255, 190])

        lower_gray = np.array([0, 5, 60])
        upper_gray = np.array([165, 130, 140])

        red_mask    = cv.inRange(hsv, red_lower, red_upper)
        yellow_mask = cv.inRange(hsv, yellow_lower, yellow_upper)
        gray_mask   = cv.inRange(hsv, lower_gray, upper_gray)

        mask = yellow_mask + red_mask + gray_mask

        frame  = cv.bitwise_and(frame, frame, mask = mask)

        # kernel = np.ones((3, 3), np.uint8)
        # frame = cv.erode(frame, kernel, iterations = 1)

        # kernel = np.ones((3, 3), np.uint8)
        # frame = cv.dilate(frame, kernel, iterations = 7)

        # frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv.imshow('image', frame)
        # cv.waitKey(0)
        #break

        threshold = 0
        box = (xmin + threshold, ymin + threshold, xmax - xmin - threshold, ymax - ymin - threshold - 0)

        ok = tracker.init(frame, box)
        
        counter = 0
        line = '{} {} {} {} {}\n'.format(counter, xmin, ymin, xmax, ymax)
        # prediction_file.write(line)
        counter += 1

        while True:
            ok, frame = vs.read()
            
            if not ok:
                break
            
            copy = frame.copy()
            hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            red_mask    = cv.inRange(hsv, red_lower, red_upper)
            yellow_mask = cv.inRange(hsv, yellow_lower, yellow_upper)
            gray_mask   = cv.inRange(hsv, lower_gray, upper_gray)

            mask = yellow_mask + red_mask + gray_mask

            frame  = cv.bitwise_and(frame, frame, mask = mask)

            # kernel = np.ones((3, 3), np.uint8)
            # frame = cv.erode(frame, kernel, iterations = 1)

            # kernel = np.ones((3, 3), np.uint8)
            # frame = cv.dilate(frame, kernel, iterations = 7)

            # Update tracker
            ok, bbox = tracker.update(frame)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(copy, p1, p2, (0, 255,0), 2, 1)

            if counter < frames:
                line = '{} {} {} {} {}\n'.format(counter, p1[0], p1[1], p2[0], p2[1])
            else:
                line = '{} {} {} {} {}'.format(counter, p1[0], p1[1], p2[0], p2[1])

            # prediction_file.write(line)
            counter += 1

            cv.imshow("Tracking", copy)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27 : break

        #break
    #prediction_file.close()
    
    #print(*evaluate_results_task3(PATH_TO_PREDICTIONS, PATH_TO_GT, 1))




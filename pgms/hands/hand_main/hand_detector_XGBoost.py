import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from pgms.hands.hand_main.filename import filen
from pgms.hands.hand_main.model import KeyPointClassifier
import joblib
import joblib  
from pgms.hands.hand_main.hand_functions import get_args, select_mode, calc_bounding_rect, calc_landmark_list, pre_process_landmark, logging_csv, draw_bounding_rect, draw_landmarks, draw_info_text

GREEN = (0, 255, 0)

from xgboost import XGBClassifier  # Updated for XGBoost model

# Argument parsing
args = get_args()

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True


# Model load
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

# Load XGBoost model
xgboost_model = joblib.load(r'E:\project demo\pgms\hands\hand_main\model\keypoint_classifier\xgboost_model.pkl')  # Replace with the path to your XGBoost model file

# Read labels
with open('E:\pgms\hands\hand-gesture-recognition-mediapipe-main\model\keypoint_classifier\keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

def hand(vid, loading_bar_hand):

    # Camera preparation
    cap = cv.VideoCapture(vid)
    loading_bar_hand.progress(10)

    mode = 0
    opencount = 0
    closecount = 0
    pointcount = 0
    count = 0

    output_frames = []

    output_file = filen(vid)

    #print(output_file)

    # Create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'avc1')
    fps = 30.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    loading_bar_hand.progress(20)

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(image)

        height, width = debug_image.shape[:2]

        # Determine the scaling factor to make the longest edge 600 pixels
        scaling_factor = 800 / max(height, width)

        # Calculate the new dimensions
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Resize the frame
        debug_image = cv.resize(debug_image, (new_width, new_height))

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                count =  count + 1
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification using XGBoost
                hand_sign_id = xgboost_model.predict([pre_processed_landmark_list])[0]

                if hand_sign_id == 0:
                    opencount = opencount+1
                    color = GREEN
                elif hand_sign_id == 1:
                    cosecount = closecount + 1
                    color = (0, 0, 255)
                elif hand_sign_id == 2:
                    pointcount = pointcount + 1
                    color = (0, 0, 255)

                cv.putText(debug_image, str(hand_sign_id), (100, 250), cv.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect, color)
                #debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    "",
                )

        cv.imshow('Hand Gesture Recognition', debug_image)
        output_frames.append(debug_image)
        out.write(debug_image)

        if cv.waitKey(24) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

    loading_bar_hand.progress(80)

    try:

        open_score = (opencount/count)*100
        close_score = (closecount/count)*100
        point_score = (pointcount/count)*100

        print(keypoint_classifier_labels[0], open_score, keypoint_classifier_labels[1], close_score, keypoint_classifier_labels[2], point_score)

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        message = ''

        if open_score>=70:
            messagep = messagep + " Good job on using open hand gestures most of the time. Open hand gestures convey openness and enthusiasm."
        else:
            messagen = messagen + " Practice using open hand gestures to convey openness and enthusiasm."

        if close_score >=10:
            messagen = messagen + " Refrain from using closed hand gestures. Closed hands can be interpreted as nervousness and defensiveness."

        if point_score >=10:
            messagen = messagen + " Don't point your fingers too much. Pointing fingers can be seen as aggressive."
        
        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''
        
        message = messagep + '\n\n' + messagen
    
    except:
        open_score = 0
        message = 'No hand gestures detected.'
    
    loading_bar_hand.progress(90)
    
    return output_frames, message, open_score
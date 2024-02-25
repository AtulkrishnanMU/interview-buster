import csv
import copy
import os
import cv2
import mediapipe as mp

def encode_label(label_name, category):
    for i in category:
        if i == label_name:
            return category.index(i)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark[0:25]:  # Only take the upper body landmarks
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = [item for sublist in temp_landmark_list for item in sublist]

    max_value = max(map(abs, temp_landmark_list))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, landmark_list):
    if 0 <= number <= 5:
        csv_path = r'E:\project demo\pgms\body\model\keypoint_classifier\keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

root = "E:/website files/pose/"
IMAGE_FILES = []
category = ['arms crossed', 'arms raised', 'explain', 'straight', 'touch face']

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

for path, subdirs, files in os.walk(root):
    for name in files:
        IMAGE_FILES.append(os.path.join(path, name))

for idx, file in enumerate(IMAGE_FILES):
    print(file)
    label_name = file.rsplit("/", 1)[-1]

    label_name = label_name.rsplit("\\", 1)[0]
    print(label_name)
    label = encode_label(label_name, category)

    try:
        image = cv2.imread(file)
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            #print(results.pose_landmarks)
            landmark_list = calc_landmark_list(debug_image, results.pose_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            logging_csv(label, pre_processed_landmark_list)
    except:
        print('FAILED!!')

import cv2
import numpy as np
import math
from pgms.head_eye.colors import GREEN, WHITE

import numpy as np

def compute_ipd(left_iris_landmark, right_iris_landmark):
  """Calculates the interpupillary distance (IPD) between two 3D landmarks.

  Args:
    left_iris_landmark: A `np.ndarray` of shape (3,) containing the 3D coordinates of the left iris landmark.
    right_iris_landmark: A `np.ndarray` of shape (3,) containing the 3D coordinates of the right iris landmark.

  Returns:
    A `float` representing the IPD in millimeters.
  """

  # Calculate the distance between the two landmarks.
  distance = np.linalg.norm(left_iris_landmark - right_iris_landmark)

  # Convert the distance to millimeters.
  ipd_in_mm = distance * 1000

  return ipd_in_mm

def are_points_collinear(point1, point2, point3, tolerance):
    # Extract x and y coordinates of the points
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    # Calculate slopes between pairs of points
    slope12 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    slope13 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else float('inf')
    slope23 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else float('inf')
    
    # Check if the slopes are equal or almost equal within the given tolerance
    if abs(slope12 - slope13) <= tolerance and abs(slope12 - slope23) <= tolerance:
        return True
    else:
        return False

def calculate_percentage(binary_list):
    if not binary_list:
        return 0.0  # Handle the case where the list is empty
    
    count_ones = sum(1 for bit in binary_list if bit == 1)
    total_bits = len(binary_list)
    
    percentage_ones = (count_ones / total_bits) * 100.0
    return percentage_ones

def find_leftmost_rightmost(coordinates):
    leftmost = (float('inf'), float('inf'))
    rightmost = (-float('inf'), -float('inf'))
    
    for x, y in coordinates:
        leftmost = (min(leftmost[0], x), min(leftmost[1], y))
        rightmost = (max(rightmost[0], x), max(rightmost[1], y))
    
    return leftmost, rightmost

def transform_coordinates(coordinates):
    leftmost, rightmost = find_leftmost_rightmost(coordinates)
    
    # Calculate the scaling factor
    scaling_factor = 100 / (rightmost[0] - leftmost[0])
    
    transformed_coordinates = []
    for x, y in coordinates:
        # Translate to (50, 50)
        translated_x = x - leftmost[0] + 50
        translated_y = y - leftmost[1] + 50
        
        # Scale the coordinates
        scaled_x = translated_x * scaling_factor
        scaled_y = translated_y * scaling_factor
        
        transformed_coordinates.append([int(scaled_x), int(scaled_y)])
    
    return [np.array(transformed_coordinates)]

def landmarkdet(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    return mesh_coord

#euclidean dist
def eucli(p1, p2):
    x, y = p1
    x1, y1 = p2
    dist = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return dist

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = eucli(iris_center, right_point.ravel())
    tot_dist = eucli(left_point.ravel(), right_point.ravel())
    ratio = center_to_right_dist/tot_dist
    return ratio

def iris_position2(iris_center, landmarks, indices):
    top1 = landmarks[indices[12]]
    top2 = landmarks[indices[11]]
    top = (int((top1[0] + top2[0])/2), int((top1[1] + top2[1])/2))
    bottom = landmarks[indices[4]]
    center_to_top_dist = eucli(iris_center, top)
    tot_dist = eucli(top, bottom.ravel())
    try:
        ratio = center_to_top_dist/tot_dist
    except:
        ratio = -1
    return ratio

def blinkratio(img, landmarks, right_indices, left_indices):
    #right eyes horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #right eyes vertical line
    rv_top1 = landmarks[right_indices[12]]
    rv_top2 = landmarks[right_indices[11]]
    rv_top = (int((rv_top1[0] + rv_top2[0])/2), int((rv_top1[1] + rv_top2[1])/2))
    rv_bottom = landmarks[right_indices[4]]

def head_pose_estimate(model_points, landmarks, K):
    #h, w = image_size
    '''
    K = np.float64(
                [[w,   0,      0.5*(w-1)],
                [0,         h, 0.5*(h-1)],
                [0.0,       0.0,    1.0]])
    '''
    dist_coef = np.zeros((4, 1))
    ret, rvec, tvec = cv2.solvePnP(model_points, landmarks, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)

    rot_mat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
    yaw   = int(eulerAngles[1, 0]*360)
    pitch = int(eulerAngles[0, 0]*360)
    roll  = eulerAngles[2, 0]*360
    return roll, yaw, pitch

def newirispos2(transformed_eye_coordinates, image):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p1 = flat_cords[0]
    p4 = flat_cords[8]
    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    iris = flat_cords[17]
    #right = flat_cords[0]
    #left = flat_cords[8]

    #earclosed = ((10) / (eucli(right, left)))
    #earopen = ((100) / (eucli(right, left)))

    p = (p1+p4)/2

    #p = (((p2+p6)/2)+((p3+p5)/2))/2 #center of left eye
    #print(p)
    #print(iris)
    con = p-iris
    con = (abs(con[0]), abs(con[1]))
    #print(p, iris)
    #print(p-iris) #center of left eye - center of left iris

    #print("eucli: ", eucli(p,iris))

    # 8 - 9.5, 1-2.5

    # Convert points to integers
    point1_int = (int(p[0]), int(p[1]))
    point2_int = (int(iris[0]), int(iris[1]))

    # Draw circles for the points on the image
    cv2.circle(image, point1_int, 5, (0, 0, 255), -1)  # Red color for point1
    cv2.circle(image, point2_int, 5, (0, 255, 0), -1)  # Green color for point2

    return con

#blink ratio
def newbratio(transformed_eye_coordinates):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2*(eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2*(eucli(right, left)))

    ear = (eucli(p2,p6) + eucli(p3,p5))/2*(eucli(right, left))

    thresh = (earopen + earclosed)/2

    if ear<=thresh:
        return True
    else:
        return False

def yratio(transformed_eye_coordinates, iris_center):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2*(eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2*(eucli(right, left)))

    ear = (eucli(p2,p6) + eucli(p3,p5))/2*(eucli(right, left))

    thresh = (earopen + earclosed)/2

    if ear<=thresh:
        return True
    else:
        return False

def smiledet(point1, point2, point33):
    # Calculate vectors from point1 to point2 and from point1 to point33
    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
    vector2 = (point33[0] - point1[0], point33[1] - point1[1])

    # Calculate the cross product of vector1 and vector2
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # If the cross product is positive, point2 is below the line
    return cross_product

def smileratio(img, landmarks, LOWER_LIPS, CHIN):
    # Calculate lips width
    lips_width = abs(eucli(landmarks[LOWER_LIPS[0]], landmarks[LOWER_LIPS[16]]))

    # Calculate jaw width
    jaw_width = abs(eucli(landmarks[CHIN[0]], landmarks[CHIN[1]]))

    # Calculate the ratio of lips and jaw widths
    ratio = lips_width/jaw_width

    return ratio

def blinkratio(img, landmarks, right_indices, left_indices):
    #right eyes horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #right eyes vertical line
    rv_top1 = landmarks[right_indices[12]]
    rv_top2 = landmarks[right_indices[11]]
    rv_top = (int((rv_top1[0] + rv_top2[0])/2), int((rv_top1[1] + rv_top2[1])/2))
    rv_bottom = landmarks[right_indices[4]]
    #left eyes horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    #left eyes vertical line
    lv_top1 = landmarks[left_indices[12]]
    lv_top2 = landmarks[left_indices[13]]
    lv_top = (int((lv_top1[0] + lv_top2[0])/2), int((lv_top1[1] + lv_top2[1])/2))
    lv_bottom = landmarks[left_indices[4]]

    rhdist = eucli(rh_right, rh_left)
    rvdist = eucli(rv_top, rv_bottom)

    lhdist = eucli(lh_right, lh_left)
    lvdist = eucli(lv_top, lv_bottom)

    cv2.line(img, rh_right, rh_left, GREEN,2)
    cv2.line(img, rv_top, rv_bottom, WHITE,2)
    cv2.line(img, lh_right, lh_left, GREEN,2)
    cv2.line(img, lv_top, lv_bottom, WHITE,2)

    try:

        rratio = rhdist/rvdist
        lratio = lhdist/lvdist

        ratio = (rratio + lratio)/2
    except:
        ratio = -1

    return ratio

def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the slopes of the lines
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Calculate the y-intercepts of the lines
    b1 = y1 - m1 * x1 if m1 != float('inf') else None
    b2 = y3 - m2 * x3 if m2 != float('inf') else None

    # Check for parallel lines (no intersection)
    if m1 == m2:
        return False

    # Calculate the intersection point
    if m1 != float('inf') and m2 != float('inf'):
        x_intersect = (b2 - b1) / (m1 - m2)
    elif m1 == float('inf'):
        x_intersect = x1
    else:
        x_intersect = x3

    # Check if the intersection point is within the line segments
    if (
        min(x1, x2) <= x_intersect <= max(x1, x2) and
        min(x3, x4) <= x_intersect <= max(x3, x4)
    ):
        return True
    else:
        return False
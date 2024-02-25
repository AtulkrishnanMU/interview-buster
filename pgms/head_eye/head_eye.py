import cv2
import mediapipe as mp
import numpy as np
from pgms.head_eye.function_mod_old import head_pose_estimate, landmarkdet, transform_coordinates, iris_position, newirispos2, blinkratio, compute_ipd, newbratio, iris_position2, eucli, calculate_percentage, are_points_collinear, smileratio
from pgms.head_eye.landmarks import FACE_OVAL, LIPS, LOWER_LIPS, UPPER_LIPS, LEFT_EYE, LEFT_EYEBROW, RIGHT_EYE, RIGHT_EYEBROW, L_iris_center, R_iris_center, RIGHT_IRIS, LEFT_IRIS, LH_LEFT, LH_RIGHT, RH_LEFT, RH_RIGHT, THAADI, CHIN
from pgms.head_eye.colors import WHITE, GREEN, ORANGE, BLACK
import tensorflow 
from pgms.head_eye.filename import filen
import streamlit as st

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    info_text =''
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def head_eye(vid, loading_bar_smile):
    count = 0

    text=''

    eyecount = 0
    headcount = 0

    straight = 0

    blinkcount = 0
    blinklist = []

    fps=0

    prev = 0
    consecutive_blink = 0
    blink_too_long =0

    output_frames = []

    map_face_mesh = mp.solutions.face_mesh

    rect_color = (0, 255, 0)  # Green

    cap = cv2.VideoCapture(vid)

    loading_bar_smile.progress(10)

    output_file = r'E:\project demo\media\eye-contact.mp4'

    #print(output_file)

    # Create a VideoCapture object
    #cap = cv2.VideoCapture(r'E:\website files\bodylang\myapp\pgms\WhatsApp Video 2023-08-18 at 20.00.43.mp4')

    # Check if the camera or video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Change to 'avc1' for H.264 codec
    fps = 30.0  # Frames per second (you can adjust this)

    # Define the output video dimensions (use the same as the input if not resizing)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    loading_bar_smile.progress(30)

    progress = 30

    with map_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Find the dimensions of the frame
            height, width, _ = frame.shape

            # Determine the scaling factor to make the longest edge 600 pixels
            scaling_factor = 800 / max(height, width)

            # Calculate the new dimensions
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)

            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))

            fps = fps+1

            face_3d = []
            face_2d = []

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    count = count + 1

                    if progress<=80:
                        progress + 0.01
                        loading_bar_smile.progress((int(progress)))
                    
                    if fps%1441 == 0:
                        blinklist.append(blinkcount)
                        blinkcount = 0
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1  or idx == 61 or idx ==291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * width, lm.y * height)
                                nose_3d = (lm.x * width, lm.y * height, lm.z * 3000)
                            
                            x,y = int(lm.x * width), int(lm.y * height)

                            face_2d.append([x,y])
                            face_3d.append([x,y,lm.z])
                    
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * width

                    cam_matrix = np.array([
                        [focal_length, 0, height/2],
                        [0,focal_length, width/2],
                        [0,0,1]  
                    ])

                    #distortion parameters 
                    dist_matrix = np.zeros((4,1), dtype=np.float64)

                    #solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    #X,Y,Z = head_pose_estimate(face_3d, face_2d, cam_matrix)
                    #print(X,Y,Z)

                    rot_matrix, jac = cv2.Rodrigues(rot_vec)

                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)

                    x = angles[0] * 360 #pitch
                    y = angles[1] * 360 #yaw
                    #z = angles[2] * 360

                    #print(x,y,z)

                    #x_deg = angles[0]
                    #y_deg = angles[1]
                    #z_deg = angles[2] 

                    #cv2.putText(frame, str(z), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                    #print(x,' ', y)

                    #cv2.putText(frame, str(int(y)) + ", " + str(int(x)), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0])-100, int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y*10)-100, int(nose_2d[1] - x*10))

                    if not (-5<int(y)<5 and -5<int(x)<5): 
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        #print(count, ": Head not straight")
                    else:
                        #cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        #cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        straight = 1
                        #print(count, ": Head straight")
                
                mesh_coords = landmarkdet(frame, results, False)

                mesh_points = np.array(mesh_coords)

                #cv2.line(frame, p1, p2, (255, 0, 0), 3)
                fhead = tuple(mesh_points[151])
                chin = tuple(mesh_points[175])

                #cv2.line(frame, fhead, chin, (0, 255, 0), 2)

                threshold = 10 

                print(abs(fhead[0] - chin[0]))

                # Check if the slope is almost straight
                if straight == 1:
                    if abs(fhead[0] - chin[0]) < threshold:
                        straight = 1
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        headcount = headcount+1
                    else:
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                (l_cx, l_cy), l_radius =  cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius =  cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                #detecting smile

                forehead = tuple(mesh_points[FACE_OVAL][0])
                #print(forehead)

                lip_point1 = tuple(mesh_points[LOWER_LIPS][0])
                lip_point2 = tuple(mesh_points[LOWER_LIPS][16])
                lip_point3 = tuple(mesh_points[UPPER_LIPS][13])
                lip_point4 = tuple(mesh_points[LOWER_LIPS][10])

                #cv2.circle(frame, mesh_points[CHIN][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[CHIN][1], 2, (255,0,255), 1, cv2.LINE_AA)

                #cv2.circle(frame, mesh_points[L_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[R_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)

                #cv2.circle(frame, mesh_points[LOWER_LIPS][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[LOWER_LIPS][10], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[LOWER_LIPS][16], 2, (255,0,255), 1, cv2.LINE_AA)

                x1,y1 = mesh_points[LOWER_LIPS][0]
                x2,y2 = mesh_points[LOWER_LIPS][10]
                x3,y3 = mesh_points[LIPS][25]
                x4,y4 = mesh_points[THAADI][0]

                x5,y5 = mesh_points[CHIN][0]
                x6,y6 = mesh_points[CHIN][1]
                
                eye_coordinates = []
                eye_cont_coordinates = []
                r_eye_cont_coordinates = []

                for i in LEFT_EYE:
                    eye_coordinates.append(tuple(mesh_points[i]))

                LEFT_EYE_and_IRIS =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398, 468, 473]
                RIGHT_EYE_and_IRIS  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246, 473, 468]

                for i in RIGHT_EYE_and_IRIS:
                    r_eye_cont_coordinates.append(tuple(mesh_points[i]))
                
                for i in LEFT_EYE_and_IRIS:
                    eye_cont_coordinates.append(tuple(mesh_points[i]))
                
                # Transform the coordinates
                transformed_eye_coordinates = transform_coordinates(eye_coordinates)
                transformed_eyecont_coordinates = transform_coordinates(eye_cont_coordinates)
                rtransformed_eyecont_coordinates = transform_coordinates(r_eye_cont_coordinates)

                #cv2.polylines(frame, transformed_eye_coordinates, True, GREEN)

                blink = newbratio(transformed_eye_coordinates)
                if blink:
                    if prev == 1:
                        if consecutive_blink<=72:
                            consecutive_blink = consecutive_blink+1
                        else:
                            blink_too_long =1
                    prev = 1
                else:
                    if prev == 1:
                        blinkcount = blinkcount + 1
                    prev = 0
                    consecutive_blink = 0
                    
                #cont =  newirispos2(transformed_eyecont_coordinates)
                cont =  newirispos2(transformed_eyecont_coordinates, frame)
                rcont = newirispos2(rtransformed_eyecont_coordinates, frame)

                print((cont[0]+rcont[0])/2, (cont[1]+rcont[1])/2)

                if 0<=((cont[0]+rcont[0])/2)<=2.5 and 0<=((cont[1]+rcont[1])/2)<=3.5:
                    contact = True
                else:
                    contact = False

                #if blink ==True:
                    #cv2.rectangle(frame, (25, 40), (200, 66), BLACK, -1)
                    #cv2.putText(frame, 'Blink', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                l1x, l1y = lip_point1
                l2x, l2y = lip_point2

                vertical_distance = l2y - l1y

                rv_bottom = mesh_coords[RIGHT_EYE[4]] #right eye vertical bottom point

                #print(eucli(tuple(center_right),rv_bottom))
                #cv2.rectangle(frame, (25, 60), (200, 90), BLACK, -1)

                #check if traingle is formed in the form \/

                # Calculate lips width
                
                col = are_points_collinear(lip_point1, lip_point3, lip_point4, 0.8)
                #col2 = are_points_collinear(lip_point1, lip_point2, lip_point4, 0.8)


                ratio1 = iris_position(center_right, mesh_points[RH_RIGHT], mesh_points[RH_LEFT])
                ratio2 = iris_position(center_left, mesh_points[LH_RIGHT], mesh_points[LH_LEFT])
                ratio = (ratio1 + ratio2)/2

                topratio2 = iris_position2(center_right, mesh_points, RIGHT_EYE)
                topratio1 = iris_position2(center_left, mesh_points, LEFT_EYE)
                topratio = (topratio1 + topratio2)/2

                #print(int((1/topratio)*100))
                
                #cv2.putText(frame, str(int(ratio*100)), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                '''
                    if cont:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        #print(count, ": Eye contact")
                '''
                
                #if straight == 1:
                if not blink:
                    if contact:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        #print(count, ": Eye contact")
                        eyecount = eyecount + 1
                        '''
                    if 45<(int(ratio*100))<55 and 11<=int((1/topratio)*100)<=17:
                            #cv2.rectangle(frame, (25, 40), (200, 66), BLACK, -1)
                            text = 'Eye Contact'
                            rect_color = (0, 255, 0)  # Green
                            print(count, ": Eye contact")
                            #cv2.putText(frame, str(eucli(tuple(center_right),rv_bottom)), (200,100), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                    '''
                    else:
                        text = 'Not Eye contact'
                        rect_color = (0, 0, 255)  # Red
                        #print(count, ": Not Eye contact")
                else:
                    text = 'Blink'
                    rect_color = (0, 0, 255)  # Red
                    #print(count, ": Not Eye contact")
                #else:
                    #text = "Look straight"
                    #rect_color = (0, 0, 255)
                
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate bounding rectangle
                    brect = calc_bounding_rect(frame, face_landmarks)

                    # Draw bounding rectangle around the head
                    frame = draw_bounding_rect(True, frame, brect, rect_color)

                    frame = draw_info_text(
                    frame,
                    brect,
                    text)
            
            cv2.imshow('Frame', frame)
            output_frames.append(frame)
            #st.image(frame, channels="BGR", caption="Processed Frame")

            if cv2.waitKey(24) & 0xFF == ord('q'):
                break

        # Release the VideoWriter object.

        #cv2.imshow('Image', frame)
        cv2.destroyAllWindows()
    
    try:

        head_score = ((headcount/count)*100)
        eye_score = ((eyecount/count)*100)

        loading_bar_smile.progress(70)

        print(head_score)

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        if head_score<=50:
            messagen += "Your head was not straight most of the time. Keep it straight."
        elif 50<head_score<=90:
            messagen += "Consider maintaining a more consistent straight head posture."
        elif 90<head_score:
            messagep += "Great job maintaining your head straight! It showcases your focus and attentiveness."

        if blink_too_long == 1:
            messagen =  messagen + " Don't close your eyes for too long."

        if eye_score<=25:
            messagen = messagen + " It seems like you are looking away occasionally. Consider practicing maintaining eye contact."
        elif 25<eye_score<=50:
            messagen = messagen + " Limited eye contact detected. Consider practicing maintaining eye contact for longer stretches to increase your confidence and connect with your audience."
        elif 50<eye_score<=75:
            messagen = messagen + " Your eye contact is not bad, but try holding it longer."
        elif 75<eye_score<=90:
            messagep = messagep + " Good Job in maintaining eye contact for most of the time."
        elif 90<eye_score:
            messagep = messagep + " Impressive! Your eye contact is very strong!"
        
        try:
            total_blink = sum(blinklist)/len(blinklist)
            print(total_blink)
        except:
            print('')

        print("count = ", fps)

        try:
            if total_blink>20:
                messagen = messagen + " You are blinking too much. On average, most people blink around 15 to 20 times each minute. You were blinking on an average of " + str(int(total_blink)) + ". Too much blinking indicate lack of concentration."
        except:
            print('')

        loading_bar_smile.progress(90)

        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''

        message = messagep + "\n\n" + messagen

    except:
        head_score, eye_score = 0, 0
        message = 'No face detected.'

    return output_frames, message, head_score, eye_score
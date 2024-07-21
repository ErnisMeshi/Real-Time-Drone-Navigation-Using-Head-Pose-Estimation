import cvzone
import keyboard
from cvzone.FPS import FPS
from cvzone.PIDModule import PID
from cvzone import stackImages as stack_images
from face_detection import RetinaFace
from sixdrepnet import SixDRepNet

from PlotModule import SignalPlot
from camera_utils import SimpleRecoroder
from eye_blinking_utils import DroneInitializationLogic
from face_det_utils import FaceDetector, HeadPoseDetector
from djitellopy import Tello

import cv2
import numpy as np

from visualization import draw_axis

model = SixDRepNet()

# Open webcam
webcam = cv2.VideoCapture(1)
face_bbox_detector = RetinaFace(gpu_id=0)

running = True
drone_state = 'off'
take_off_logic = DroneInitializationLogic()

reference_angle = 0
roll_pid = PID([0.3, 0, 0.1], reference_angle)
yaw_pid = PID([0.3, 0, 0.1], reference_angle)
pitch_pid = PID([0.3, 0, 0.1], reference_angle)
keepRecording = True
drone = Tello()
drone.connect()
drone.streamon()
drone_cam = drone.get_frame_read()

fpsReader = FPS(avgCount=30)
take_off_command = False
first_measurement = None
all_windows = SimpleRecoroder('all_windows', 1280, 480)
velocity_plots = SignalPlot(yLimit=[-60, 60])

velocity_graph = velocity_plots.update(0, 0, 0, 0)

num = 0


class Region:
    def __init__(self, webcam):
        _, frame = webcam.read()

        self.frame = frame
        frame_height, frame_width, _ = frame.shape
        self.region_width = frame_width // 4
        self.region_height = frame_height // 3
        frame_center_y, frame_center_x = frame_height // 2, frame_width // 2

        self.region_xmin = frame_center_x - int(self.region_width / 2)
        self.region_ymin = frame_center_y - int(self.region_height / 2)

    def is_point_in_region(self, point):
        return self.region_xmin <= point[0] <= self.region_xmin + self.region_width and self.region_ymin <= point[
            1] <= self.region_ymin + self.region_height

    def draw(self, frame, bbox, color, text, bg_color=(0, 0, 0)):
        text_x = bbox[0] - 30
        text_y = bbox[1] - 50
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Draw background rectangle for text
        cv2.rectangle(frame, (text_x, text_y - text_height - baseline),
                      (text_x + text_width, text_y + baseline), bg_color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # cv2.putText(self.frame,text,(self.center_region-15,self.region_ymin-15),cv2.FONT_HERSHEY_PLAIN,1,color,1)
        cvzone.cornerRect(frame, bbox,
                          colorC=color, rt=0, t=5)


center_region = Region(webcam)
face_detector = FaceDetector()
head_pose = HeadPoseDetector()
region_color = (0, 0, 0)
text = ''
while running:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    if not drone.is_flying:
        drone_frame = np.zeros((384, 640, 3), np.uint8)
        face_landmarks = take_off_logic.detect_face_mesh(frame)
        if face_landmarks:
            x_coords = [landmark[0] for landmark in face_landmarks]
            y_coords = [landmark[1] for landmark in face_landmarks]

            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)

            bbox = (int(min_x), int(min_y), int((max_x - min_x)), int((max_y - min_y)))

            nose_tip = face_landmarks[1]
            left_eye = [face_landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

            cv2.drawContours(frame, [np.array(left_eye)], -1, (255, 255, 0), 1)
            face_inside = center_region.is_point_in_region(nose_tip)
            if face_inside:
                text = 'Blink left eye to start'
                take_off_command = take_off_logic.check_for_eye_blink(face_landmarks)

                region_color = (0, 200, 0)
            else:

                text = 'Align head with the region'
                region_color = (0, 0, 200)
                take_off_command = False
            center_region.draw(frame, bbox, region_color, text)
        else:
            cv2.putText(frame, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    if take_off_command and not drone.is_flying:
        print(drone.get_battery())
        if int(drone.get_battery()) < 20:
            raise ValueError(f'Low battery {drone.get_battery}')

        cv2.putText(frame, 'Drone take off', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        four_frames = stack_images([frame, velocity_graph], 2, 1)

        faces = face_bbox_detector(frame)

        drone.takeoff_custom(webcam, all_windows)

    if drone.is_flying:
        drone_frame = drone_cam.frame
        drone_frame = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
        drone_frame = cv2.resize(drone_frame, (640, 384))
        bbox, landmarks = face_detector.predict(frame)
        if not len(bbox) and not len(landmarks):
            continue
        cropped_face = face_detector.crop_face(frame, bbox)

        head_pose.update_angles(cropped_face)

        head_pose.update_depth(landmarks)
        action = head_pose.get_action()
        head_pose.draw(frame, bbox, action, bg_color=(0, 0, 0))

        if action == 'flip_left':
            drone.flip_left_custom(webcam, all_windows)
        elif action == 'flip_right':
            drone.flip_right_custom(webcam, all_windows)
        elif action == 'flip_forward':
            drone.flip_forward_custom(webcam, all_windows)

        elif action == 'land' or keyboard.is_pressed('q'):
            running = False
            drone.land_custom(webcam, all_windows)

        else:
            drone.send_rc_control(*head_pose.speed)
            head_pose.draw_direction_arrow(drone_frame, action)
        draw_axis(frame, head_pose.yaw, head_pose.pitch, head_pose.roll, 100, 50, size=75)

    four_frames = stack_images([frame, drone_frame], 2, 1)

    cv2.imshow("window", four_frames)
    all_windows.write(four_frames)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
webcam.release()

# head_pose_video.release()
cv2.destroyAllWindows()

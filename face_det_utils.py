import math

import cv2
import cvzone
import numpy as np
from cvzone.PlotModule import LivePlot
from face_detection import RetinaFace
from sixdrepnet import SixDRepNet

def draw_dotted_arrowed_line(img, start_point, end_point, color, thickness, segment_length=3, gap_length=3,
                             tip_length=0.3):
    # Calculate the distance between start and end points
    distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)

    # Calculate the number of segments excluding the arrow tip
    num_segments = int((distance * (1 - tip_length)) // (segment_length + gap_length))

    # Calculate the direction vector
    direction = ((end_point[0] - start_point[0]) / distance, (end_point[1] - start_point[1]) / distance)

    # Draw each segment
    for i in range(num_segments):
        segment_start = (int(start_point[0] + i * (segment_length + gap_length) * direction[0]),
                         int(start_point[1] + i * (segment_length + gap_length) * direction[1]))
        segment_end = (int(segment_start[0] + segment_length * direction[0]),
                       int(segment_start[1] + segment_length * direction[1]))
        cv2.line(img, segment_start, segment_end, color, thickness)

    # Draw the arrow tip
    arrow_tip_start = (int(start_point[0] + num_segments * (segment_length + gap_length) * direction[0]),
                       int(start_point[1] + num_segments * (segment_length + gap_length) * direction[1]))
    cv2.arrowedLine(img, arrow_tip_start, end_point, color, thickness, tipLength=tip_length)
class HeadPoseDetector:
    def __init__(self):
        self.speed = [0, 0, 0, 0]

        self.model = SixDRepNet()
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        self.delta_depth = 0
        self.first_measurement = None

        self.action_colors = {
            "move_left": (230, 216, 173),  # Red for moving left
            "move_right": (0, 255, 255),  # Green for moving right
            "move_forward": (100, 70, 255),  # Blue for moving forward
            "move_backward": (255, 255, 0),  # Yellow for moving backward
            "move_up": (0, 255, 0),  # Cyan for moving up
            "move_down": (0, 0, 255),  # Magenta for moving down
            "rotate_clockwise": (200, 0, 228),  # Purple for rotating clockwise
            "rotate_counterclockwise": (0, 165, 200),  # Orange for rotating counterclockwise
            "stationary": (128, 128, 128),  # Gray for stationary
            "flip_left": (0, 128, 128),  # Teal for flip left
            "flip_right": (128, 128, 0),  # Olive for flip right
            'land': (128, 128, 128),
            'flip_forward': (222, 128, 222),
        }

        self.focal_length = 720
        self.eyes_distance_cm = 6.4
        self.scaled_eye_distance = self.eyes_distance_cm * self.focal_length

    def get_action(self):
        # Calculate absolute values
        abs_pitch = abs(self.pitch)
        abs_yaw = abs(self.yaw)
        abs_roll = abs(self.roll)
        abs_depth = abs(self.delta_depth)

        # Find the largest absolute value
        max_value = max(abs_pitch, abs_yaw, abs_roll, abs_depth)
        if max_value <= 10:
            self.speed = [0, 0, 0, 0]
            return 'stationary'
        # Determine the action based on the largest value and its sign
        if max_value == abs_pitch:
            self.speed = [0, 0, self.pitch, 0]
            if self.pitch > 0:
                if self.pitch > 35:
                    return 'flip_forward'
                return "move_up"
            else:
                if self.pitch < -40:
                    return 'land'
                return "move_down"
        elif max_value == abs_yaw:
            self.speed = [0, 0, 0, -self.yaw]
            if self.yaw < 0:
                return "rotate_clockwise"
            else:

                return "rotate_counterclockwise"
        elif max_value == abs_roll:
            self.speed = [int(1.5 * self.roll), 0, 0, 0]

            if self.roll > 0:
                if self.roll > 40:
                    return 'flip_right'
                return "move_right"
            else:
                if self.roll < -40:
                    return 'flip_left'

                return "move_left"
        elif max_value == abs_depth:
            self.speed = [0, int(-1.5 * self.delta_depth), 0, 0]

            if self.delta_depth > 0:

                return "move_backward"
            else:
                return "move_forward"

    def update_depth(self, landmarks):
        distance_from_cam = self.get_distance_from_cam(landmarks)

        if self.first_measurement is None:
            self.first_measurement = distance_from_cam
        self.delta_depth = distance_from_cam - self.first_measurement

    def get_distance_from_cam(self, landmarks):
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        eyes_distance_pixel = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        return int(self.scaled_eye_distance / eyes_distance_pixel)

    def update_angles(self, cropped_face):
        self.pitch, self.yaw, self.roll = \
            [int(value.item()) for value in self.model.predict(cropped_face)]

    def draw(self, frame, bbox, action, bg_color=(0, 0, 0)):
        # region_xmin, region_ymin, region_width, region_height = bbox
        text_x = bbox[0] - 30
        text_y = bbox[1] - 50
        bbox_width = abs(bbox[2] - bbox[0])
        bbox_height = abs(bbox[3] - bbox[1])

        (text_width, text_height), baseline = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        color = self.action_colors[action]
        # Draw background rectangle for text
        cv2.rectangle(frame, (text_x, text_y - text_height - baseline),
                      (text_x + text_width, text_y + baseline), bg_color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(frame, action, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # cv2.putText(self.frame,text,(self.center_region-15,self.region_ymin-15),cv2.FONT_HERSHEY_PLAIN,1,color,1)
        cvzone.cornerRect(frame, (bbox[0], bbox[1], bbox_width, bbox_height), colorC=color, rt=0,
                          t=5)

    def draw_direction_arrow(self, frame, action):
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        color = (0, 255, 0)
        thickness = 2
        # length = 50

        if action == "move_left":
            length = int(1.5 * self.roll)
            end_point = (center[0] + length, center[1])
            cv2.arrowedLine(frame, center, end_point, self.action_colors[action], thickness, tipLength=0.5)
            cv2.putText(frame, "LEFT", (center[0] + 60, center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)

        elif action == "move_right":
            length = int(1.5 * self.roll)
            end_point = (center[0] + length, center[1])
            cv2.arrowedLine(frame, center, end_point, self.action_colors[action], thickness, tipLength=0.5)
            cv2.putText(frame, "RIGHT", (center[0] - 60, center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)


        elif action == "move_up":
            length = -self.pitch
            cv2.putText(frame, "UP", (center[0] - 25, center[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)

            end_point = (center[0], center[1] + length)
            cv2.arrowedLine(frame, center, end_point, self.action_colors[action], thickness, tipLength=0.5)
        elif action == "move_down":
            cv2.putText(frame, "DOWN", (center[0] - 25, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)

            length = self.pitch
            end_point = (center[0], center[1] - length)

            cv2.arrowedLine(frame, center, end_point, self.action_colors[action], thickness, tipLength=0.5)
        elif action == "move_forward":

            length = self.delta_depth
            end_point = (center[0], center[1] + length)
            cv2.putText(frame, "FWD", (center[0] - 25, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)
            # cv2.arrowedLine(frame, center, end_point, color, thickness, tipLength=0.5)

            draw_dotted_arrowed_line(frame,center,end_point,self.action_colors[action],thickness)


        elif action == "move_backward":
            length = self.delta_depth
            end_point = (center[0], center[1] + length)
            cv2.putText(frame, "BWD", (center[0] - 25, center[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)
            # cv2.arrowedLine(frame, center, end_point, color, thickness, tipLength=0.5)
            draw_dotted_arrowed_line(frame, center, end_point, self.action_colors[action], thickness)
        elif action == "rotate_clockwise":
            start_angle =180
            radius=50
            cv2.putText(frame, "CW", (center[0] - 25, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.action_colors[action], thickness)
            cv2.ellipse(frame, center, (radius, radius), 0, start_angle,start_angle-self.yaw , self.action_colors[action], thickness)

        elif action == "rotate_counterclockwise":
            start_angle =0
            radius=50
            # Draw counter-clockwise rotation arrow
            cv2.putText(frame, "CCW", (center[0] - 25, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.action_colors[action], thickness)
            cv2.ellipse(frame, center, (radius, radius), 0, start_angle, start_angle-self.yaw, self.action_colors[action], thickness)



class FaceDetector:
    def __init__(self):
        self.model = RetinaFace(gpu_id=0)

        self.bbox = [0, 0, 0, 0]
        self.text_offset_x = 30
        self.text_offset_y = 50

    def predict(self, frame):
        faces = self.model(frame)
        if len(faces) == 0:
            print('No face detected')
            return [], []
        bbox, landmarks, score = faces[0]
        if score < 0.7:
            print('Low confidence score')
            return [], []
        bbox = [int(coord) for coord in bbox]

        return bbox, landmarks

    @property
    def bbox_width(self):
        return abs(self.bbox[2] - self.bbox[0])

    @property
    def bbox_height(self):
        return abs(self.bbox[3] - self.bbox[1])

    # def update_head_pose(self, frame):
    #     cropped_face = self.crop_face(frame)
    #     self.head_pose.update_angles(cropped_face)
    #     self.head_pose.update_depth(self.landmarks)

    def crop_face(self, frame, bbox):
        """Crop the face based on the bounding box coordinates."""

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)

        # Add padding to the bounding box
        padding_x = int(0.2 * bbox_height)
        padding_y = int(0.2 * bbox_width)
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_max = min(frame.shape[0], y_max + padding_y)

        # Crop the face region from the frame
        cropped_face = frame[y_min:y_max, x_min:x_max]

        return cropped_face


def crop_face(frame, box):
    """Crop the face based on the bounding box coordinates."""

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in box]

    # Calculate bounding box dimensions
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)

    # Add padding to the bounding box
    padding_x = int(0.2 * bbox_height)
    padding_y = int(0.2 * bbox_width)
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(frame.shape[1], x_max + padding_x)
    y_max = min(frame.shape[0], y_max + padding_y)

    # Crop the face region from the frame
    cropped_face = frame[y_min:y_max, x_min:x_max]

    return cropped_face

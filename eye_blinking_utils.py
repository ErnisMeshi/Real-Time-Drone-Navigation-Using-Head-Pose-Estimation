import cv2
from cvzone.FaceMeshModule import FaceMeshDetector


class DroneInitializationLogic:
    def __init__(self):
        self.mesh_detector = FaceMeshDetector(maxFaces=1)
        self.idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
        self.ratioList = []

    def detect_face_mesh(self, frame):
        _, face_landmarks = self.mesh_detector.findFaceMesh(frame, draw=False)
        if not face_landmarks:
            return
        return face_landmarks[0]



    def get_eye_ratio(self, face_landmarks):
        leftUp = face_landmarks[159]
        leftDown = face_landmarks[23]
        leftLeft = face_landmarks[130]
        leftRight = face_landmarks[243]
        lengthVer, _ = self.mesh_detector.findDistance(leftUp, leftDown)
        lengthHor, _ = self.mesh_detector.findDistance(leftLeft, leftRight)
        return int((lengthVer / lengthHor) * 100)

    def get_average_eye_ratio(self, ratio):
        self.ratioList.append(ratio)

        if len(self.ratioList) > 30:
            self.ratioList.pop(0)
        ratioAvg = sum(self.ratioList) / len(self.ratioList)
        return ratioAvg

    def check_for_eye_blink(self, face_landmarks):
        ratio = self.get_eye_ratio(face_landmarks)
        ratioAvg = self.get_average_eye_ratio(ratio)

        return ratioAvg < 30

    def update_drone_state(self, frame):
        face_landmarks = self.detect_face_mesh(frame)
        if face_landmarks is None:
            return False
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Convert normalized coordinates to pixel values
        h, w, _ = frame.shape
        bbox = (int(min_x * w), int(min_y * h), int((max_x - min_x) * w), int((max_y - min_y) * h))
        print('bbox ',bbox)
        eye_is_blinking = self.check_for_eye_blink(face_landmarks)
        if eye_is_blinking:
            return True
        else:
            return False

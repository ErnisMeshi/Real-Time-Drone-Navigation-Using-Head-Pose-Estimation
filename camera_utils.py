import datetime
import os
import time
from threading import Thread

import cv2


def get_video_filename(folder, base_name='video', extension='avi'):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    return os.path.join(folder, filename)


class VideoRecorder:
    def __init__(self, save_dir, cap, fps=12):
        self.filename = get_video_filename(save_dir)
        self.frame_read = None
        self.recorder_thread = None
        self.keep_recording = False
        self.cap = cap
        self.fps = fps

    def video_recorder(self):
        if isinstance(self.cap, cv2.VideoCapture):
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(self.filename, fourcc, fps, (frame_width, frame_height))
            while self.keep_recording:
                ret, frame = self.cap.read()
                if ret:
                    video.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video.release()
            cv2.destroyAllWindows()
        elif isinstance(self.cap, type(self.cap)):  # Assuming self.cap is a frame read object
            height, width, _ = self.cap.frame.shape
            video = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))
            while self.keep_recording:
                video.write(cv2.cvtColor(self.cap.frame, cv2.COLOR_BGR2RGB))
                time.sleep(1 / self.fps)
            video.release()

    def start_recording(self):
        self.keep_recording = True
        self.recorder_thread = Thread(target=self.video_recorder)
        self.recorder_thread.start()

    def stop_recording(self):
        self.keep_recording = False
        if self.recorder_thread:
            self.recorder_thread.join()


def SimpleRecoroder(save_dir, img_width, img_height):
    vid_cod = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(get_video_filename(save_dir), vid_cod, 8, (img_width, img_height))

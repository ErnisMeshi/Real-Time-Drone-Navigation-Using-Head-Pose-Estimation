import math
import time
import cv2
import numpy as np


class SignalPlot:
    """
    A class for real-time plotting in OpenCV.
    """

    def __init__(self, w=640, h=480, yLimit=[0, 100], interval=0.001, invert=True):
        """
        Initialize the LivePlot object.

        :param w: Width of the plotting window
        :param h: Height of the plotting window
        :param yLimit: Y-axis limits [y_min, y_max]
        :param interval: Time interval for updating the plot
        :param invert: Whether to invert the y-axis
        """

        self.yLimit = yLimit
        self.w = w
        self.h = h
        self.invert = invert
        self.interval = interval
        self.imgPlot = np.zeros((self.h, self.w, 3), np.uint8)
        self.imgPlot[:] = 225, 225, 225
        self.xP = 0
        self.yP = 0
        self.signals = [[] for _ in range(4)]  # List to hold data for four signals
        self.xList = [x for x in range(0, 100)]
        self.ptime = 0

    def update(self, *y_values, colors=((255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0))):
        """
        Update the plot with new y-values for multiple signals.

        :param y_values: The new y-values to plot for each signal
        :param colors: RGB colors for the plot lines

        :return: Updated image of the plot
        """

        # Check if enough time has passed for an update
        if time.time() - self.ptime > self.interval:
            self.imgPlot[:] = 225, 225, 225  # Refresh
            self.drawBackground()  # Draw static parts

            for signal_index, y in enumerate(y_values):
                if self.invert:
                    yP = int(np.interp(y, self.yLimit, [self.h, 0]))
                else:
                    yP = int(np.interp(y, self.yLimit, [0, self.h]))

                self.signals[signal_index].append(yP)
                if len(self.signals[signal_index]) == 100:
                    self.signals[signal_index].pop(0)

                # Draw plot lines
                for i in range(2, len(self.signals[signal_index])):
                    x1 = int((self.xList[i - 1] * (self.w // 100)) - (self.w // 10))
                    y1 = self.signals[signal_index][i - 1]
                    x2 = int((self.xList[i] * (self.w // 100)) - (self.w // 10))
                    y2 = self.signals[signal_index][i]
                    cv2.line(self.imgPlot, (x1, y1), (x2, y2), colors[signal_index], 2)

            self.ptime = time.time()

        return self.imgPlot

    def drawBackground(self):
        """
        Draw the static background elements of the plot.
        """

        cv2.rectangle(self.imgPlot, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)
        cv2.line(self.imgPlot, (0, self.h // 2), (self.w, self.h // 2), (150, 150, 150), 2)

        # Draw grid lines and y-axis labels
        for x in range(0, self.w, 50):
            cv2.line(self.imgPlot, (x, 0), (x, self.h), (50, 50, 50), 1)
        for y in range(0, self.h, 50):
            cv2.line(self.imgPlot, (0, y), (self.w, y), (50, 50, 50), 1)
            y_label = int(self.yLimit[1] - ((y / 50) * ((self.yLimit[1] - self.yLimit[0]) / (self.h / 50))))
            cv2.putText(self.imgPlot, str(y_label), (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)

        cv2.putText(self.imgPlot, "Yaw", (self.w - 100, self.h - 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.putText(self.imgPlot, "Roll", (self.w - 100, self.h - 430), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(self.imgPlot, "Pitch", (self.w - 100, self.h - 410), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(self.imgPlot, "Depth", (self.w - 100, self.h - 390), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)


if __name__ == "__main__":
    xPlot = SignalPlot(w=1200, yLimit=[-100, 100], interval=0.01)
    x = 0
    while True:
        x += 1
        if x == 360: x = 0
        y1_value = math.sin(math.radians(x)) * 100
        y2_value = math.sin(math.radians(x)) * 50
        y3_value = math.sin(math.radians(x)) * 75
        y4_value = math.sin(math.radians(x)) * 25
        imgPlot = xPlot.update(y1_value, y2_value, y3_value, y4_value)

        cv2.imshow("Image", imgPlot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

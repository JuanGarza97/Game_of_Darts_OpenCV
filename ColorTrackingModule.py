import cv2
import numpy as np
import logging


class ColorFinder:
    def __init__(self, track_bar=False):
        self.trackBar = track_bar
        if self.trackBar:
            self.init_trackbars()

    def empty(self, a):
        pass

    def init_trackbars(self):
        """
        To initialize Trackbars . Need to run only once
        """
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, self.empty)
        cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, self.empty)
        cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, self.empty)
        cv2.createTrackbar("Val Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Val Max", "TrackBars", 255, 255, self.empty)

    @staticmethod
    def get_values(self):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        hsvVals = {"h_min": h_min, "s_min": s_min, "v_min": v_min,
                   "h_max": h_max, "s_max": s_max, "v_max": v_max}
        print(hsvVals)
        return hsvVals

    def update(self, img, color=None):
        """
        :param img: Image in which color needs to be found
        :param color: List of lower and upper hsv range
        :return: (mask) bw image with white regions where color is detected
                 (imgColor) colored image only showing regions detected
        """
        img_color = [],
        mask = []

        if self.trackBar:
            color = self.get_values()

        if isinstance(color, str):
            color = self.get_color_hsv(color, )

        if color is not None:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([color['h_min'], color['s_min'], color['v_min']])
            upper = np.array([color['h_max'], color['s_max'], color['v_max']])
            mask = cv2.inRange(img_hsv, lower, upper)
            img_color = cv2.bitwise_and(img, img, mask=mask)
        return img_color, mask

    @staticmethod
    def get_color_hsv(self, color):

        if color == 'red':
            output = {'h_min': 146, 's_min': 141, 'v_min': 77, 'h_max': 179, 's_max': 255, 'v_max': 255}
        elif color == 'green':
            output = {'h_min': 44, 's_min': 79, 'v_min': 111, 'h_max': 79, 's_max': 255, 'v_max': 255}
        elif color == 'blue':
            output = {'h_min': 103, 's_min': 68, 'v_min': 130, 'h_max': 128, 's_max': 255, 'v_max': 255}
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output


def main():
    color_finder = ColorFinder(False)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Custom Orange Color
    hsv_values = {'h_min': 10, 's_min': 55, 'v_min': 215, 'h_max': 42, 's_max': 255, 'v_max': 255}

    while True:
        success, img = cap.read()
        img_red, _ = color_finder.update(img, "red")
        img_green, _ = color_finder.update(img, "green")
        img_blue, _ = color_finder.update(img, "blue")
        img_orange, _ = color_finder.update(img, hsv_values)

        cv2.imshow("Red", img_red)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
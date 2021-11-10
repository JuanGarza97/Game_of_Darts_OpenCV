import cv2
import numpy as np
import pickle


def mouse_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        path.append([x, y])


# All the polygons and the points
poly = []
# Current single polygon
path = []

img = cv2.imread('Resources/Images/empty_board.png')

while True:
    for point in path:
        cv2.circle(img, (point[0], point[1]), 7, (0, 0, 255), cv2.FILLED)

    points = np.array(path, np.int32).reshape((-1, 1, 2))
    img = cv2.polylines(img, [points], True, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_points)
    key = cv2.waitKey(1)
    if key == ord('q'):
        score = int(input("Enter Score: "))
        poly.append([path, score])
        path = []
    if key == ord('p'):
        with open('polygons', 'wb') as file:
            print(poly)
            pickle.dump(poly, file)
        break

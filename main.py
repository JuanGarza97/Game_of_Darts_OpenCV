import cv2
import numpy as np
import ColorTrackingModule as ct
import pickle


def text_rect(img, text, pos, scale=3, thickness=3, color_t=(255, 255, 255), color_r=(255, 0, 255),
              font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, color_b=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), color_r, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color_b, border)
    cv2.putText(img, text, (ox, oy), font, scale, color_t, thickness)

    return img, [x1, y2, x2, y1]


def find_contours(img, img_pre, min_area=1000, sort=True, threshold=0, draw=True, color=(255, 0, 0)):
    contours_found = []
    img_contours = img.copy()
    _, contours, hierarchy = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) == threshold or threshold == 0:
                if draw:
                    cv2.drawContours(img_contours, cnt, -1, color, 3)
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + (w // 2), y + (h // 2)
                cv2.rectangle(img_contours, (x, y), (x + w, y + h), color, 2)
                cv2.circle(img_contours, (x + (w // 2), y + (h // 2)), 5, color, cv2.FILLED)
                contours_found.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        contours_found = sorted(contours_found, key=lambda a: a["area"], reverse=True)

    return img_contours, contours_found


def get_board(img, points, draw=True):
    scale = 1.5
    width, height = int(600), int(580)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output_img = cv2.warpPerspective(img, matrix, (width, height))
    if draw:
        for i in range(4):
            cv2.circle(img, (points[i][0], points[i][1]), 15, (0, 255, 0), cv2.FILLED)
    return output_img


def get_darts(img, color_finder, hsv_values):
    img_blur = cv2.GaussianBlur(img, (7, 7), 2)
    _, mask = color_finder.update(img_blur, hsv_values)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=5)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    cap = cv2.VideoCapture('Resources/Videos/Video2.mp4')
    color_finder = ct.ColorFinder()

    hit_count = 0
    corners = [[377, 52], [944, 71], [261, 624], [1058, 612]]
    hsv_values = {'h_min': 30, 's_min': 34, 'v_min': 0, 'h_max': 41, 's_max': 255, 'v_max': 255}
    img_hit = []
    draw_info_list = []
    game_score = 0

    with open('polygons', 'rb') as file:
        polygons_with_score = pickle.load(file)

    while True:

        success, img = cap.read()

        if success:
            img_board = get_board(img, corners)
            mask = get_darts(img_board, color_finder, hsv_values)

            # Remove previous hit
            for hit in img_hit:
                mask = mask - hit

            img_contours, contours = find_contours(img_board, mask, 3500, draw=False)
            if contours:
                hit_count += 1
                if hit_count == 10:
                    img_hit.append(mask)
                    print("Hit detected")
                    hit_count = 0
                    for polygon in polygons_with_score:
                        center = contours[0]['center']
                        poly = np.array([polygon[0]], np.int32)
                        # Check if the center is inside the polygon -1 -> Outside 1 -> Inside
                        is_inside = cv2.pointPolygonTest(poly, (center[0], center[1]), False)
                        if is_inside == 1.0:
                            draw_info_list.append([contours[0]['bbox'], center, poly])
                            game_score += polygon[1]
                            print(game_score)

            blank_img = np.zeros((img_contours.shape[0], img_contours.shape[1], 3), np.uint8)
            for bounding_box, center, polygon in draw_info_list:
                cv2.rectangle(img_contours, (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 255), 2)
                cv2.circle(img_contours, (center[0], center[1]), 5, (0, 255, 0), cv2.FILLED)
                cv2.drawContours(blank_img, polygon, -1, color=(0, 255, 0), thickness=cv2.FILLED)

            img_contours = cv2.addWeighted(img_contours, 0.7, blank_img, 0.5, 0)

            img_contours, _ = text_rect(img_contours, f'Score: {game_score}', (10, 40), scale=2, offset=20)
            # cv2.imshow("Image", img)
            # cv2.imshow("Image Board", img_board)
            # cv2.imshow("Mask", mask)
            cv2.imshow("Image Contours", img_contours)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()

import numpy as np
import cv2

def floor_perimeter(img, output_file="floor_points.npy"):
    clone = img.copy()
    points = []

    window_name = "Select the perimeter of the floor"
    cv2.namedWindow(window_name)

    def draw_image():
        display = clone.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display, points[i - 1], pt, (255, 0, 0), 2)
        if len(points) > 2:
            cv2.line(display, points[-1], points[0], (0, 255, 0), 2)
        return display

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.imshow(window_name, draw_image())

    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)

    while True:
        cv2.imshow(window_name, draw_image())
        key = cv2.waitKey(1) & 0xFF

        if key == 13:
            if len(points) >= 3:
                print("Polygon closed and saved.")
                np.save(output_file, np.array(points))
                break
            else:
                print("You must select at least 3 points.")
        elif key == ord('c'):
            print("Points cleared, start again.")
            points.clear()

    cv2.destroyAllWindows()
    return points

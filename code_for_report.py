import cv2
import numpy as np

def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask

def process_image(mask):
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, 7, L2gradient=True)
    dilated_edges = cv2.dilate(edges, kernel=np.ones((5, 5), np.uint8), iterations=1)
    return dilated_edges

def scale_contour(contour, scale=2):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    center = np.array([cx, cy])
    
    scaled = (contour - center) * scale + center
    return scaled.astype(np.int32)
    
def detect_contours(dilated_edges):
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scaled_contours = []
    all_vertices = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx)  != 4:
            continue
        scaled = scale_contour(approx, scale=2)
        scaled_contours.append(scaled)
        vertices = scaled.reshape(-1, 2)
        all_vertices.append(vertices)

    return scaled_contours, all_vertices

def draw_contours(image, scaled_contours, all_vertices):
    output = image.copy()
    for vertices in all_vertices:
        for (x, y) in vertices:
            cv2.circle(output, (x, y), 20, (0, 0, 255), -1)

    cv2.drawContours(output, scaled_contours, -1, (0, 0, 255), 6)

    return output
import cv2
import numpy as np
from pytesseract import image_to_string


MAX_COLOR_VAL = 255
BLOCK_SIZE = 15
SUBTRACT_FROM_MEAN = -2
MIN_CELL_SIZE = 20

BLUR_KERNEL_SIZE = (17, 17)
STD_DEV_X_DIRECTION = 0
STD_DEV_Y_DIRECTION = 0

class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def showImage(name, image):
    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def recognize(image, rect, transpose):

    roiRect = Rect(rect.x1, rect.y1, rect.x2, rect.y2)
    if transpose:
        roiRect.x1 = rect.y1
        roiRect.y1 = rect.x1
        roiRect.x2 = rect.y2
        roiRect.y2 = rect.x2

    # expand for better recognition
    if roiRect.x1 - 5 >= 0: roiRect.x1 -= 5
    if roiRect.x2 + 5 <= image.shape[1]: roiRect.x2 += 5
    if roiRect.y1 - 5 > 0: roiRect.y1 -= 5
    if roiRect.y2 + 5 <= image.shape[0]: roiRect.y2 += 5

    roi = image[roiRect.y1:roiRect.y2, roiRect.x1:roiRect.x2]

    return image_to_string( roi, config='--psm 7' ).replace("\n\f", "")

def getDigits(header_cell, transpose) -> []:
    thresholdedOrig = cv2.threshold(header_cell, 200, 255, cv2.THRESH_BINARY)[1]

    thresholded = thresholdedOrig
    if transpose:
        thresholded = cv2.transpose(thresholded)

    morphed = cv2.erode(thresholded, cv2.getStructuringElement(cv2.MORPH_RECT, (1, thresholded.shape[0] * 2)))
    morphed = cv2.bitwise_not(morphed)

    line = morphed[0:1, 0:morphed.shape[1]]
    nonZero = cv2.findNonZero(line)

    startPoint = nonZero[0][0][0] + 1

    digitsCoord = []
    isDigit = True
    for i in range(startPoint, line.shape[1]):
        if line[0][i] == line[0][i - 1]:
            continue
        if isDigit:
            digitsCoord.append([int(startPoint), int(i)])
        startPoint = i
        isDigit = not isDigit

    if len(digitsCoord) == 1:
        return recognize(thresholdedOrig, Rect(0, 0, thresholded.shape[1], thresholded.shape[0] ), transpose)

    digits = []
    isSkipAfterCouple = False
    for i in range(0, len(digitsCoord)):
        if isSkipAfterCouple:
            isSkipAfterCouple = False
            continue

        # check last symbol
        if i == len(digitsCoord) - 1 :
            rect = Rect(digitsCoord[i][0], 0, digitsCoord[i][1], thresholded.shape[0])
            digits.append(recognize(thresholdedOrig, rect, transpose))
            continue

        space = digitsCoord[i+1][0] - digitsCoord[i][1]

        if space < 10:
            rect = Rect(digitsCoord[i][0], 0, digitsCoord[i+1][1], thresholded.shape[0])
            digits.append(recognize(thresholdedOrig, rect, transpose))
            isSkipAfterCouple = True
        else:
            rect = Rect(digitsCoord[i][0], 0, digitsCoord[i][1], thresholded.shape[0] )
            digits.append(recognize(thresholdedOrig, rect, transpose))

    return digits


def extract_table(image):
    ### Get table contours
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)

    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    SCALE = 30
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_width / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)

    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))

    mask = horizontally_dilated + vertically_dilated


    # Get play area bbox
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    contour = sorted(contours, key=lambda c: cv2.contourArea(c))[0]
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    play_bbox = cv2.boundingRect(approx_poly)

    ### Get table bbox
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.05 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]

    # Filter out contours that aren't rectangular. Those that aren't rectangular
    # are probably noise.
    approx_rects = [p for p in approx_polys if len(p) == 4]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]

    # Filter out rectangles that are too narrow or too short.
    bounding_rects = [
        r for r in bounding_rects if MIN_CELL_SIZE < r[2] and MIN_CELL_SIZE < r[3]
    ]

    # The largest bounding rectangle is assumed to be the entire table.
    # Remove it from the list. We don't want to accidentally try to OCR
    # the entire table.
    biggest_bbox = max(bounding_rects, key=lambda r: r[2] * r[3])
    cells = [b for b in bounding_rects if b is not biggest_bbox]

    table_size = int(np.rint(np.sqrt(len(cells))))

    table_x1 = min([x for x, y, w, h in cells])
    table_x2 = max([x+w for x, y, w, h in cells])
    table_y1 = min([y for x, y, w, h in cells])
    table_y2 = max([y+h for x, y, w, h in cells])

    table_bbox = (table_x1, table_y1, table_x2 - table_x1, table_y2 - table_y1)

    ### Get headers
    # Get vertical header
    vertical_header = []
    cell_height = table_bbox[3] / table_size
    for i in range(table_size):
        header_cell = image[table_bbox[1]+int(cell_height*i):table_bbox[1]+int(cell_height*(i+1)),play_bbox[0]:table_bbox[0]-10]
        vertical_header.append(getDigits(header_cell, False))

    # Get horizontal header
    horizontal_header = []
    cell_width = table_bbox[2] / table_size
    for i in range(table_size):
        header_cell = image[play_bbox[1]:table_bbox[1]-10,table_bbox[0]+int(cell_width*i):table_bbox[0]+int(cell_width*(i+1))]
        horizontal_header.append(getDigits(header_cell, True))

    ### Get cells values
    rows_means = np.zeros((table_size, table_size))
    for i in range(table_size):
        for j in range(table_size):
            rows_means[i][j] = np.mean(image[
                table_bbox[1]+int(cell_height*(i+0.25)):table_bbox[1]+int(cell_height*(i+0.75)),
                table_bbox[0]+int(cell_width*(j+0.25)):table_bbox[0]+int(cell_width*(j+0.75))
            ])
    table = cv2.threshold(rows_means.astype(np.uint8), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] == 0
    return table, vertical_header, horizontal_header
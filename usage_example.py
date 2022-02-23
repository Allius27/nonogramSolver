import cv2
from table_extractor import extract_table


image = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
table, vertical_header, horizontal_header = extract_table(image)
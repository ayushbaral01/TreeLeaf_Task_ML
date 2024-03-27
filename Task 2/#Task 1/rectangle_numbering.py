import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_shape(contour):
    area = cv2.contourArea(contour)

    vertices = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    num_vertices = len(vertices)

    if area<1000 and num_vertices<3:
      return 'Line'
    else: return 'Rectangle'

image = cv2.imread(r'C:\Users\Ayush\Desktop\TreeLeaf_Task_ML\Task 2\image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours[1:], -1, (0, 255, 0), 2)

for i, contour in enumerate(contours):
    print(f"Contour {i}:")
    print(f"Number of Points: {len(contour)}")
    area = cv2.contourArea(contour)
    print(f"Area: {area}")
    perimeter = cv2.arcLength(contour, True)
    print(f"Perimeter: {perimeter}")

    # Calculating bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Bounding Rectangle: (x={x}, y={y}, w={w}, h={h})")

    # aspect ratio of bounding rectangle
    aspect_ratio = float(w) / h
    print(f"Aspect Ratio: {aspect_ratio}")

    vertices = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    num_vertices = len(vertices)
    print(f"Number of vertices: {num_vertices}")


    print()


rectangles = []
lines = []

for contour in contours:
    shape = get_shape(contour)
    if shape == 'Rectangle':
      rectangles.append(contour)
    elif shape == 'Line':
        lines.append(contour)
    else:
      pass



peri_dict = {}
for i, contour in enumerate(lines):
  perimeter = cv2.arcLength(contour, True)
  peri_dict[i+1] = perimeter


peri_dict = {k: v for k, v in sorted(peri_dict.items(), key=lambda item: item[1])}
ranks = ['1st', '2nd','3rd', '4th' ]
i=0
for idx, val in peri_dict.items():
  peri_dict[idx] = [val,ranks[i]]
  i+=1

ranked_image = image.copy()
for i, contour in enumerate(lines):
    perimeter = peri_dict[i+1][0]  
    x, y, w, h = cv2.boundingRect(contour) 

    text = peri_dict[i+1][1]
    text_position = (x + w + 10, y + h-35)
    cv2.putText(ranked_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.title("Ranked Image")
plt.imshow(ranked_image)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

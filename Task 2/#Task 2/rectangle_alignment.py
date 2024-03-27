#importing libraries
import numpy as np
import cv2

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_images = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width, height = rect[1]
        angle = rect[2] if width > height else rect[2] + 90

        rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
        processed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        rect_points = cv2.transform(np.array([box]), rotation_matrix).squeeze().astype(int)
        x, y, w, h = cv2.boundingRect(rect_points)
        processed_image = processed_image[y:y + 2 * h, x:x + 2 * w]

        processed_images.append(processed_image)

    return processed_images

# Loading the image
image_path = r"C:\Users\Ayush\Desktop\TreeLeaf_Task_ML\Task 2\image.jpg"
image = cv2.imread(image_path)

# Processing the image
processed_images = process_image(image)

# images
for i, processed_image in enumerate(processed_images):
    cv2.imshow(f'Processed Image {i + 1}', processed_image)

# close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

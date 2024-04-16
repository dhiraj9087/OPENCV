import cv2
import pytesseract
import numpy as np

# # def extract_web_page_title(image_path):
# #     # Load the screenshot image using OpenCV
# #     screenshot = cv2.imread(image_path)

# #     # Convert the image to grayscale
# #     gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

# #     # Apply thresholding to emphasize text
# #     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# #     # Apply GaussianBlur to reduce noise
# #     blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

# #     # Use Tesseract OCR to extract text and bounding box information
# #     custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings as needed
# #     d_boxes = pytesseract.image_to_boxes(blurred, config=custom_config)

# #     # Extract the title and its bounding box coordinates
# #     title_box = None
# #     for box in d_boxes.splitlines():
# #         b = box.split()
# #         if len(b) == 6 and b[0].strip() == 'title':
# #             x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
# #             title_box = (x, y, w, h)
# #             break

# #     # Draw a rectangular box around the detected title text
# #     if title_box:
# #         cv2.rectangle(screenshot, (title_box[1], title_box[2]), (title_box[3], title_box[4]), (0, 255, 0), 2)

# #     # Display the image with the rectangular box
# #     cv2.imshow('Web Page Title Detection', screenshot)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     # Extract the title text
# #     title = pytesseract.image_to_string(blurred, config=custom_config)

# #     return title.strip()

# # if __name__ == "__main__":
# #     # Replace 'screenshot.png' with the path to your screenshot image
# #     screenshot_path = '/Users/dhirajmarathe/Desktop/ss.png'

# #     web_page_title = extract_web_page_title(screenshot_path)

# #     # Print or use the title as needed
# #     print("Web Page Title:", web_page_title)
# import cv2
# import pytesseract

# def extract_web_page_title(image_path):
#     # Load the screenshot image using OpenCV
#     screenshot = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding to emphasize text
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

#     # Use Tesseract OCR to extract text and bounding box information
#     custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings as needed
#     d_boxes = pytesseract.image_to_boxes(blurred, config=custom_config)

#     # Extract the title and its bounding box coordinates
#     title_box = None
#     for box in d_boxes.splitlines():
#         b = box.split()
#         if len(b) == 6 and b[0].strip() == 'title':
#             x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#             title_box = (x, y, w, h)
#             break

#     # Draw a rectangular box around the detected title text
#     if title_box:
#         cv2.rectangle(screenshot, (title_box[1], title_box[2]), (title_box[3], title_box[4]), (0, 255, 0), 2)

#     # Display the image with the rectangular box
#     cv2.imshow('Web Page Title Detection', screenshot)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Extract the title text
#     title = pytesseract.image_to_string(blurred, config=custom_config)

#     return title.strip()

# if __name__ == "__main__":
#     # Replace 'screenshot.png' with the path to your screenshot image
#     screenshot_path = '/Users/dhirajmarathe/Desktop/ss.png'

#     web_page_title = extract_web_page_title(screenshot_path)

#     # Print or use the title as needed
    
#     print("Web Page Title:", web_page_title)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('/Users/dhirajmarathe/Desktop/photo.png')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(grey,1.3,5)
print(faces)
if faces is ():
    print("NO face found")
for (x,y,h,w) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow("img",image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
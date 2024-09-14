import cv2
import numpy as np

# Manually specify the paths to the Haar cascades
face_cascade = cv2.CascadeClassifier('/Users/kristermartinez/open-cv-projects/1/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/kristermartinez/open-cv-projects/1/haarcascade_eye.xml')

# Load an image
image = cv2.imread('/Users/kristermartinez/open-cv-projects/1/Human_faces.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around the eyes
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Add the text "Press any key to close the window, or it will close automatically in 10 seconds."
message = "Press any key to close the window, or it will close automatically in 10 seconds."
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
# font_color = (0, 0, 0)  # Black
font_color = (0, 255, 0)  # Green
# font_color = (0, 255, 255)  # Yellow
# font_color = (255, 255, 0)  # Cyan
thickness = 4
text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]

# Position the text at the bottom center of the image
text_x = int((image.shape[1] - text_size[0]) / 2)  # Centered horizontally
text_y = image.shape[0] - 40  # 30 pixels from the bottom

cv2.putText(image, message, (text_x, text_y), font, font_scale, font_color, thickness)

# Display the result for 10 seconds, or close earlier if a key is pressed
cv2.startWindowThread()  # This may help with windowing issues on some systems
cv2.imshow('Facial Feature Detection', image)

print("Press any key to close the window, or it will close automatically in 10 seconds.")
cv2.waitKey(10000)  # Wait for a key press or 10 seconds, whichever comes first
cv2.destroyAllWindows()

# Display the result
# cv2.imshow('Facial Feature Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

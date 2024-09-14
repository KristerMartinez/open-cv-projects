import cv2
import numpy as np
# import _tkinter
from tkinter import *
from PIL import Image, ImageTk

# Function to handle closing the window
def close_window():
    root.quit()
    root.destroy()

# Manually specify the paths to the Haar cascades
face_cascade = cv2.CascadeClassifier('/Users/kristermartinez/open-cv-projects/1/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/kristermartinez/open-cv-projects/1/haarcascade_eye.xml')

# Load an image
image = cv2.imread('/Users/kristermartinez/open-cv-projects/1/Human_faces.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

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

# Convert the image from BGR (OpenCV format) to RGB for displaying in Tkinter
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
image_tk = ImageTk.PhotoImage(image_pil)

# Create the Tkinter window
root = Tk()
root.title("Facial Feature Detection")

# Create a label and pack the image into the window
label = Label(root, image=image_tk)
label.pack()

# Add the message below the image
message = Label(root, text="Press the red 'X' button to close the window.")
message.pack()

# Set a protocol to properly close the window
root.protocol("WM_DELETE_WINDOW", close_window)

# Run the Tkinter main loop
root.mainloop()

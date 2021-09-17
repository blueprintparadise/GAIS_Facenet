from __future__ import print_function
# Allows use of print like a function in Python 2.x
import uuid
# Import Numpy and OpenCV modules
import numpy as np
import cv2
import os

# Print Debug Info
Image_Directory = r"./images"
print('OpenCV Version:', cv2.__version__)
print('Numpy Version:', np.__version__)
print('OpenCV Video Capture Sample')
print('Type c to Capture and q to Quit')
User_Name = input("Enter your Name: ")
if type(User_Name) == str:
    print("Thank You")
else:
    print("Please add only characters not numbers")
os.makedirs(Image_Directory + "/" + User_Name, exist_ok=True)

try:
    # Initialize GUI window to grab keystrokes when it has focus
    cv2.namedWindow("Capture")
    # Initialize Capture Counter
    cap_cnt = 0
    # Initialize Video Web Camera for capture.
    # The default camera is 0 (usually built-in)
    # The second camera would be 1 and so on
    webcam = cv2.VideoCapture(0)
    # Check if Camera initialized correctly
    success = webcam.isOpened()
    if success == False:
        print('Error: Camera could not be opened')

    while True:
        # Read each frame in video stream
        ret, frame = webcam.read()
        # Display each frame in video stream
        cv2.imshow("Capture", frame)
        if not ret:
            break
        # Monitor keystrokes
        k = cv2.waitKey(1)

        if k & 0xFF == ord('q'):
            # q key pressed so quit
            print("Quitting...")
            break
        elif k & 0xFF == ord('c'):
            # c key pressed so capture frame to image file
            cap_name = "capture_{}.png".format(uuid.uuid1())
            # Creating Directory

            cv2.imwrite(os.path.join(Image_Directory + "/" + User_Name, "capture_{}.png".format(uuid.uuid1())), frame)
            print("Saving {}!".format(cap_name))
            # Increment Capture Counter for next frame to capture
            cap_cnt += 1

    # Release all resources used
    webcam.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print(e)

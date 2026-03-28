## MAKITEST : Suspicious Exam Behaviour Detection System
This serves a guide to the usage and system framework and architeture of the dual gaze tracking system of MAKITEST.

### System Breakdown
The system uses different modules which are split into their own respective processes.
There are __ proccesses in the gaze tacking system:

#### Main Tracker Process (main_trackerprocess.py)
Uses OpenCV to initalize a camera feed of the selected camera index (0-1). 
This module also initializes all the other modules which returns various face data related to the estimated gaze position of the detected face.

#### Face Track Process (face_trackprocessor.py)
This module loads a face landmarker model which is used to detect faces on a webcam feed.
It creates and tracks points located on the different features of a face.

#### Face Distance Process (face_distanceprocessor.py)
This uses degree of polynomials to gather the distance between the the iris of both eyes, and the manually measured distance of the face from the given iris distance.

<img width="720" height="540" alt="image" src="https://github.com/user-attachments/assets/8a1238e4-8496-4b4b-9eac-b3f6588d700e" />
<br><br>
Using the second degree of polynomial gives the process a great balance between efficiency and accuracy. This is implemented by gathering the distance of the irises with their corresponding distance from the camera. The process then uses "polyfit()" to predict the values of input data outside the given samples.



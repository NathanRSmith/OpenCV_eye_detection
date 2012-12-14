Semester Project - Gaze Detection for Android

Class: Computer Vision CSE x0535 Fall 2012
Professor: Dr. Patrick Flynn
Author: Nathan Smith

Purpose: Develop a gaze detection application for Android with prototype in Python.

Language: Python, Java
Dependencies: numpy, opencv(2.4.3) (Python), OpenCV-2.4.2-android-sdk, Android 3+ sdk, Android ndk
Usage:  python tabletSimulator.py
        Android App

The Android application is far from complete, but it can be imported as an Eclipse project and run
on a target Android device.  It originated from the "OpenCV Sample - face-detection" project.
Be aware that it will NOT work in the Android emulator as the emulator does not have cameras.
In order to build, import the OpenCV sdk as described at
http://docs.opencv.org/doc/tutorials/introduction/android_binary_package/android_dev_intro.html
and make sure to set the sdk as a library for the "OpenCV - eye-detection" project with a path
pointing to where it is located in the system.  Make sure the NDKROOT build variable is set to the
ndk location in project properties -> C/C++ Build -> Build Variables.


python_investigation directory

The tabletSimulator.py program originated from the OpenCV examples/python2/facerec_demo.py program.
It reads from a webcam and uses several self made modules: frameProcessing.py, calibrationHandler.py,
gazeFunctions.py, and pupilIsolation.py.

faceEyeDetection.py was where the original development started.  The script may not be completely
functional anymore as it depends on pupilIsolation.py and gazeFunctions.py which has both been
updated more recently.  It also contains commented out code for the original approach of finding
gaze location based on pupil eccentricity until the current approach was started.

Please see FinalReport.docx for more details.

Project code and report located at: https://github.com/NathanRSmith/OpenCV_eye_detection

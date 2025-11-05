# Machine-Learning-Internship-Projects-NULL-CLASS
This repository contains all six machine learning projects developed during my internship at NULL CLASS. Each project focuses on real-world computer vision and AI applications - ranging from human emotion detection to vehicle analysis - implemented using deep learning models and Python-based tools.
📁 Projects Overview
1. 🎓 Attendance System Model

Description:
A smart attendance system that detects and identifies students in a classroom using facial recognition.
If a student is detected, the model automatically marks them as present, records the time of detection, and stores the data in an Excel/CSV file.
Additionally, it recognizes student emotions (happy, sad, neutral, etc.) during attendance.
The model is active only between 9:30 AM to 10:00 AM each day.
GUI: Not mandatory (model-based functionality).

Key Features:

Face recognition for student identification

Emotion detection

Time-based attendance restriction

Auto export to Excel/CSV

2. 🐾 Animal Detection Model

Description:
A machine learning model capable of detecting and classifying multiple animals in images or videos.
It highlights carnivorous animals in red, while showing a pop-up message with the total number of detected carnivores.
A simple and interactive GUI allows users to upload or preview input images and videos.

Key Features:

Multi-animal detection

Species classification

Carnivore highlighting with red boxes

Real-time preview with GUI support

3. 😴 Drowsiness Detection Model

Description:
A model that identifies whether a person is awake or asleep in a vehicle environment.
It detects multiple people in a single image or video, predicts how many are sleeping, and estimates their age.
Sleeping individuals are highlighted in red, and a pop-up message displays the count and age of detected sleeping persons.

Key Features:

Real-time drowsiness detection

Multi-person detection

Age prediction for sleeping individuals

Image & video preview with GUI

4. 🌍 Nationality Detection Model

Description:
A deep learning model that predicts a person’s nationality and emotion from an uploaded image.
Depending on the nationality, the model provides additional insights:

🇮🇳 Indian: Predicts age, emotion, and dress color

🇺🇸 United States: Predicts age and emotion

🌍 African: Predicts emotion and dress color

🌏 Other Nationalities: Predicts nationality and emotion

Key Features:

Image-based nationality classification

Emotion and attribute prediction

Dynamic output based on nationality

GUI with input preview and result display

5. 🤟 Sign Language Detection Model

Description:
A machine learning system to detect sign language gestures and recognize known words in real-time.
The model operates only between 6 PM and 10 PM, ensuring controlled access.
Users can upload images or use live video for detection through a GUI interface.

Key Features:

Real-time sign language recognition

Time-bound functionality (6 PM–10 PM)

Supports both image and live video input

GUI for interactive usage

6. 🚗 Car Colour Detection Model

Description:
A model designed to analyze traffic scenes by detecting car colors and counting vehicles at signals.
Blue cars are marked with red rectangles, while other colors use blue rectangles.
Additionally, if people are detected near the signal, their count is displayed as well.

Key Features:

Car color recognition

Vehicle counting at traffic signals

Person detection and counting

GUI with input image preview

🧩 Tools & Technologies

Python

TensorFlow / Keras

OpenCV

Scikit-learn

NumPy & Pandas

Matplotlib / Seaborn

Tkinter / PyQt for GUI (where applicable)

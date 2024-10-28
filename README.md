# Pose_estimation
Pose estimation</br>

libraries:</br>
--> pip install opencv-python</br>
--> pip install mediapipe</br>

## Overview
This project aims to create a system that performs pose estimation using either a skeleton-based approach or a 3D model-based approach. The system takes inputs from video or images to accurately recognize and analyze various human poses, enhancing applications in fields such as dance, yoga, health, and wellness.

## Objective
- **Develop a web-based application** that takes video or image inputs to estimate human poses.
- **Build AI/ML models** for accurate pose detection, estimation, and prediction.
- **Deliver an intuitive user interface** for a seamless user experience.

## Target Users
1. **Fitness Trainers & Health Professionals** - Monitoring exercise form and fitness progress.
2. **Guardians & Caretakers** - Recognizing sign languages for assistive communication.
3. **Dance Instructors & Learners** - Improving dance techniques through pose analysis.
4. **Therapists** - Tracking movements in rehabilitation for physical therapy.
5. **Automotive Designers** - Assisting drivers by recognizing poses to avoid distractions.

## Use Cases
1. **Fitness Tracking**
   - Exercise form correction and progress tracking.
   - Virtual training sessions for real-time feedback.

2. **Sign Language Recognition**
   - Assists in understanding sign language and providing educational tools.

3. **Dance Improvement**
   - Analyzes dance routines, supports choreography, and provides performance feedback.

## Functional Flow
1. **User Authentication**
   - Users sign in to access pose estimation functionality.
   
2. **Pose Estimation Options**
   - **Manual Estimation**: Uses webcam input for real-time pose detection.
   - **Automatic Estimation**: Allows users to upload image pairs for pose comparison.

3. **Output and Data Storage**
   - Displays pose accuracy and provides visualization.
   - Saves results and logs in MongoDB, including user data, timestamps, and accuracy scores.

## Features
1. **Webcam Pose Estimation**: Counts repetitions for exercises if performed accurately in front of the webcam.
2. **Image Comparison**: Allows users to upload reference and comparison images to calculate pose similarity.
3. **Pose Similarity Score**: Provides an accuracy score to show pose correctness and offer feedback.
4. **History Log**: Logs each session for user reference and tracking.

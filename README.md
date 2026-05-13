# ❤️ Heart Rate Measurement using Image Processing and Computer Vision

---

## 📌 Overview
This project focuses on **non-contact heart rate measurement** using **image processing and video signal analysis**. The system extracts subtle color changes from facial regions in a video stream to estimate heart rate (BPM).

---

## 📚 1. Introduction

### 1.1 Problem Statement
Traditional heart rate measurement methods require physical contact sensors, which may not always be convenient or available. This project addresses the need for a **contactless and convenient heart rate monitoring system**.

### 1.2 Motivation and Objectives
- Develop a non-contact heart rate estimation system  
- Apply computer vision and signal processing techniques  
- Extract physiological signals from video input  

### 1.3 Scope
The system focuses on:
- Face-based heart rate estimation  
- Video-based signal processing  
- Real-time or near real-time analysis  

---

## 📖 2. Literature Review
The project is based on previous research in:
- Remote photoplethysmography (rPPG)
- Video signal processing
- Face detection and tracking techniques

---

## ⚙️ 3. Methodology

### 3.1 Video Signal Processing
Video frames are extracted and processed as time-series signals for analysis.

### 3.2 Face Detection and Tracking
The system detects and tracks the face region to maintain a stable region of interest.

### 3.3 Region of Interest (ROI)
A specific facial area is selected to extract color intensity variations.

### 3.4 Normalization and Detrending
The ROI signal is normalized and filtered to reduce noise and lighting variations.

### 3.5 Bandpass Filtering and FFT
Signal processing techniques such as bandpass filtering and Fast Fourier Transform (FFT) are applied to isolate the heart rate frequency.

### 3.6 Heart Rate Estimation
The dominant frequency from the processed signal is converted into BPM (beats per minute).

### 3.7 Experimental Tools
- Python / OpenCV  
- Signal processing libraries (NumPy, SciPy)  

---

## 📊 4. Results and Discussion
The system successfully extracts pulse signals from video input and estimates heart rate with reasonable accuracy under stable lighting conditions.

---

## ✅ 5. Conclusion
This project demonstrates the feasibility of estimating heart rate using **computer vision and signal processing techniques** without physical contact sensors.

---

## 🚀 Keywords
Heart Rate, Image Processing, Computer Vision, rPPG, FFT, Face Detection

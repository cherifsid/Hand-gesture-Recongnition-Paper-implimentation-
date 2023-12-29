# Hand Gesture Recognition Project

## Overview
This project focuses on developing an algorithm for recognizing hand gestures using image processing techniques. It is an exploration into the realm of computer vision and gesture recognition.

## Objectives
- Technical Objective: Master image processing for specific gesture recognition.
- Scientific Objective: Understand the principles of computer vision in gesture recognition.

## Scientific Program
### Entry and Preprocessing (Segmentation)
- RGB image capture of hand gestures.
- ![RGB image capture of hand gestures.](/results/rgb.png)
- Conversion to YCbCr color space for better segmentation.
- ![Image 1: RGB to YCbCr Conversion](/results/ycbr.png)
- Black and white conversion and median filtering.
![Image 2: B&W Conversion](/results/median.png)

### Recognition Algorithm
- The proposed algorithm starts by identifying the middle finger, then draws lines (slopes) from from a reference point near the middle finger.
- It defines a "flip count" as the number of alternations between black and white pixels along a scanned path. and white pixels along a scanned trajectory.
- The algorithm counts the number of reversals along this trajectory, which corresponds to the numerical value of the hand gesture
- ![Image 3: Algorithm Step 1](/results/pn.png)

### Calculation Logic
- Explain how the gesture is calculated.
- For each number from 1 to 5, the document describes specific criteria based on the number of flips (pixel change from white to black and vice versa).
- For Number 0, solidity is calculated if it is >0.8, so the predicted gesture is 0 (according to experiments experiments conducted by the authors of the work).
![Image 4: Calculation Logic](/results/surface.png)


## Experimentations
![Image 5: Experiment Result 1](/results/final.png)

## Limitations
- Although this method has demonstrated its robustness and effectiveness in recognizing hand gestures on a wide range of images, representing numbers from 0 to 5, under various nail and lighting conditions, it does have its limitations. In particular, the algorithm tends to fail in prediction when fingers are too close together, leaving little space between them. This proximity reduces the accuracy of detection of color transitions ("flips"), essential for correct finger counting. counting.

## Contributors
- Sidhoum Mohamed Cherif
- Nedjai Azeddine



## Bibliography
- REAL-TIME NUMERICAL 0-5 COUNTING BASED ON HAND-FINGER GESTURES RECOGNITION Article  in  Journal of Theoretical and Applied Information Technology · January 2017


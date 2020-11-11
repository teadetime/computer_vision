# Gesture Controlled Robot via TensorFlow Object Detection and OpenCV

## Intro: 
__*Ever wanted to control a robot via hand gestures?*__ You’ve come to the right place! This project is an implementation of a hand gesture controlled robot. This project was completed by Nathan Faber and Sander Miller, for the course [Computational Robotics](https://comprobo20.github.io/) @[Olin College of Engineering](https://www.olin.edu/).

This project was chosen to further explore algorithms in computer vision. This project was scoped to allow us to learn about multiple parts of computer vision processing. Our plan was to utilize a deep-learning object detection model to localize the hand and then use color thresholding and a variety of image processing techniques to determine how many fingers are showing. This data is then used to change/control the neato’s movement.

## Goal of project
This project seeks to show a working implementation of the above methods/techniques such that others could understand our process and that we could understand more about how these types of problems are tackled in industry.

## Overview / an example of it in action!
Read on to see how this type of control is implemented.
![Running via xy hand control](/docs/images/handcontrol.gif)
The above screen capture shows one way of control taht we implemented. The robot is controlled based on the x/y position of the localized hand.

INSERT OTHER CONTROL METHOD and GIF HERE

## Machine Learning Model Development Process
Delvoping the ML object deteciton model can be broken down into multiple steps.
For even more detail of the process to create this model please look [at our other documentation](/docs/model_dev.md)
### Preliminary Decisions
- We chose to use the Tensorflow Object Detection API. This was chosen over PyTorch because of our prior experience with it.
- Inputs would be webcam data in real time
- Interested in hand orientation/pose

### Research/Experimental Phase
The original plan was to train our own Model to detect different hand poses based on data we created.
For this to be feasible we could only label a dataset for full frame detection (adding bounding boxes to images is extremely labor intensive).
This was tested at a very small scale with roughly 50 or so images in 3 different labels. The results were not at all promising, however that was likely because the model had very little training data and we were not utilizing a transfer learning approach.
It also became evident that we were not interested in pouring tons of time into making a dataset as opposed to exploring more computer vision processes.

This led to a pivot to an object detection model. We were very interested in the repo here: [github.com/victordibia/handtracking](https://github.com/victordibia/handtracking)
We planned to port this code over to the upgraded/new TF2 object detection API. This project also made use of a labeled dataset: [Egohands](vision.soic.indiana.edu/projects/egohands/)

This dataset and paper gave us confidence that we could train a model that would recognize hands and allow us to select a bounding box around them.
This bounding box will then be cropped out and passed on to more image processing to determine the pose/shape of the hand


### Development Phase (Object Detection)
This process can be broken down into several different steps.
##### Obtaining Training Data
- The above [Egohands](vision.soic.indiana.edu/projects/egohands/) dataset is publicly available. This contains a set of all the images and annotations. These annotations are bbox level annotations and split into 4 different classes.

##### Converting Training Data/labels
- The annotations are saved into a format that can't be read by the tensorflow dataset creators.
- Modifying scripts from [github.com/victordibia/handtracking](https://github.com/victordibia/handtracking) took the processing from .m files to .xml to .csv which can be read by tensorflow scripts.
- The images and .csvs are converted together in to TFRecords (.record). There is a train.record and test.record to reflect the training and validation image sets. TFRecords are the final format used in training.

##### Choosing a Model to Transfer Learn off of
- Tensorflow had a [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) containing models that are suitable to fine tune for object detection tasks.
- We had experience with mobilenetv2, and thought that the 320x320 size would be sufficient for our webcam and would be trainable on our hardware (GTX 1060 6gb)
- We also trained a model on EfficientNet 320x320. his model showed similar results of accuracy but took much longer to train and was too slow at performing inference to be effective in real time on CPU-only machine (one of our requirements)
##### The model at work!
Here is the first run of the Mobilenetv2 320x320 custom model
![The model](/docs/images/object%20detection.gif)

# Machine Learning Model Development Process
This doc explains the process used to create the Object Detection model and decisions made throughout the process.

## Preliminary Decisions
- We chose to use the Tensorflow Object Detection API. This was chosen over PyTorch because of our prior experience with it. 
- Inputs would be webcam data in real time
- Interested in hand orientation/pose

## Research/Experimental Phase
The original plan was to train our own Model to detect different hand poses based on data we created. 
For this to be feasible we could only label a dataset for full frame detection (adding bounding boxes to images is extremely labor intensive).
This was tested at a very small scale with roughly 50 or so images in 3 different labels. The results were not at all promising, however that was likely because the model had very little training data and we were not utililzing a transfer learning approach. 
It also became evident that we were not interestd in pouring tons of time into making a dataset as opposed to eploring more computer vision processes. 

This led to a pivot to an object detection model. We were very interested in the repo here: (https://github.com/victordibia/handtracking)[https://github.com/victordibia/handtracking]
We planned to port this code over to the upgraded/new TF2 object detection API. This project also made use of a labeled dataset: (Egohands)[vision.soic.indiana.edu/projects/egohands/]

This dataset and paper gave us confidence that we could train a model that would recognize hands and allow us to select a bounding box around them. 
This bouding box would then be cropped out and passed on to more image processing to determine the pose/shape of the hand


## Development Process (Object Detection)
This process can be broken down into several different steps.
#### Obtaining Training Data

#### Converting Training Data/labels

#### Choosing a Model to Transfer Learn off of

#### Training the model

#### Exporting the model

#### Testing/Using the model


## Extra Notes
My sample folder structure:



## Resources
Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks,
https://github.com/victordibia/handtracking

vision.soic.indiana.edu/projects/egohands/

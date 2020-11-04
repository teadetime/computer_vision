# computer_vision
# Schedule/Todo List
- Documentation 
  - Install TF directions (NATHAN)
  - Resources used (BOTH)
  - Code cleanup (BOTH)
  - Make general outline of layout
  - Choose place to upload model (NATHAN)
- CV and ROS integration (NATHAN)
- Post processing of BBOX over time for smooth results (NATHAN)
  - larger bbox?
  - Centroid same as previous frames
- BBOX -> command (SANDER)
  - DONE BBox Creation 
  - IN PROGRESS Segmenting out the hand
  - TODO Process Segmented hand to a command
    - Finding fingertips
    - Finding Anti-fingertips
 -Extras
   - Opencv play from video
   - Test EfficientNet Model
   - Train model longer
   - Neato control based on xy coordinate of bbox or two hands?!

## Wednesday
- Sander segments hand from BBOX
- Nathan to run cv in ros node

## Thursday
- Nathan runs model with cv/webcam in ros node

## Friday

## Saturday
- Sander to finish output from bbox

## Sunday

## Monday
- Class time/talk
- MVP report minus visuals and some final touches






## Who is on your team?
Nathan and Sander

## What is the main idea of your project?
Use hand signals to direct the neato. This could take form in using image classification, object detection, or other image processing techniques

## What are your learning goals for this project?
Nathan
- Implement webcam usage with opencv and ROS
- Experiment with multiple image processing techniques(ie using mobilenet object detection as well as simplistic hue masking)
- Attempt to understand hand/finger orientation
- Create portfolio ready code that is demonstrated working along with a portfolio quality writeup

Sander
- Implement a deep learning model using a provided data set
- Become more familiar with various computervision techniques
- Create an interactive control system through webcam integration
- Produce a professional final deliverable for my portfolio

## What algorithms or computer vision areas will you be exploring?
- Mobilenet image classifier (tensorflow)
- mobilenet object detection (tensorflow)
- Image masking via color and finger localization via polygons

What components of the algorithm will you implement yourself, which will you use built-in code for? Why?
- We will be retraining the final layers of mobilenet
- We will also be going off of someone's implementation of fingertip localization (with the intention of modifying an expanding it)

## What is your MVP?
- Neato that moves and stops with different hand positions

## Desired Goal
- Sophisticated behavior of neato via complex hand movements or patterns.


Current Plan
- Nathan will work on trainging the hand detector object detection
- Sander will work on finger point detection and hand masking!
- Nathan Will work on getting some basic images
- ie number of fingers to control speed or closeness to camera, make neato follow the hand

## What is a stretch goal?
- Utilize stereo camera to even crazier things!
- Make this work for more than one person ie different skin tones / backgrounds

## What do you view as the biggest risks to you being successful (where success means achieving your learning goals) on this project?
- labeled data may be needed to get the Neural nets working, this could be problematic
- Implementing the pipeline from webcam to opencv to tensorflow to the neato seems like the riskiest part
- Tensorflow can be a pain to get running on different hardware
- Polygon mapping for fingers may only work for certain orientations of the hand
- Overscoped to tools that we aren't familiar with (deeplearning and openCV)

## What might you need from the teaching team for you to be successful on this project?
- OpenCV help(Nathan has no experience)
- Image processing help/techniques like making a mask etc


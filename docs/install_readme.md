# Install notes
There are many messy dependencies needed for this codebase to work.

### High level requirements/notes
- Use python3.7+ (we used 3.8) and pip3
- Use our requirements.txt to install required python packages
- OpenCV 4+
- Tensorflow (TF) 2.3+ (with or without GPU support)
- Tensorflow Object Detection 2 api (The hardest part of the install unless installing gpu drivers for TF)
- We used Ubuntu 20.04 but a similar proceedure should work on windows or other Linux distros
 

## Install Directions
__Pre-req:__

Follow instructions [Comprobo ROS setup](https://comprobo20.github.io/How%20to/setup_your_environment) to make sure your ros installation is correct



1. Install Python(3.8 worked for us) an pip3
2. Clone this repo into `~/catkin/src/` and navigate to its root ie: `$ cd ~/catkin/src/computer-vision/`
3. Install the packages in requirements.txt via 
      -   `$ pip3 install -r requirements.txt`
4. Install Tensorflow Object Detection API
  - Follow official instructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
  - Note that GPU support is unneccesary
  - Ensure that the tests mentioned correctly run
5. Go test!
  - run `opencv_test.py`with: `python3 opencv_test.py`
  - If you see bounding boxes around your hands and the script doens't crash you did everything right!
6. Go run the ros node
  - Start the neato simulator: `$ roscore` 
  - In another terminal `$ roslaunch neato_gazebo neato_empty_world.launch`
  - Start the hand detection ros node! `$ roslaunch hand neato`

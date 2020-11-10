# Machine Learning Model Development Process
This doc explains the process used to create the Object Detection model and decisions made throughout the process.

## Preliminary Decisions
- We chose to use the Tensorflow Object Detection API. This was chosen over PyTorch because of our prior experience with it.
- Inputs would be webcam data in real time
- Interested in hand orientation/pose

## Research/Experimental Phase
The original plan was to train our own Model to detect different hand poses based on data we created.
For this to be feasible we could only label a dataset for full frame detection (adding bounding boxes to images is extremely labor intensive).
This was tested at a very small scale with roughly 50 or so images in 3 different labels. The results were not at all promising, however that was likely because the model had very little training data and we were not utilizing a transfer learning approach.
It also became evident that we were not interested in pouring tons of time into making a dataset as opposed to exploring more computer vision processes.

This led to a pivot to an object detection model. We were very interested in the repo here: [github.com/victordibia/handtracking](https://github.com/victordibia/handtracking)
We planned to port this code over to the upgraded/new TF2 object detection API. This project also made use of a labeled dataset: [Egohands](vision.soic.indiana.edu/projects/egohands/)

This dataset and paper gave us confidence that we could train a model that would recognize hands and allow us to select a bounding box around them.
This bounding box will then be cropped out and passed on to more image processing to determine the pose/shape of the hand


## Development Process (Object Detection)
This process can be broken down into several different steps.
#### Obtaining Training Data
- The above [Egohands](vision.soic.indiana.edu/projects/egohands/) dataset is publicly available. This contains a set of all the images and annotations. These annotations are bbox level annotations and split into 4 different classes.

#### Converting Training Data/labels
- The annotations are saved into a format that can't be read by the tensorflow dataset creators.
- Modifying scripts from [github.com/victordibia/handtracking](https://github.com/victordibia/handtracking) took the processing from .m files to .xml to .csv which can be read by tensorflow scripts.
- The images and .csvs are converted together in to TFRecords (.record). There is a train.record and test.record to reflect the training and validation image sets. TFRecords are the final format used in training.

#### Choosing a Model to Transfer Learn off of
- Tensorflow had a [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) containing models that are suitable to fine tune for object detection tasks.
- We had experience with mobilenetv2, and thought that the 320x320 size would be sufficient for our webcam and would be trainable on our hardware (GTX 1060 6gb)
- We also trained a model on EfficientNet 320x320 FILL IN RESULTS HERE

#### Setting up the model
- After choosing a model you must download a saved "checkpoint" of the model that contains a starting point to train from
- The parameters of the model are set in a file called "pipeline.config"
  - This file requires edits to indicate how long to train, paths to the dataset, model checkpoint, labels, and output directory

#### Training the model
- Once the pipeline is configured the model can be trained with `$ some command`
- The model will save checkpoints periodically that can be trained off from if the process is stopped
- Training may be monitored by tensorboard `$ tensorboard --logdir=models/YOUR_MODEL_OUTPUT_DIR`
  - Additionally you can run evaluation metrics to see how the model does on the evaluation dataset in real time `$ COMMAND --checkpoint-dir` *NOTE* This will not work if you have one GPU. To perform evaluation on cpu add `cpu only` in a copy of tf_train_main.py and adjust and rerun the command above
- I trained the model for ~8hrs and 50,000 epochs on a GTX 1060 6gb

#### Exporting the model
- The final checkpoint from the model is stored in a format that allows it to be retrained from. This is a very large file format and not practical if we simply want to use the model to perform inference.
- We convert it into the tf.savedModel format using the command `$ python3 exporter_main_v2.py --pipeline_config_path models/effnet/pipeline.config --trained_checkpoint_dir models/effnet/ --output_directory exported-models/my_effnet
`
- This model is small enough to put onto github, [here is our mobilenetv2 hand detector](/my_model_mnetv2)

#### Testing/Using the model
- Importing this model in python is simple:  ` model = tf.saved_model.load(PATH_TO_SAVED_MODEL)`
- More details on running the model can be found [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html)


## Resources
Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks,
https://github.com/victordibia/handtracking

vision.soic.indiana.edu/projects/egohands/

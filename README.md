# Gait Detection

You can read about the detailed approach used in this repository in my [Medium article]().

## Environment Setup
To setup the environment, run the following command:
```commandline
pip install -r requirements.txt
```

## Prerequisites:
The code base makes use of Human Pose Estimation model. We have made use of Thunder variant of MoveNet model. If you wish to download the same model, you may run this command in your terminal.
```commandline
wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite
```

### Video data
You have to collect 6-10 seconds video of people walking in front of the camera. For each person, you will have to keep a separate video. Collect all these videos in a single folder.

## Keypoints Data Generation
Much of the code inside data generation is taken from official tutorial of TensorFlow's [Human Pose estimation](https://www.tensorflow.org/hub/tutorials/movenet). The code has been structured inside the Python classes inside this repository.

You will have to set the path of video folder and the location where you wish to save the Numpy file. The details regarding these and other variables are present at the very end inside the `DataGeneration.py`

Once the required variables are set, generate the Keypoints data with various augmentation by running this command:
```commandline
python DataGeneration.py
```

## Training Model
You will have to set up the various parameters related to hyperparameter tuning and locations of your data inside the `Train.py`. The variables are present at the end and they have been commented for ease of understanding.

Once the variables are set, you can proceed the training by running the below command:
```commandline
python Train.py
```
Once the training is done, it will also evaluate the accuracy of the model against the test dataset.


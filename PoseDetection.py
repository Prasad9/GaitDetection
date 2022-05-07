import os
import tensorflow as tf


class PoseDetection:
    def __init__(self, params):
        self._input_size = params['INPUT_SIZE']

        tflite_model_path = os.path.abspath(params['TFLITE_MODEL_PATH'])
        # Initialize the TFLite interpreter
        self._interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self._interpreter.allocate_tensors()

    def run_movenet(self, input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        self._interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self._interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self._interpreter.get_tensor(output_details[0]['index'])
        # keypoints = keypoints_with_scores[0, 0, :, :2]
        return keypoints_with_scores
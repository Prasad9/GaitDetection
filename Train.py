import os

from Model import Model
from DataReader import DataReader


class Train:
    def __init__(self, params):
        self._epochs = params['EPOCHS']
        self._train_steps = params['TRAIN_STEPS']
        self._val_steps = params['VAL_STEPS']
        self._test_steps = params['TEST_STEPS']
        self._save_folder = params['SAVE_FOLDER']

        self._model = Model(params)
        self._data_reader = DataReader(params)

    def train(self):
        train_ds = self._data_reader.get_train_ds()
        val_ds = self._data_reader.get_val_ds()
        self._model.siamese_model.fit(train_ds,
                                      validation_data=val_ds,
                                      epochs=self._epochs,
                                      steps_per_epoch=self._train_steps,
                                      validation_steps=self._val_steps)

        save_path = os.path.join(self._save_folder, 'siamese_model')
        self._model.save_model(save_path)

    def test(self, should_load=True):
        if should_load:
            load_path = os.path.join(self._save_folder, 'siamese_model')
            self._model.load_model(load_path)

        test_ds = self._data_reader.get_test_ds()
        self._model.siamese_model.evaluate(test_ds, steps=self._test_steps)


if __name__ == '__main__':
    params = {
        'DATA_FOLDER': './Data/TimeSeriesData',                 # Data folder holding time series data
        'SAVE_FOLDER': './Model',                               # Folder where trained model should be saved
        'N_LABELS': 239,                                        # Total number of user videos in your dataset
        'TRAIN_VAL_TEST_RATIO': [0.90, 0.05, 0.05],             # Split of size of train validation test dataset
        'TIME_STEPS': 200,                                      # Maximum number of frames to process inside each video
        'BATCH_SIZE': 32,                                        # Batch size during training and evaluation
        'DATA_DIMENSION': 34,                                   # Number of keypoints (multiplied by coordinates in each keypoint)
        'EPOCHS': 10,                                          # Number of epochs to train
        'HIDDEN_UNITS': [60],                                    # List of hidden units inside LSTM layers
        # We are going to run on generator to yield data. Decide how many times do you want to run on
        # each epoch.
        'TRAIN_STEPS': 500,
        'VAL_STEPS': 100,
        'TEST_STEPS': 100
    }

    t = Train(params)
    t.train()
    t.test()

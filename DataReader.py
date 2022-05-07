import os
import glob
import random
import numpy as np
import tensorflow as tf


class DataReader:
    def __init__(self, params):
        self._data_folder = os.path.abspath(params['DATA_FOLDER'])
        self._n_labels = params['N_LABELS']
        self._train_val_test_ratio = params['TRAIN_VAL_TEST_RATIO']
        self._time_steps = params['TIME_STEPS']
        self._batch_size = params['BATCH_SIZE']
        self._data_dimension = params['DATA_DIMENSION']

        self._train_data, self._val_data, self._test_data = self._read_train_val_test_data()

        self._train_ds = self._prepare_ds(self._train_generator_func)
        self._val_ds = self._prepare_ds(self._val_generator_func)
        self._test_ds = self._prepare_ds(self._test_generator_func)

    def _prepare_ds(self, generator_func):
        output_signature = ({'input1': tf.TensorSpec(shape=(self._batch_size, self._time_steps, self._data_dimension),
                                                     dtype=tf.float32),
                            'input2': tf.TensorSpec(shape=(self._batch_size, self._time_steps, self._data_dimension),
                                                    dtype=tf.float32)},
                            tf.TensorSpec(shape=(self._batch_size,), dtype=tf.int32))
        ds = tf.data.Dataset.from_generator(generator_func, output_signature=output_signature)
        return ds

    def get_train_ds(self):
        return self._train_ds

    def get_val_ds(self):
        return self._val_ds

    def get_test_ds(self):
        return self._test_ds

    def _train_generator_func(self):
        return self._base_generator_func(self._train_data)

    def _val_generator_func(self):
        return self._base_generator_func(self._val_data)

    def _test_generator_func(self):
        return self._base_generator_func(self._test_data)

    def _base_generator_func(self, dataset):
        gt_arr = list(range(self._n_labels))
        while True:
            data1_arr, data2_arr, labels_arr = [], [], []
            for _ in range(self._batch_size):
                is_same = random.choice([True, False])
                if is_same:
                    label = random.choice(gt_arr)
                    file_paths = random.sample(dataset[label], k=2)
                    file_path1 = file_paths[0]
                    file_path2 = file_paths[1]
                else:
                    labels = random.sample(gt_arr, k=2)
                    file_path1 = random.choice(dataset[labels[0]])
                    file_path2 = random.choice(dataset[labels[1]])

                def trim_or_pad_np(data):
                    pad_at = self._time_steps - data.shape[0]
                    if pad_at <= 0:
                        return data[:self._time_steps]
                    return np.pad(data, [[0, pad_at], [0, 0]], constant_values=0)

                data1 = trim_or_pad_np(np.load(file_path1))
                data2 = trim_or_pad_np(np.load(file_path2))
                label = int(is_same)

                data1_arr.append(data1)
                data2_arr.append(data2)
                labels_arr.append(label)

            data1_arr = np.array(data1_arr)
            data2_arr = np.array(data2_arr)
            labels_arr = np.array(labels_arr)
            yield {'input1': data1_arr, 'input2': data2_arr}, labels_arr

    def _read_train_val_test_data(self):
        train_data = {}
        val_data = {}
        test_data = {}
        for label in range(self._n_labels):
            file_paths = glob.glob(self._data_folder + '/{}_*.npy'.format(label))
            random.shuffle(file_paths)

            train_count = int(self._train_val_test_ratio[0] * len(file_paths))
            train_data[label] = file_paths[:train_count]

            val_count = int(self._train_val_test_ratio[1] * len(file_paths))
            val_data[label] = file_paths[train_count: train_count + val_count]

            test_data[label] = file_paths[train_count + val_count:]
        return train_data, val_data, test_data


if __name__ == '__main__':
    params = {
        'DATA_FOLDER': './Data/TimeSeriesData',
        'N_LABELS': 239,
        'TRAIN_VAL_TEST_RATIO': [0.90, 0.05, 0.05],
        'TIME_STEPS': 200,
        'BATCH_SIZE': 32,
        'DATA_DIMENSION': 34,
        'TRAIN_STEPS': 500,
        'VAL_STEPS': 100,
        'TEST_STEPS': 100
    }
    d = DataReader(params)

    train_ds = d.get_train_ds()
    for data, label in train_ds.take(1):
        print(data['input1'].shape)
        print(data['input2'].shape)
        print(label)

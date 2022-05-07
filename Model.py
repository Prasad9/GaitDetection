import tensorflow as tf


class Model:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']
        self._units = params['HIDDEN_UNITS']
        self._time_steps = params['TIME_STEPS']
        self._data_dimension = params['DATA_DIMENSION']

        self._base_model, self.siamese_model = self._build_model()

    def _build_model(self):
        base_model = self._build_base_model()

        input1_layer = tf.keras.layers.Input(shape=(self._time_steps, self._data_dimension), name='input1')
        input2_layer = tf.keras.layers.Input(shape=(self._time_steps, self._data_dimension), name='input2')
        input1_features = base_model(input1_layer)
        input2_features = base_model(input2_layer)

        combined_features = tf.keras.layers.concatenate([input1_features, input2_features])
        output_score_layer = tf.keras.layers.Dense(1, activation='sigmoid')(combined_features)

        siamese_model = tf.keras.Model(inputs=[input1_layer, input2_layer], outputs=[output_score_layer])
        siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return base_model, siamese_model

    def print_model(self):
        self._base_model.summary()
        self.siamese_model.summary()

    def _build_base_model(self):
        inputs = tf.keras.layers.Input(batch_shape=(self._batch_size, self._time_steps, self._data_dimension))
        hidden = inputs
        for layer_no, unit in enumerate(self._units):
            return_sequences = not (layer_no == len(self._units) - 1)
            hidden = tf.keras.layers.LSTM(unit, return_sequences=return_sequences, stateful=False)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
        output = tf.keras.layers.Dense(8, activation='relu')(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def save_model(self, savepath):
        self.siamese_model.save(savepath)

    def load_model(self, loadpath):
        self.siamese_model = tf.keras.models.load_model(loadpath)


if __name__ == '__main__':
    params = {
        'BATCH_SIZE': 32,
        'HIDDEN_UNITS': [60],
        'TIME_STEPS': 300,
        'DATA_DIMENSION': 34
    }
    m = Model(params)
    m.print_model()

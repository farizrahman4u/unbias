from keras.layers import Input, Lambda
from keras.layers import multiply, add
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as K
from keras.activations import softmax
from keras.utils import Progbar
from keras.engine.saving import pickle_model, unpickle_model
import warnings
import numpy as np
import math

try:
    import h5py
except ImportError:
    h5py = None


class Unbias(object):
    def __init__(self, task, morpher, discriminators):
        self.task = task
        self.morpher = morpher
        if type(discriminators) not in [list, tuple]:
            discriminators = [discriminators]
        self.discriminators = discriminators
        self._validate_inner_models()
        self._build()

    def _check_shape_compatibility(self, shape1, shape2):
        typ1 = type(shape1)
        typ2 = type(shape2)
        assert typ1 == typ2, 'Shape mismatch: {} and {}'.format(shape1, shape2)
        if typ1 is list:
            n1 = len(shape1)
            n2 = len(shape2)
            assert n1 == n2, 'Shape mismatch: {} and {}'.format(shape1, shape2)
            for s1, s2 in zip(shape1, shape2):
                self._check_shape_compatibility(s1, s2)
        elif typ2 is tuple:
            ndim1 = len(shape1)
            ndim2 = len(shape2)
            assert ndim1 == ndim2, 'Shape mismatch: {} and {}'.format(shape1, shape2)
            for d1, d2 in zip(shape1, shape2):
                match = None in [d1, d2] or d1 == d2
                assert match, 'Shape mismatch: {} and {}'.format(shape1, shape2)

    def _validate_inner_models(self):
        morpher_out_shape = self.morpher.output_shape
        if isinstance(morpher_out_shape, list):
            raise ValueError('Morpher has multiple outputs.')
        task_in_shape = self.task.input_shape
        try:
            self._check_shape_compatibility(morpher_out_shape, task_in_shape)
        except AssertionError as e:
            err = 'Task model is not compatible with morpher model. ' + str(e) 
            raise ValueError(err)
        for i, disc in enumerate(self.discriminators):
            disc_out_shape = disc.output_shape
            if isinstance(disc_out_shape, list):
                err = 'Discriminator at index {} has multiple outputs.'.format(i)
                raise ValueError('err')
            if len(disc_out_shape) != 2:
                err = 'Discriminator at index {} has rank {} output. Expected rank 2.'
                err = err.format(i, len(disc_out_shape))
                raise ValueError(err)
            disc_in_shape = disc.input_shape
            assert disc_out_shape[-1] >= 2, 'Discriminator should have 2 or more output classes.'
            final_layer = disc.output._keras_history[0]
            if getattr(final_layer, 'activation') != softmax:
                err = ('Output of discriminator at index' + 
                '{} does not seem to be a probability distribution.'.format(i))
                warnings.warn(err)
            try:
                self._check_shape_compatibility(morpher_out_shape, disc_in_shape)
            except AssertionError as e:
                err = 'Discriminator at index {} not compatible with morpher. '
                err = err.format(i)
                err += str(e)
                raise ValueError(e)

    def _build(self):
        disc_trainers = []
        self.morpher.trainable = False
        for disc in self.discriminators:
            disc.trainable = True
            inp = Input(batch_shape=self.morpher.input_shape)
            morphed = self.morpher(inp)
            disc_out = disc(morphed)
            trainer = Model(inp, disc_out)
            loss = disc.loss
            opt = disc.optimizer
            met = disc.metrics
            trainer.compile(loss=loss, optimizer=opt, metrics=met)
            disc_trainers.append(trainer)
        self.disc_trainers = disc_trainers
        inp = Input(batch_shape=self.morpher.input_shape)
        self.morpher.trainable = True
        morphed = self.morpher(inp)
        disc_outs = []
        for disc in self.discriminators:
            disc.trainable = False
            disc_out = disc(morphed)
            disc_outs.append(disc_out)
        def maxf(x):
            am = K.max(x, -1)
            am = K.expand_dims(am, -1)
            return am
        max_layer = Lambda(maxf, output_shape=lambda s: s[:-1] + (1,))
        maxes = [max_layer(disc_out) for disc_out in disc_outs]
        def gate(x, n):
            return 1. - x
        gates = []
        for i, mx in enumerate(maxes):
            n = self.discriminators[i].output_shape[-1]
            gate_layer = Lambda(gate, arguments={'n': n})
            gates.append(gate_layer(mx))
        if len(gates) > 1:
            prod = multiply(gates)
        else:
            prod = gates[0]
        morpher_trainer = Model(inp, prod)
        morpher_trainer.compile(loss='mse', optimizer='adam', metrics=['acc'])
        self.morpher_trainer = morpher_trainer
        for disc in self.discriminators:
            disc.trainable = True
        task_input = morphed
        out = self.task(task_input)
        model = Model(inp, out)
        loss = self.task.loss
        opt = self.task.optimizer
        met = self.task.metrics
        model.compile(loss=loss, optimizer=opt, metrics=met)
        self.inference_model = model

    def train_discriminators_on_batch(self, x, labels):
        if isinstance(labels[0], np.ndarray):
            labels = [labels]
        assert len(labels) == len(self.discriminators)
        morphed = self.morpher.predict(x)
        for label, disc in zip(labels, self.disc_trainers):
            disc.train_on_batch(morphed, label)

    def train_morpher_and_task_on_batch(self, x, y):
        self.inference_model.train_on_batch(x, y)

    def train_morpher_on_batch(self, x):
        self.morpher_trainer.train_on_batch(x, np.zeros((x.shape[0], 1)))

    def train_on_batch(self, x, y, labels):
        if isinstance(labels[0], np.ndarray):
            labels = [labels]
        self.train_morpher_on_batch(x)
        self.train_discriminators_on_batch(x, labels)
        self.train_morpher_and_task_on_batch(x, y)


    def fit(self, x, y, labels, batch_size=None, epochs=1, validation_split=None):
        if batch_size is None:
            batch_size = 32
        num_steps = float(len(x)) / batch_size
        num_steps = math.ceil(num_steps)
        if validation_split is None:
            x_train = x
            y_train = y
            labels_train = labels
        else:
            num_validation = int(len(x) * validation_split)
            if isinstance(x, list):
                x_train = [arr[:-num_validation] for arr in x]
            else:
                x_train = x[:-num_validation]
            if isinstance(y, list):
                y_train = [arr[:-num_validation] for arr in y]
            else:
                y_train = y[:-num_validation]
            labels_train = [l[:-num_validation] for l in labels]
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch + 1))
            pbar = Progbar(len(x_train))
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i: i + batch_size]
                y_batch = y_train[i: i + batch_size]
                labels_batch = [l[i: i + batch_size] for l in labels_train]
                self.train_on_batch(x_batch, y_batch, labels_batch)
                pbar.add(len(x_batch))
        if validation_split is not None:
            x_test = x[-num_validation:]
            y_test = y[-num_validation:]
            labels_test = [l[-num_validation:] for l in labels]
            task_result = self.inference_model.evaluate(x_test, y_test)
            bias = get_bias(self.morpher.predict(x_test), labels_test)
            return task_result, bias

    def save(self, file):
        if isinstance(file, h5py.Group):
            must_close = False
        else:
            must_close = True
            file = h5py.File(file, 'w')
        task_group = file.create_group['task']
        self.task.save(task_group)
        morpher_group = file.create_group['morhper']
        self.morpher.save(morpher_group)
        disc_names = ['discriminator_' + str(i) for i in range(len(self.discriminators))]
        file.attrs['discriminator_names'] = disc_names
        for disc, disc_name in zip(self.discriminators, disc_names):
            disc_group = file.create_group(disc_name)
            disc.save(disc_group)
        file.flush()
        if must_close:
            file.close()

    @classmethod
    def load(cls, file, custom_objects={}):
        if isinstance(file, str):
            file = h5py.File(file, 'r')
        task_group = file['task']
        task = load_model(task_group, custom_objects)
        morpher_group = file['morhper']
        morpher = load_model(morpher_group, custom_objects)
        disc_names = file.attrs['discriminator_names']
        discriminators = []
        for disc_name in disc_names:
            disc_group = file[disc_name]
            disc = load_model(disc_group, custom_objects)
            discriminators.append(disc)
        return cls(task, morpher, discriminators)

    def get_weights(self):
        task_w = self.task.get_weights()
        morpher_w = self.morpher.get_weights()
        disc_w = []
        for disc in self.discriminators:
            disc_w += disc.get_weights()
        return task_w + morpher_w + disc_w

    def set_weights(self, weights):
        num_task_weights =len(self.task.weights)
        task_weights = weights[:num_task_weights]
        weights = weights[num_task_weights:]
        self.task.set_weights(task_weights)
        num_morpher_weights = len(self.morpher.weights)
        morpher_weights = weights[:num_morpher_weights]
        weights = weights[num_morpher_weights:]
        self.morpher.set_weights(morpher_weights)
        for disc in self.discriminators:
            num_disc_weights = len(disc.weights)
            disc_weights = weights[:num_disc_weights]
            weights = weights[num_disc_weights:]
            disc.set_weights(disc_weights)

    def __getstate__(self):
        task_state = pickle_model(self.task)
        morpher_state = pickle_model(self.morpher)
        disc_state = [pickle_model(disc) for disc in self.discriminators]
        return {'task': task_state, 
                'morpher': morpher_state,
                'discriminators': disc_state}

    def __setstate__(self, state):
        task_state = state['task']
        morpher_state = state['morphers']
        disc_state = state['discriminators']
        task = unpickle_model(task_state)
        morpher = unpickle_model(morpher_state)
        discriminators = [unpickle_model(disc) for disc in disc_state]
        unb = Unbias(task, morpher, discriminators)
        self.__dict__.update(unb.__dict__)


def get_bias(inputs, labels, model=None):
    input_dim = inputs.shape[-1]

    biases = []
    for disc_vecs in labels:
        num_categories = disc_vecs.shape[-1]
        if model is None:
            model = Sequential()
            model.add(Dense(input_dim, input_dim=input_dim))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(num_categories))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        split = 0.9
        num_train = int(len(inputs) * split)
        model.fit(inputs[:num_train], disc_vecs[:num_train], epochs=100)
        acc = model.evaluate(inputs[num_train:], disc_vecs[num_train:])[1]
        bias = acc - 1. / num_categories
        if bias < 0:
            bias = 0.
        biases.append(bias)

    return biases

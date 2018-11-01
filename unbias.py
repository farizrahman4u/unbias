from keras.layers import Input, Lambda
from keras.layers import multiply, add
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model, Sequential
import keras.backend as K
from keras.activations import softmax
from keras.utils import Progbar
import warnings
import numpy as np
import math


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
        for disc in self.discriminators:
            inp = Input(batch_shape=self.morpher.input_shape)
            self.morpher.trainable = False
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
            x_train = x[:-num_validation]
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


def get_bias(inputs, labels):
    input_dim = inputs.shape[-1]

    biases = []
    for disc_vecs in labels:
        num_categories = disc_vecs.shape[-1]
        clf = Sequential()
        clf.add(Dense(input_dim, input_dim=input_dim))
        clf.add(Activation('tanh'))
        clf.add(Dropout(0.2))
        clf.add(Dense(input_dim))
        clf.add(Activation('tanh'))
        clf.add(Dropout(0.2))
        clf.add(Dense(input_dim))
        clf.add(Activation('tanh'))
        clf.add(Dropout(0.2))
        clf.add(Dense(num_categories))
        clf.add(Activation('softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        split = 0.9
        num_train = int(len(inputs) * split)
        clf.fit(inputs[:num_train], disc_vecs[:num_train], epochs=100)
        acc = clf.evaluate(inputs[num_train:], disc_vecs[num_train:])[1]
        bias = acc - 1. / num_categories
        if bias < 0:
            bias = 0.
        biases.append(bias)

    return biases

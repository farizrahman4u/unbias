from keras.layers import Input, Lambda
from keras.layers import multiply
from keras.layers import Activation
from keras.models import Model
import keras.backend as K
from keras.activations import softmax
import warnings



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
        def bias(x, n):
            return (n * x - 1.) / (n - 1.)
        def gate(x, n):
            return 1. - bias(x, n)
        gates = []
        for i, mx in enumerate(maxes):
            n = self.discriminators[i].output_shape[-1]
            gate_layer = Lambda(gate, arguments={'n': n})
            gates.append(gate_layer(mx))
        prod = multiply(gates + [morphed])
        switch = Lambda(lambda x: K.in_train_phase(x[0], x[1]), output_shape=lambda s: s[0])
        task_input = switch([prod, morphed])
        out = self.task(task_input)
        model = Model(inp, out)
        loss = self.task.loss
        opt = self.task.optimizer
        met = self.task.metrics
        model.compile(loss=loss, optimizer=opt, metrics=met)
        self.combined_model = model
        inp = Input(batch_shape=self.morpher.input_shape)
        morphed = self.morpher(inp)
        out = self.task(morphed)
        self.inference_model = Model(inp, out)

    def train_discriminators_on_batch(self, x, *labels):
        assert type(labels) in (list, tuple)
        assert len(labels) == len(self.discriminators)
        morphed = self.morpher.predict(x)
        for label, disc in zip(labels, self.disc_trainers):
            disc.train_on_batch(morphed, label)

    def train_morpher_and_task_on_batch(self, x, y):
        self.combined_model.train_on_batch(x, y)

    def train_on_batch(self, x, y, *labels):
        self.train_discriminators_on_batch(x, *labels)
        self.train_morpher_and_task_on_batch(x, y)

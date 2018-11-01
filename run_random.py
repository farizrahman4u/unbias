from random_data import get_data
from unbias import get_bias
from unbias import Unbias
from keras.layers import *
from keras.models import *

inputs, outputs, labels = get_data(3200, 100, 'classification', 2, 2, epochs=100)

bias = get_bias(inputs, labels)



def get_morpher():
    model = Sequential()
    model.add(Dense(100, input_dim=100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    return model


def get_discriminator():
    model = Sequential()
    model.add(Dense(100, input_dim=100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_task():
    model = Sequential()
    model.add(Dense(100, input_dim=100))
    #model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


task = get_task()
morpher = get_morpher()
discriminator = get_discriminator()

unbias = Unbias(task, morpher, discriminator)

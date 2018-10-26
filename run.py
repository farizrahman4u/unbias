from boston import get_data
from unbias import Unbias
import numpy as np
from keras.layers import *
from keras.models import Sequential


np.random.seed(1337)


print('Loading data...')

train_data, test_data = get_data()

x_train, y_train = train_data
x_test, y_test = test_data

print('Data loaded.')
print('X train: {}'.format(x_train.shape))
print('Y train: {}'.format(y_train.shape))
print('X test: {}'.format(x_test.shape))
print('Y test: {}'.format(y_test.shape))

# Normalize
x_mx = x_train.max(axis=0)

x_train /= x_mx
x_test /= x_mx

blk_index = 11
blk_median = np.median(x_train[:, blk_index])

blk_labels_train = np.cast['float32'](x_train[:, blk_index] > blk_median)
blk_labels_test = np.cast['float32'](x_test[:, blk_index] > blk_median)

x_no_blk_train = np.concatenate([x_train[:, :blk_index], x_train[:, blk_index + 1:]], 1)
x_no_blk_test = np.concatenate([x_test[:, :blk_index], x_test[:, blk_index + 1:]], 1)


def blk_predictor(blk_included):
    if blk_included:
        input_dim = 13
    else:
        input_dim = 12
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



model_blk_included = blk_predictor(True)
model_blk_included.fit(x_train, blk_labels_train, epochs=200, batch_size=16)
loss1, acc1 = model_blk_included.evaluate(x_test, blk_labels_test)

model_blk_excluded = blk_predictor(False)
model_blk_excluded.fit(x_no_blk_train, blk_labels_train, epochs=200, batch_size=16)
loss2, acc2 = model_blk_excluded.evaluate(x_no_blk_test, blk_labels_test)

print("MLP was able to predict blk label with {} % accuracy when blk was provided explicitly.".format(acc1 * 100))
print("When blk was excluded from the input, MLP was able to predict blk label with {} % accuracy.".format(acc2 * 100))


assert acc2 <= acc1, 'You messed up.'


def get_morpher():
    model = Sequential()
    model.add(Dense(100, input_dim=13))
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
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


unbias = Unbias(get_task(), get_morpher(), get_discriminator())

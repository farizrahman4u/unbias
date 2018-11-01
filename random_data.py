import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Activation, Input, Dropout


def get_data(num_samples, input_dim, task_type='classification', output_dim=2, num_labels=2, **fit_args):
    '''
    # Arguments

    num_samples: (int) Number of samples
    input_dim: (int) Size of input vectors
    task_type: (str) Type of output. One of {'regression', 'classification'}
    output_dim: (int) Size of output vectors
    labels: (list<str> or list<list<str>>) Discriminatory labels. If more than one discriminator,
    pass a list of list of strings.
    '''

    input_vectors = np.random.uniform(-1, 1, (num_samples, input_dim))
    if task_type == 'regression':
        output_vectors = np.random.uniform(-1, 1, (num_samples, output_dim))
    elif task_type == 'classification':
        output_vectors = np.zeros((num_samples, output_dim))
        for vec in output_vectors:
            vec[np.random.randint(output_dim)] = 1.

    if isinstance(num_labels, int):
        num_labels = [num_labels]

    encoder_input_dim = input_dim + output_dim + sum(num_labels)

    encoder = Sequential()
    encoder.add(Dense(input_dim * 2, input_dim=encoder_input_dim))
    encoder.add(Activation('tanh'))
    encoder.add(Dense(input_dim * 2))
    encoder.add(Activation('tanh'))
    encoder.add(Dense(input_dim * 2))
    encoder.add(Activation('tanh'))
    encoder.add(Dense(input_dim))
    encoder.add(Activation('tanh'))

    task = Sequential()
    task.add(Dense(input_dim, input_dim=input_dim))
    task.add(Activation('tanh'))
    task.add(Dense(input_dim))
    task.add(Activation('tanh'))
    task.add(Dense(output_dim))
    if task_type == 'regression':
        task.add(Activation('tanh'))
    else:
        task.add(Activation('softmax'))

    disc_vecs = []
    for num_cat in num_labels:
        vecs = np.zeros((num_samples, num_cat))
        for vec in vecs:
            vec[np.random.randint(num_cat)] = 1.
        disc_vecs.append(vecs)
    disc_vec = np.concatenate(disc_vecs, axis=-1)

    encoder_input_vectors = np.concatenate([input_vectors, output_vectors, disc_vec], axis=-1)

    encoder_input = Input((encoder_input_dim,))
    encoded = encoder(encoder_input)
    task_output = task(encoded)

    model = Model(encoder_input, task_output)

    if task_type == 'regression':
        loss = 'mse'
    elif task_type == 'classification':
        loss = 'categorical_crossentropy'

    model.compile(loss=loss, optimizer='adam', metrics=['acc'])

    model.fit(encoder_input_vectors, output_vectors, **fit_args)

    encoder_input_vectors = np.concatenate([input_vectors, output_vectors, disc_vec], axis=-1)
    encoder_output_vectors = encoder.predict(encoder_input_vectors)

    return encoder_output_vectors, output_vectors, disc_vecs

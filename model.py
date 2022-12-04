from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import Sequential, regularizers, optimizers


def get_model(x):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=x))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.4))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.4))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dense(256, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import Sequential


def get_model(x):

    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(x.shape[1:])))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.4))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.4))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    return model
    #2
    # model = Sequential()
    #
    # model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(x.shape[1:])))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(.4))
    #
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x.shape[1:])))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(.4))
    #
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    #
    # model.add(Dense(3, activation='softmax'))

    #3 learning rate 0.0001 30 epochs

    # model = Sequential()
    #
    # model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(x.shape[1:])))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(.4))
    #
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x.shape[1:])))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(.4))
    #
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    #
    # model.add(Dense(3, activation='softmax'))
import pandas as pd
import os
from model import get_model
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from contextlib import redirect_stdout
import seaborn as sns
from sklearn.metrics import classification_report

PATH = r"C:\Users\Michal\PycharmProjects\Inz\data"
TITLES = [
    ("Accuracy", "Dokładność"),
    ("Loss", "Wartość funkcji straty"),
]
LABELS = [
    ("training set", "zestaw uczący"),
    ("validation set", "zestaw walidacyjny"),
    ("test set", "zestaw testowy")
]

XY_LABELS = [
    ("Value", "Wartość"),
    ("Epoch number", "Numer epoki")
]


def split_data(images, labels, mode='train_test', parts=5):
    out = {}
    data_ = list(zip(images, labels))
    np.random.shuffle(data_)

    if mode == 'train_test':
        out["train"] = tuple(map(lambda x: np.array(x), zip(*data_[:-len(data_)//3])))
        out["test"] = tuple(map(lambda x: np.array(x), zip(*data_[-len(data_)//3:])))

    elif mode == 'equal':
        for i in range(parts):
            out[i] = list(map(lambda x: np.array(x), zip(*data_[i*17485//parts:(i+1)*17485//parts])))

    return out


def plot_on_axis(ax, data, metric, labels, title, xy_labels):

    ax.plot(data[metric])
    ax.plot(data[f"val_{metric}"])
    ax.legend(labels)
    ax.set_title(title)

    ax.set_ylabel(xy_labels[0])
    ax.set_xlabel(xy_labels[1])

    return ax


def save_results(history, model, data_set, additional_info=''):
    sns.set()
    for i, language in enumerate(['ENG', 'PL']):
        # both
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0] = plot_on_axis(ax[0], history.history, "accuracy", [LABELS[0][i], LABELS[1][i]], TITLES[0][i], [XY_LABELS[0][i], ""])
        ax[1] = plot_on_axis(ax[1], history.history, "loss", [LABELS[0][i], LABELS[1][i]], TITLES[1][i], [XY_LABELS[0][i], XY_LABELS[1][i]])

        plt.savefig(f'./results/{language}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_both.png', dpi=600)

        # only accuracy
        fig, ax = plt.subplots(1, 1)
        _ = plot_on_axis(ax, history.history, "accuracy", [LABELS[0][i], LABELS[1][i]], TITLES[0][i], [XY_LABELS[0][i], XY_LABELS[1][i]])

        plt.savefig(f'./results/{language}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_accuracy.png', dpi=600)

        # only loss
        fig, ax = plt.subplots(1, 1)
        _ = plot_on_axis(ax, history.history, "loss", [LABELS[0][i], LABELS[1][i]], TITLES[1][i], [XY_LABELS[0][i], XY_LABELS[1][i]])

        plt.savefig(f'./results/{language}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_loss.png', dpi=600)

        with open(f'./results/{language}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.txt', 'w') as output_file:
            with redirect_stdout(output_file):
                model.summary()

            pred = model.predict(data_set["test"][0], batch_size=32, verbose=1)
            predicted = np.argmax(pred, axis=1)
            report = classification_report(data_set["test"][1], predicted)
            output_file.write(
                f'\n Test set results: {model.evaluate(data_set["test"][0], data_set["test"][1], return_dict=True)} \n'
                f'{report} \n'
                f'{additional_info}')


def normalize(image):
    return (image + 140)/70


def read_data(filename):

    categories = os.listdir(r"C:\Users\Michal\PycharmProjects\Inz\data")
    label = [i for i, category in enumerate(categories) if category in filename]
    df = pd.read_csv(filename, sep=',', header=None)
    image = df.to_numpy()
    image = image.reshape((image.shape[0], image.shape[1], 1))
    image = normalize(image)
    images = [image]
    labels = [label]
    name = ["\\".join(filename.split("\\")[-3:-1])]

    return np.array(images), np.array(labels), np.array(name)


def stack_in_time(images, names, labels):
    stacked_images = []
    stacked_names = []
    stacked_labels = []
    for i in range(len(names)):
        if i - 2 >= 0 and np.all(names[i - 2:i + 1] == names[i]):
            stacked_images.append(np.concatenate(images[i - 2:i + 1], axis=2))
            stacked_names.append(names[i])
            stacked_labels.append(labels[i])

    return np.array(stacked_images), np.array(stacked_names), np.array(stacked_labels)


def read_files(path):
    start = time.time()

    files = []
    for r, _, f in os.walk(PATH):
        for file in f:
            files.append(os.path.join(r, file))

    with Pool(processes=cpu_count()) as pool:

        images_df, labels_df, names_df = zip(*pool.map(read_data, files))
        images_df = np.concatenate(images_df)
        labels_df = np.concatenate(labels_df)
        names_df = np.concatenate(names_df)

    end = time.time()

    return images_df, labels_df, names_df


def train_valid_test():
    """
    Performs learning process with data split into train, test and validation set

    """
    images_df, labels_df, names_df = read_files(PATH)
    print(f"Images read {len(images_df)}")
    images_df, names_df, labels_df = stack_in_time(images_df, names_df, labels_df)
    data_set = split_data(images_df, labels_df)
    print(f'Images after split: train = {len(data_set["train"][0])}, test = {len(data_set["test"][0])}')
    model = get_model(images_df)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # start training and display summary
    history = model.fit(data_set["train"][0], data_set["train"][1], batch_size=28, epochs=200, validation_split=0.3)
    save_results(history, model, data_set)

    plt.show()
    model.summary()


def cross_validation(k):
    """
    Performs learning process with k-fold validation
    """
    images_df, labels_df, names_df = read_files(PATH)
    print(f"Images read {len(images_df)}")
    images_df, names_df, labels_df = stack_in_time(images_df, names_df, labels_df)
    folded_data_set = split_data(images_df, labels_df, mode='equal', parts=k)

    for i in range(k):
        data_set = {"train": (np.concatenate([folded_data_set[part][0] for part in range(k) if part != i]),
                              np.concatenate([folded_data_set[part][1] for part in range(k) if part != i])),
                    "test": (folded_data_set[i][0],
                             folded_data_set[i][1])}
        # data_set["train"][0] = np.concatenate([folded_data_set[part][0] for part in range(k) if part != i])
        # data_set["train"][1] = np.concatenate([folded_data_set[part][1] for part in range(k) if part != i])
        # data_set["test"][0] = folded_data_set[i][0]
        # data_set["test"][1] = folded_data_set[i][1]
        model = get_model(images_df)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # start training and display summary
        history = model.fit(data_set["train"][0], data_set["train"][1], batch_size=10, epochs=100, validation_split=0.3)
        save_results(history, model, data_set, additional_info=f'{k}-fold cross_validation: test set number = {i}')

        plt.show()
        model.summary()


def main():
    # cross_validation(5)
    train_valid_test()


if __name__ == '__main__':
    main()

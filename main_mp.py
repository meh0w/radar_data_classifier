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
sns.set()


def split_data(images, labels):
    out = {}
    data_ = list(zip(images, labels))
    np.random.shuffle(data_)

    out["train"] = tuple(map(lambda x: np.array(x), zip(*data_[:-len(data_)//3])))
    out["test"] = tuple(map(lambda x: np.array(x), zip(*data_[-len(data_)//3:])))

    return out


def plot_on_axis(ax, data, metric, labels, title, xy_labels):

    ax.plot(data[metric])
    ax.plot(data[f"val_{metric}"])
    ax.legend(labels)
    ax.set_title(title)

    ax.set_ylabel(xy_labels[0])
    ax.set_xlabel(xy_labels[1])

    return ax

def save_results(history, model, data_set):

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
            output_file.write(
                f'\n Test set results: {model.evaluate(data_set["test"][0], data_set["test"][1], return_dict=True)}')


def read_data(filename):

    images = []
    labels = []
    categories = os.listdir(r"C:\Users\Michal\PycharmProjects\Inz\data")
    label = [i for i, category in enumerate(categories) if category in filename]
    df = pd.read_csv(filename, sep=',', header=None)
    image = df.to_numpy()
    image = image.reshape((image.shape[0], image.shape[1], 1))/200
    images.append(image)
    labels.append(label)

    return np.array(images), np.array(labels)


def main():
    start = time.time()

    files = []
    for r, _, f in os.walk(PATH):
        for file in f:
            files.append(os.path.join(r, file))

    with Pool(processes=cpu_count()) as pool:

        images_df, labels_df = zip(*pool.map(read_data, files))
        images_df = np.concatenate(images_df)
        labels_df = np.concatenate(labels_df)

    end = time.time()
    print(f"Images read {len(images_df)}")
    data_set = split_data(images_df, labels_df)
    print(f'Images after split: train = {len(data_set["train"][0])}, test = {len(data_set["test"][0])}')
    model = get_model(images_df)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # start training and display summary
    history = model.fit(data_set["train"][0], data_set["train"][1], batch_size=10, epochs=100, validation_split=0.3)
    save_results(history, model, data_set)

    plt.show()
    model.summary()


if __name__ == '__main__':
    main()

import pandas as pd
import os
from model import get_model
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import redirect_stdout
import seaborn as sns
from sklearn.metrics import classification_report


def normalize(image):
    return (image - np.min(image))/(np.max(image) - np.min(image))


class Classifier:

    def __init__(self, data_path, results_path):

        self.data_path = data_path
        self.results_path = results_path
        self.images = None
        self.labels = None
        self.names = None
        self.data = {}
        self.model = None
        self.history = None

    def get_model(self, input_shape=(11, 61, 3)):
        self.model = get_model(input_shape)

    def normalize(self):
        self.images = (self.images - np.min(self.images))/(np.max(self.images) - np.min(self.images))

    def read_files(self):

        files = []
        for r, _, f in os.walk(self.data_path):
            for file in f:
                files.append(os.path.join(r, file))

        with Pool(processes=cpu_count()) as pool:

            images, labels, names = zip(*pool.map(self.read_data, files))
            self.images = np.concatenate(images)
            self.labels = np.concatenate(labels)
            self.names = np.concatenate(names)
        self.normalize()

    def read_data(self, filename):

        categories = os.listdir(self.data_path)
        label = [i for i, category in enumerate(categories) if category in filename]
        df = pd.read_csv(filename, sep=',', header=None)
        image = df.to_numpy()
        image = image.reshape((image.shape[0], image.shape[1], 1))
        images = [image]
        labels = [label]
        name = ["\\".join(filename.split("\\")[-3:-1])]

        return np.array(images), np.array(labels), np.array(name)

    def stack_in_time(self):
        stacked_images = []
        stacked_names = []
        stacked_labels = []
        for i in range(len(self.names)):
            if i - 2 >= 0 and np.all(self.names[i - 2:i + 1] == self.names[i]):
                stacked_images.append(np.concatenate(self.images[i - 2:i + 1], axis=2))
                stacked_names.append(self.names[i])
                stacked_labels.append(self.labels[i])

        self.images, self.labels, self.names = \
            np.array(stacked_images), np.array(stacked_labels), np.array(stacked_names)

    def split_data(self, mode='train_test', parts=5):
        out = {}
        data = list(zip(self.images, self.labels))
        np.random.shuffle(data)

        if mode == 'train_test':
            out["train"] = tuple(map(lambda x: np.array(x), zip(*data[:-len(data) // 3])))
            out["test"] = tuple(map(lambda x: np.array(x), zip(*data[-len(data) // 3:])))

        elif mode == 'equal':
            for i in range(parts):
                out[i] = list(map(lambda x: np.array(x), zip(*data[i * 17485 // parts:(i + 1) * 17485 // parts])))

        return out

    def train_valid_test(self):
        """
        Performs learning process with data split into train, test and validation set

        """
        data_set = self.split_data()
        print(f'Images after split: train = {len(data_set["train"][0])}, test = {len(data_set["test"][0])}')

        # start training and display summary
        self.history = self.model.fit(data_set["train"][0], data_set["train"][1], batch_size=28, epochs=100,
                                      validation_split=0.1)

        if os.path.isdir(self.results_path):
            self.save_results(data_set)

        self.model.summary()
        plt.show()

    def cross_validation(self, k):
        """
        Performs learning process with k-fold validation
        """
        folded_data_set = self.split_data(mode='equal', parts=k)

        for i in range(k):
            data_set = {"train": (np.concatenate([folded_data_set[part][0] for part in range(k) if part != i]),
                                  np.concatenate([folded_data_set[part][1] for part in range(k) if part != i])),
                        "test": (folded_data_set[i][0],
                                 folded_data_set[i][1])}

            # start training and display summary
            self.history = self.model.fit(data_set["train"][0], data_set["train"][1], batch_size=25, epochs=100,
                                          validation_split=0.1)
            if os.path.isdir(self.results_path):
                self.save_results(data_set, additional_info=f'{k}-fold cross_validation: test set number = {i}')

            self.model.summary()
            plt.show()

    def plot_on_axis(self, ax, metric, title):

        ax.plot(self.history.history[metric])
        ax.plot(self.history.history[f"val_{metric}"])
        ax.legend(['zestaw uczący', 'zestaw walidacyjny', 'zestaw testowy'])
        ax.set_title(title)

        ax.set_ylabel('Wartość')
        ax.set_xlabel('Numer epoki')

        return ax

    def save_results(self, data_set, additional_info=''):
        sns.set()
        # both
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0] = self.plot_on_axis(ax[0], 'accuracy', 'Dokładność')
        ax[1] = self.plot_on_axis(ax[1], 'loss', 'Wartość funkcji straty')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_both.png', dpi=600)

        # only accuracy
        fig, ax = plt.subplots(1, 1)
        _ = self.plot_on_axis(ax, 'accuracy', 'Dokładność')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_accuracy.png', dpi=600)

        # only loss
        fig, ax = plt.subplots(1, 1)
        _ = self.plot_on_axis(ax, 'loss', 'Wartość funkcji straty')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_loss.png', dpi=600)

        with open(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.txt', 'w') as output_file:
            with redirect_stdout(output_file):
                self.model.summary()

            pred = self.model.predict(data_set["test"][0], batch_size=32, verbose=1)
            predicted = np.argmax(pred, axis=1)
            report = classification_report(data_set["test"][1], predicted)
            output_file.write(
                f'\n Test set results: {self.model.evaluate(data_set["test"][0], data_set["test"][1], return_dict=True)} \n'
                f'{report} \n'
                f'{additional_info}')
        self.save_weights()

    def save_weights(self):
        self.model.save_weights(f'{self.results_path}/weights_{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, paths):
        files = []
        for file_path in paths:
            df = pd.read_csv(file_path, sep=',', header=None)
            image = df.to_numpy()
            image = image.reshape((image.shape[0], image.shape[1], 1))
            image = normalize(image)
            files.append(image)
        input_file = np.concatenate(np.array(files), axis=2)
        return self.model.predict(input_file.reshape((-1,) + input_file.shape))

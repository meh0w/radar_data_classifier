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
        self.MIN, self.MAX = -200, -20

        self.macro_avg = {'precision': [], 'recall': [], 'f1-score': [], 'accuracy': []}
        self.weighted_avg = {'precision': [], 'recall': [], 'f1-score': [], 'accuracy': []}

    def normalize(self, data):
        """
        Normalizes the provided data
        :param data: array of data
        :return: normalized data
        """
        return (data - self.MIN)/(self.MAX - self.MIN)

    def get_model(self, input_shape=(11, 61, 3)):
        """
        Gets deep CNN model for classifier
        :param input_shape: shape of the models input
        :return:
        """
        self.model = get_model(input_shape)

    def read_files(self):
        """
        Reads files from folder specified by the data_path attribute
        """
        files = []
        for r, _, f in os.walk(self.data_path):
            for file in f:
                files.append(os.path.join(r, file))

        with Pool(processes=cpu_count()) as pool:

            images, labels, names = zip(*pool.map(self.read_data, files))
            self.images = np.concatenate(images)
            self.labels = np.concatenate(labels)
            self.names = np.concatenate(names)

        self.images = self.normalize(self.images)

    def read_data(self, filename):
        """
        Reads data from csv file
        :param filename: path to a csv file
        :return: Array with data, Array with label, Array with name
        """
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
        """
        Concatenates loaded data
        """
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
        """
        Splits data
        :param mode: describes how to split the data
        :param parts: number of parts to split data into
        :return: split data
        """
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
        print(f'Number of data after split: train = {len(data_set["train"][0])}, test = {len(data_set["test"][0])}')

        # start training and display summary
        self.history = self.model.fit(data_set["train"][0], data_set["train"][1], batch_size=28, epochs=100,
                                      validation_split=0.3)

        if os.path.isdir(self.results_path):
            self.save_results(data_set)
            with open(f'{self.results_path}/{datetime.now().strftime("report_%d_%m_%Y__%H_%M_%S")}.txt',
                      'w') as output_file:
                formatted_weighted_dict = str(self.weighted_avg).replace("],", "],\n")
                formatted_macro_dict = str(self.macro_avg).replace("],", "],\n")
                output_file.write(f'Weighted: \n'
                                  f'{formatted_weighted_dict}\n'
                                  f'Mean {[np.mean(metric) for metric in self.weighted_avg.values()]}\n'
                                  f'Macro:\n'
                                  f'{formatted_macro_dict}\n'
                                  f'Mean {[np.mean(metric) for metric in self.macro_avg.values()]}\n')

        self.model.summary()
        plt.show()

    def cross_validation(self, k):
        """
        Performs learning process with k-fold validation
        """
        folded_data_set = self.split_data(mode='equal', parts=k)

        for i in range(k):
            print(f"Iteration {i}/{k-1}")
            self.get_model()
            data_set = {"train": (np.concatenate([folded_data_set[part][0] for part in range(k) if part != i]),
                                  np.concatenate([folded_data_set[part][1] for part in range(k) if part != i])),
                        "test": (folded_data_set[i][0],
                                 folded_data_set[i][1])}

            # start training and display summary
            self.history = self.model.fit(data_set["train"][0], data_set["train"][1], batch_size=28, epochs=100,
                                          validation_split=0.3)
            if os.path.isdir(self.results_path):
                self.save_results(data_set, additional_info=f'{k}-fold cross_validation: test set number = {i}')

            self.model.summary()

        with open(f'{self.results_path}/{datetime.now().strftime("report_%d_%m_%Y__%H_%M_%S")}.txt',
                  'w') as output_file:
            formatted_weighted_dict = str(self.weighted_avg).replace("],", "],\n")
            formatted_macro_dict = str(self.macro_avg).replace("],", "],\n")
            output_file.write(f'Weighted: \n'
                              f'{formatted_weighted_dict}\n'
                              f'Mean {[np.mean(metric) for metric in self.weighted_avg.values()]}\n'
                              f'Macro:\n'
                              f'{formatted_macro_dict}\n'
                              f'Mean {[np.mean(metric) for metric in self.macro_avg.values()]}\n')

    def plot_on_axis(self, ax, metric, title):
        """
        Plots metric on provided axis
        :param ax: axis to plot data on
        :param metric: selected metric
        :param title: the title of the plot
        :return: ax with plotted metric
        """
        ax.plot(self.history.history[metric])
        ax.plot(self.history.history[f"val_{metric}"])
        ax.legend(['zestaw uczący', 'zestaw walidacyjny', 'zestaw testowy'])
        ax.set_title(title)

        ax.set_ylabel('Wartość')
        ax.set_xlabel('Numer epoki')

        return ax

    def save_results(self, data_set, additional_info=''):
        """
        Saves results to files
        :param data_set: loaded data set
        :param additional_info: additional info to write into a file
        """
        sns.set()
        # both
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0] = self.plot_on_axis(ax[0], 'accuracy', 'Dokładność')
        ax[1] = self.plot_on_axis(ax[1], 'loss', 'Wartość funkcji straty')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_both.svg')

        # only accuracy
        fig, ax = plt.subplots(1, 1)
        _ = self.plot_on_axis(ax, 'accuracy', 'Dokładność')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_accuracy.svg')

        # only loss
        fig, ax = plt.subplots(1, 1)
        _ = self.plot_on_axis(ax, 'loss', 'Wartość funkcji straty')

        plt.savefig(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_loss.svg')

        with open(f'{self.results_path}/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.txt', 'w') as output_file:
            with redirect_stdout(output_file):
                self.model.summary()

            pred = self.model.predict(data_set["test"][0], verbose=1)
            predicted = np.argmax(pred, axis=1)
            evaluation = self.model.evaluate(data_set["test"][0], data_set["test"][1], return_dict=True)
            output_file.write(
                f'\n Test set results: {evaluation} \n'
                f'{additional_info}')

        report = classification_report(data_set["test"][1], predicted, output_dict=True)
        for metric in list(report['macro avg'].keys())[:-1]:
            self.macro_avg[metric].append(report['macro avg'][metric])
            self.weighted_avg[metric].append(report['weighted avg'][metric])

        self.macro_avg['accuracy'].append(evaluation['accuracy'])
        self.weighted_avg['accuracy'].append(evaluation['accuracy'])
        self.save_weights()

    def save_weights(self):
        """
        Saves models weights
        """
        self.model.save_weights(f'{self.results_path}/weights_{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.h5')

    def load_weights(self, path):
        """
        Loads models weights
        """
        self.model.load_weights(path)

    def predict(self, paths):
        """
        Classifies provided files
        :param paths: list of paths to files that form a single input
        :return: model prediction
        """
        files = []
        for file_path in paths:
            df = pd.read_csv(file_path, sep=',', header=None)
            image = df.to_numpy()
            image = image.reshape((image.shape[0], image.shape[1], 1))
            image = self.normalize(image)
            files.append(image)
        input_file = np.concatenate(np.array(files), axis=2)
        return self.model.predict(input_file.reshape((-1,) + input_file.shape))

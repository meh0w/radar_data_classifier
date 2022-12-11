from radar_classifier import Classifier
import argparse
import os
import numpy as np


def run(config):
    """
    Handles communication between user and classifier
    :param config: configuration of a run
    """
    radar = Classifier(data_path=config['data_path'],
                       results_path=config['results_path'])

    names = {0: 'Car', 1: 'Drone', 2: 'Human'}
    required_files = 3
    data_ready = False
    if config['data_path'] is not None and os.path.isdir(config['data_path']):
        print("Reading data...")
        radar.read_files()
        if config['input_concat']:
            print("Concatenating data...")
            radar.stack_in_time()
        else:
            required_files = 1
        print("Data is ready")
        data_ready = True
    elif config['data_path'] is not None:
        print("Provided data path is incorrect, data not loaded")

    radar.get_model((11, 61, required_files))

    if config['weights'] is not None and os.path.isfile(config['data_path']) and config['weights'][-3:] == '.h5':
        radar.load_weights(config['weights'])
        print("Loaded custom weights")
    elif config['weights'] is not None:
        print("Provided weights path is incorrect, weights not loaded")

    if data_ready:
        if config['train'] == 'tvts':
            print(f"Beginning Train Valid Test evaluation")
            radar.train_valid_test()
            print(f"Train Valid Test evaluation ended")
        elif config['train'] == 'k-fold':
            print(f"Beginning {config['iterations']}-fold validation")
            radar.cross_validation(config['iterations'])
            print(f"{config['iterations']}-fold validation ended")

    elif config['train'] == 'tvts' or config['train'] == 'k-fold':
        print("Skipping training due to missing data...")

    end = False

    while not end:
        print("Enter path to a data you would like to classify or 'quit' to exit: ")
        paths = []
        i = 0
        while i < required_files:
            path = input()
            if path == 'quit':
                end = True
                break
            if os.path.isfile(path) and path[-4:] == '.csv':
                paths.append(path)
                i += 1
                print(f'Provided {i}/{required_files} files')

            else:
                print("Wrong path")
        if not end:
            print(f'Provided {required_files}/{required_files} files')
            result = radar.predict(paths)
            print(result)
            print(f'The provided data contains a {names[np.argmax(result)]}')
        else:
            print('Exiting...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Radar data classifier',
                                     description='')
    parser.add_argument('-d', '--data_path', type=str, action='store', required=False)
    parser.add_argument('-r', '--results_path', type=str, action='store', required=False)
    parser.add_argument('-w', '--weights', type=str, required=False)
    parser.add_argument('-c', '--input_concat', action='store_true', required=False)
    parser.add_argument('-t', '--train', type=str, choices=['k-fold', 'tvts'], required=False)
    parser.add_argument('-i', '--iterations', type=int, required=False)
    args = parser.parse_args()
    run(vars(args))

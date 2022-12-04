from radar_classifier import Classifier


def run():
    radar = Classifier(data_path=r"C:\Users\Michal\PycharmProjects\Inz\data",
                       results_path=r"C:\Users\Michal\PycharmProjects\Inz\results\classy")

    radar.read_files()
    radar.stack_in_time()
    radar.get_model()
    # radar.load_weights(r'C:\Users\Michal\PycharmProjects\Inz\results\classy\weights_02_12_2022__20_34_21.h5')
    radar.train_valid_test()
    result = radar.predict([r'C:\Users\Michal\PycharmProjects\Inz\data\Cars\13-13\001.csv',
                            r'C:\Users\Michal\PycharmProjects\Inz\data\Cars\13-13\002.csv',
                            r'C:\Users\Michal\PycharmProjects\Inz\data\Cars\13-13\003.csv'])
    print(result)

if __name__ == '__main__':
    run()

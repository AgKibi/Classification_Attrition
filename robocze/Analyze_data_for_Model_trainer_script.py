from Model_trainer_script import ModelTrainer
import argparse # analizowanie wiersza poleceń (command-line) i pobieranie argumentów przekazywanych do programu w trakcie uruchamiania //
# parsing the command-line and retrieving the arguments passed to the program at startup
import pandas as pd

def main():
    # Parsowanie argumentów // parsing the command-line and retrieving the arguments passed to the program at startup
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train', type=str)
    parser.add_argument('--X_test', type=str)
    parser.add_argument('--y_train', type=str)
    parser.add_argument('--y_test', type=str)
    args = parser.parse_args()

    # Odczyt danych z plików // Reading data from files
    X_train = pd.read_csv(args.X_train)
    X_test = pd.read_csv(args.X_test)
    y_train = pd.read_csv(args.y_train).values.ravel()  # ravel -  spłaszczenia etykiet w jednowymiarową tablicę // flatten labels into a one-dimensional array
    y_test = pd.read_csv(args.y_test).values.ravel()  


    try:
        param_grid = {}
        classifiers = ['LogisticRegression','KNN', 'SVC', 'GaussianNB', 'DecisionTree', 'AdaBoost', 'XGBoost', 'LGBM', 'CatBoost']

        for classifier_name in classifiers:
            trainer = ModelTrainer()
            trainer.build_pipeline(classifier_name, feature_selection='KBest', X_train=X_train, y_train=y_train)
            trainer.build_grid_search(param_grid)
            trainer.train(X_train, y_train)

            train_predictions = trainer.predict_train(X_train) 
            train_metrics = trainer.calculate_metrics(y_train, train_predictions)

            test_predictions = trainer.predict(X_test)  # Prediction on test data
            test_metrics = trainer.calculate_metrics(y_test, test_predictions)

            print(f'{classifier_name} metrics on training data:')
            trainer.print_results(train_metrics, classifier_name, X_train, y_train)  # Metryki dla zbioru treningowego // Metrics for training data

            print(f'{classifier_name} metrics on test data:')
            trainer.print_results(test_metrics, classifier_name)  # Metryki dla zbioru testowego // Metrics for test data


    except Exception as e:
        print(f'Error occurred: {str(e)}')

if __name__ == '__main__':
    main()

from data_preprocessing import preprocess_data
from ann import build_ann
from sklearn.metrics import confusion_matrix
import keras
import os.path
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def main():
    x_train, x_test, y_train, y_test = preprocess_data(dataset_path='Churn_Modelling.csv')
    fit_params = {'epochs': 20, 'batch_size': 10}

    classifier = KerasClassifier(build_fn=build_ann)
    classifier.sk_params = fit_params

    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1, fit_params=fit_params)
    mean = accuracies.mean()
    variance = accuracies.std()
    print('K-fold cross validation mean:', mean)
    print('K-fold cross validation variance:', variance)

    # if os.path.isfile('trained_model.h5'):
    #     classifier = keras.models.load_model('trained_model.h5')
    # else:
    #     classifier = build_ann()
    #     classifier.fit(x_train, y_train, batch_size=10, epochs=40)
    #     classifier.save('trained_model.h5')

    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)

    y_predict = (y_predict > 0.5)
    cm = confusion_matrix(y_test, y_predict)
    print('Confusion matrix: ', cm)
    overall_accuracy = (cm[0][0] + cm[1][1])/sum(sum(cm))
    print('Overall accuracy: ', overall_accuracy)


if __name__ == "__main__":
    main()

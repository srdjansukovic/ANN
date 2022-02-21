from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


def build_ann():
    classifier = Sequential()
    classifier.add(Dense(input_dim=11, units=6, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))  # dropout regularization
    classifier.add(Dense(units=6, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', bias_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

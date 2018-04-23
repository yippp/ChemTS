from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.layers import Dropout
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket


def prepare_data(smiles, all_smile):
    all_smile_index = []
    for i in range(len(all_smile)):
        smile_index = []
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train = all_smile_index
    y_train = []
    for i in range(len(X_train)):
        x1 = X_train[i]
        x2 = x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train, y_train


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print "Saved model to disk"


if __name__ == "__main__":
    smile = zinc_data_with_bracket_original()
    valcabulary, all_smile = zinc_processed_with_bracket(smile)
    print "valcabulary:", valcabulary
    print "number of all smiles:",len(all_smile)
    X_train, y_train = prepare_data(valcabulary, all_smile)

    maxlen = 81

    X = sequence.pad_sequences(X_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)
    y = sequence.pad_sequences(y_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)

    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
    print y_train_one_hot.shape

    vocab_size = len(valcabulary)
    embed_size = len(valcabulary)

    N = X.shape[1]

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=len(valcabulary), input_length=N, mask_zero=False))
    model.add(GRU(output_dim=256, input_shape=(maxlen, 64), activation='tanh', return_sequences=True))
    # model.add(LSTM(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation='tanh', return_sequences=True))
    # model.add(LSTM(output_dim=1000, activation='sigmoid',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(embed_size, activation='softmax')))
    optimizer = Adam(lr=0.01)
    print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, y_train_one_hot, nb_epoch=100, batch_size=512, validation_split=0.1)
    save_model(model)

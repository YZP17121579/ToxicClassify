import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from textblob import TextBlob

from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNGRU, Embedding, Dropout, Activation, SpatialDropout1D, GlobalMaxPooling1D, Lambda
from keras.layers import merge, Bidirectional, Conv1D, GlobalAveragePooling1D, Layer, initializers, regularizers, constraints
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from DeepNet.deeptoxic.config import model_config, dataset_config
from DeepNet.deeptoxic.data_helper.data_loader import DataLoader
from DeepNet.deeptoxic.data_helper.data_transformer import DataTransformer

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

VOCAB_SIZE = model_config.VOCAB_SIZE
MAX_SEQUENCE_LENGTH = model_config.MAX_SEQUENCE_LENGTH
EMBEDDING_SIZE = model_config.EMBEDDING_SIZE
char_level = model_config.char_level

data_transformer = DataTransformer(max_num_words=VOCAB_SIZE, pad_seq=False,
                                   max_sequence_length=MAX_SEQUENCE_LENGTH, char_level=char_level, error_correct=False)
data_loader = DataLoader()

train_sequences, training_labels, test_sequences = data_transformer.prepare_data()

print("Loading data ...")
embeddings_index = data_loader.load_embedding(dataset_config.FASTTEXT_PATH)
embedding_matrix = data_transformer.build_embedding_matrix(embeddings_index)
print("Loaded")


# define TextBlob
def sent2pos(sentence):
    try:
        tag = TextBlob(sentence).tags
    except:
        print(sentence)
    tag = TextBlob(sentence).tags
    updated_sentence = " ".join([i[0] for i in tag])
    tagged = " ".join([i[1] for i in tag])
    return updated_sentence, tagged


word_index = data_transformer.tokenizer.word_index
inverse_word_index = {v: k for k, v in word_index.items()}


# Get training data after TextBlob
pos_updated_sentence = []
pos_comments = []
for text in train_sequences:
    text1 = " ".join([inverse_word_index[word] for word in text])
    if not isinstance(text1, str):
        print(text)
        print(text1)
    updated_sentence, text2 = sent2pos(text1)
    # pos_updated_sentence is new combination of words
    pos_updated_sentence.append(updated_sentence)
    # pos_comments is part of speech combination
    pos_comments.append(text2)
    assert len(updated_sentence.split(' ')) == len(text2.split(' ')), "T1 {} T2 {} ".format(len(text),
                                                                                            len(text2.split()))
# Get testing data after TextBlob
pos_test_updated_sentence = []
pos_test_comments = []
for text in test_sequences:
    text1 = " ".join([inverse_word_index[word] for word in text])
    updated_sentence, text2 = sent2pos(text1)
    pos_test_updated_sentence.append(updated_sentence)
    pos_test_comments.append(text2)
    assert len(updated_sentence.split(' ')) == len(text2.split(' ')), "T1 {} T2 {} ".format(len(text),
                                                                                            len(text2.split()))

# Tokenize the pos tag
pos_tokenizer = Tokenizer(num_words=50, filters='"#$%&()+,-./:;<=>@[\\]^_`{|}~\t\n')
pos_tokenizer.fit_on_texts(pos_comments + pos_test_comments)
sequences = pos_tokenizer.texts_to_sequences(pos_comments)
test_sequences = pos_tokenizer.texts_to_sequences(pos_test_comments)
pos_word_index = pos_tokenizer.word_index
print('Found %s unique tokens of part of speech' % len(pos_word_index))

# Padding sentence
pos_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
pos_test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', pos_test_data.shape)

# Get train\test sentence data
comments = pos_updated_sentence
test_comments = pos_test_updated_sentence
sequences = data_transformer.tokenizer.texts_to_sequences(comments)
test_sequences = data_transformer.tokenizer.texts_to_sequences(test_comments)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# STAMP = ""


def _train_model_by_logloss(model, batch_size, train_x, pos_train_x, train_y, val_x, pos_val_x, val_y, fold_id):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    # bst_model_path = STAMP + str(fold_id) + '.h5'
    # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    train_data = {'Onehot': train_x, 'POS': pos_train_x}
    val_data = {'Onehot': val_x, 'POS': pos_val_x}
    hist = model.fit(train_data, train_y,
                     validation_data=(val_data, val_y),
                     epochs=20, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping])
    bst_val_score = min(hist.history['val_loss'])
    predictions = model.predict(val_data)
    auc = roc_auc_score(val_y, predictions)
    print("In fold {}\tAUC Score: {}".format(fold_id + 1, auc))
    return model, bst_val_score, auc, predictions


def train_folds(X, pos_x, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    fold_predictions = []
    score = 0
    total_auc = 0
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])
        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        pos_train_x = np.concatenate([pos_x[:fold_start], pos_x[fold_end:]])
        pos_val_x = pos_x[fold_start:fold_end]
        print("In fold %d" % fold_id)
        model, bst_val_score, auc, fold_prediction = _train_model_by_logloss(get_model_func(), batch_size, train_x,
                                                                             pos_train_x, train_y, val_x, pos_val_x,
                                                                             val_y, fold_id)
        score += bst_val_score
        total_auc += auc
        fold_predictions.append(fold_prediction)
        models.append(model)
    return models, score / fold_count, total_auc / fold_count, fold_predictions


# Optimizer
adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=5, decay=1e-6)


# Attention Cell
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=False, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias

        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], ),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1], ),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b= None
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weight_input = x * a
        return K.sum(weight_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


# Keras CNN Model
def get_av_pos_cnn():

    filter_nums = 325
    drop_rate = 0.5

    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), name='Onehot')
    input_layer_2 = Input(shape=(MAX_SEQUENCE_LENGTH,), name='POS')

    embedding_layer = Embedding(VOCAB_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input_layer)

    embedding_layer2 = Embedding(50,
                                 30,
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 trainable=True)(input_layer_2)

    embedding_layer = concatenate([embedding_layer, embedding_layer2], axis=2)
    embedded_sequences = SpatialDropout1D(0.25)(embedding_layer)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    attn_0 = Attention(MAX_SEQUENCE_LENGTH)(conv_0)
    avg_0 = GlobalAveragePooling1D()(conv_0)
    maxpool_0 = GlobalMaxPooling1D()(conv_0)

    maxpool_1 = GlobalMaxPooling1D()(conv_1)
    attn_1 = Attention(MAX_SEQUENCE_LENGTH)(conv_1)
    avg_1 = GlobalAveragePooling1D()(conv_1)

    maxpool_2 = GlobalMaxPooling1D()(conv_2)
    attn_2 = Attention(MAX_SEQUENCE_LENGTH)(conv_2)
    avg_2 = GlobalAveragePooling1D()(conv_2)

    maxpool_3 = GlobalMaxPooling1D()(conv_3)
    attn_3 = Attention(MAX_SEQUENCE_LENGTH)(conv_3)
    avg_3 = GlobalAveragePooling1D()(conv_3)

    v0_col = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    v1_col = merge([attn_0, attn_1, attn_2, attn_3], mode='concat', concat_axis=1)
    v2_col = merge([avg_1, avg_2, avg_0, avg_3], mode='concat', concat_axis=1)
    merged_tensor = merge([v0_col, v1_col, v2_col], mode='concat', concat_axis=1)
    output = Dropout(0.7)(merged_tensor)
    output = Dense(units=144)(output)
    output = Activation('relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(units=6, activation='sigmoid')(output)

    model = Model(inputs=[input_layer, input_layer_2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return model


# Keras Attention RNN Model
def get_av_pos_rnn():
    recurrent_units = 64
    drop_rate = 0.35
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), name='Onehot')
    input_layer_2 = Input(shape=(MAX_SEQUENCE_LENGTH,), name='POS')

    embedding_layer = Embedding(VOCAB_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input_layer)

    embedding_layer2 = Embedding(50,
                                 30,
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 trainable=True)(input_layer_2)

    embedding_layer = concatenate([embedding_layer, embedding_layer2], axis=2)
    embedding_layer = SpatialDropout1D(drop_rate)(embedding_layer)

    r1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    r1 = SpatialDropout1D(drop_rate)(r1)
    # r2 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(r1)
    # r2 = SpatialDropout1D(drop_rate)(r2)
    # rrs = concatenate([r1 ,r2], axis=-1)

    last_1 = Lambda(lambda t: t[:, -1])(r1)
    # last_2 = Lambda(lambda t: t[:, -1])(r2)
    maxpool = GlobalMaxPooling1D()(r1)
    attn = Attention(MAX_SEQUENCE_LENGTH)(r1)
    average = GlobalAveragePooling1D()(r1)

    concatenated = concatenate([maxpool, last_1, attn, average, ], axis=1)
    x = Dropout(0.5)(concatenated)
    x = Dense(144, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[input_layer, input_layer_2], outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam_optimizer,
                  metrics=['accuracy'])
    return model


# Training
models, val_loss, avg_auc, fold_predictions = train_folds(data, pos_data, training_labels, 10, 256, get_av_pos_cnn)
print("Overall val-loss:", val_loss, "AUC", avg_auc)

# Compute average AUC of all folds
train_fold_preditcions = np.concatenate(fold_predictions, axis=0)
training_auc = roc_auc_score(training_labels, train_fold_preditcions)
print("Training AUC", training_auc)

path = 'Dataset/'
TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submit_path_prefix = "results/bgru/Fasttext-BGRU-" + str(MAX_SEQUENCE_LENGTH)

print("Predicting testing results...")
# Average testing results, using models of all folds
test_predicts_list = []
for fold_id, model in enumerate(models):
    test_predicts = model.predict(test_sequences, batch_size=256, verbose=1)
    test_predicts_list.append(test_predicts)

test_predicts = np.zeros(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts += fold_predict
test_predicts /= len(test_predicts_list)

# Reshape to template
test_ids = test_df["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]

submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(val_loss, avg_auc)
test_predicts.to_csv(submit_path, index=False)


print("Predicting training results...")
# Predict training data for future stacking
train_ids = train_df["id"].values
train_ids = train_ids.reshape((len(train_ids), 1))

train_predicts = pd.DataFrame(data=train_fold_preditcions, columns=CLASSES)
train_predicts["id"] = train_ids
train_predicts = train_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Train-L{:4f}-A{:4f}.csv".format(val_loss, avg_auc)
train_predicts.to_csv(submit_path, index=False)

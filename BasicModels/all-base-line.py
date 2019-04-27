import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, CuDNNGRU, Bidirectional, Conv1D, Dropout, PReLU, BatchNormalization
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, concatenate
from keras.layers import Flatten, add, MaxPooling1D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import initializers, constraints, regularizers
from keras.engine.topology import Layer
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')


EMBEDDING_FILE_FASTTEXT = r'F:/DataScience/NLP/pooled-gru-fasttext/crawl-300d-2M.vec'
EMBEDDING_FILE_GLOVE = r'F:/DataScience/NLP/glove.840B.300d/glove.840B.300d.txt'

train = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/train.csv')
test = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/test.csv')
submission = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/sample_submission.csv')

filepath = r"F:/DataScience/NLP/bestmodel2019"

X_train = train['comment_text'].fillna('no comment').values
X_test = test['comment_text'].fillna('no comment').values
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

max_len = 80  # 句子最大长度
max_features = 50000  # 最大单词量
embed_size = 300

tokenizer = Tokenizer(num_words=max_features)  # 只有出现次数最多的max_features个词会被记录
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_GLOVE, encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_fasttext = np.zeros((nb_words, embed_size))  # 这里定义的时候要注意里面的括号
embedding_matrix_glove = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector_fasttext = embedding_index.get(word)
    embedding_vector_glove = embedding_index.get(word)
    if embedding_vector_fasttext is not None: embedding_matrix_fasttext[i] = embedding_vector_fasttext
    if embedding_vector_glove is not None: embedding_matrix_glove[i] = embedding_vector_glove
embedding_matrix = np.concatenate((embedding_matrix_fasttext, embedding_matrix_glove), axis=1)
embed_size *= 2


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))



def get_model_BiGRU_fasttext(units=128):
    inp = Input(shape=(max_len, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    for _ in range(2):
        x = SpatialDropout1D(0.4)(x)
        x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    # x = SpatialDropout1D(0.2)(x)
    # x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def get_model_BiGRU_CNN(units=128, num_filter=64):
    inp = Input(shape=(max_len, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    x = Conv1D(num_filter, kernel_size=2, padding='valid')(x)
    avg_pool = GlobalAveragePooling1D()(x)  # (None, 79, 64) ---> (None, 64)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def get_model_textCNN(num_filter=32, filter_sizes=[1, 2, 3, 5]):
    inp = Input(shape=(max_len, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    conv_out = []
    for filter_size in filter_sizes:
        c = Conv1D(num_filter, filter_size, strides=1)(x)
        p = MaxPooling1D()(c)  # MaxPooling1D的poolsize默认为2，而globalmaxpooling是整个序列
        # 前者为对每两个词作一个maxpool，后者对这个句子的每个特征取maxpool
        conv_out.append(p)


    conc = concatenate([p for p in conv_out], axis=1)   # axis默认为-1,表示连接轴;除了连接轴，其他轴维数必须相同

    x_flatten = Flatten()(conc)
    outp = Dense(6, activation='sigmoid')(x_flatten)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def get_model_DPCNN(num_filter=64, filter_size=3, max_pool_size=3, max_pool_strides=2):
    conv_kern_reg = regularizers.l2(0.00001)
    conv_bias_reg = regularizers.l2(0.00001)

    inp = Input(shape=(max_len, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    block1 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    # 先filtersize取3，卷积成maxlen*64,然后filtersize取1，卷积成maxlen*64

    # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    # if you choose num_filter = embed_size (300 in this case) you don't have to do this part and can add x directly to block1_output
    resize_emb = Conv1D(num_filter, kernel_size=1, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
    resize_emb = PReLU()(resize_emb)

    # 相加两个卷积结果成maxlen*64，maxpool成(maxlen-1)/2 * 64,而非globalpool成1 * 64
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    ## 步2
    # 卷积成(maxlen-1)/2 * 64
    block2 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    # 再次卷积成(maxlen-1)/2 * 64
    block2 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    # 将两次卷积后的结果与上一步输出相加add（现在都是64维），然后maxpool输出
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    ## 步3 同步2
    block3 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    ## 步4 同步2
    block4 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    block4_output = add([block4, block3_output])
    block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

    ## 步5 同步2
    block5 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)
    block5 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)

    block5_output = add([block5, block4_output])
    block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

    ## 步6 前半部分同步2，后半部分换成GlobalMaxPool
    block6 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)
    block6 = Conv1D(num_filter, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)

    block6_output = add([block6, block5_output])
    # GlobalMaxPool输出1 * 64维度张量
    output = GlobalMaxPooling1D()(block6_output)  # 这里没有设为7个block，因为一个句子maxlen设为80，7个maxpool后，变为负数

    # 下面是两个dense层
    output = Dense(256, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(0.5)(output)
    output = Dense(6, activation='sigmoid')(output)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')  # 定义初始化格式

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        # 正则和限制的变量，用于后面build时声明参数
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
        # batch is every where, in the first dimension
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


def get_model_Attention(units=128, drop_rate=0.2):
    inp = Input(shape=(max_len, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(drop_rate)(x)
    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    x = Dropout(drop_rate)(x)
    x = Attention(max_len)(x)
    x = Dense(units, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    outp = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


model1 = get_model_BiGRU_fasttext()
# model2 = get_model_BiGRU_CNN()
# model3 = get_model_textCNN()
# model4 = get_model_DPCNN()
# model5 = get_model_Attention()

batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size=0.95, random_state=2019)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
check_point = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)


hist1 = model1.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                   callbacks=[RocAuc, early_stop, check_point], verbose=2)
# hist2 = model2.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                    callbacks=[RocAuc, early_stop, check_point], verbose=2)
# hist3 = model3.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                    callbacks=[RocAuc, early_stop, check_point], verbose=2)
# hist4 = model4.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                    callbacks=[RocAuc, early_stop, check_point], verbose=2)
# hist5 = model5.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                    callbacks=[RocAuc, early_stop, check_point], verbose=2)

y_pred1 = model1.predict(X_test, batch_size=1024)
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = y_pred1
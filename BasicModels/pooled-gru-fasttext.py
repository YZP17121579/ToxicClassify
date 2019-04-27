import numpy as np
np.random.seed(2019)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')


EMBEDDING_FILE = r'F:/DataScience/NLP/pooled-gru-fasttext/crawl-300d-2M.vec'

train = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/train.csv')
test = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/test.csv')
submission = pd.read_csv(r'F:/DataScience/NLP/pooled-gru-fasttext/sample_submission.csv')

X_train = train['comment_text'].fillna("fillna").values
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
X_test = test['comment_text'].fillna('fillna').values

max_features = 30000  # 最大单词量
maxlen = 100  # 一个句子、文档最大词数
embed_size = 300  # embed size

tokenizer = text.Tokenizer(num_words=max_features)  # 根据频率，只有最大的max_features个词会记录在token中
tokenizer.fit_on_texts(list(X_train) + list(X_test))  # 放入所有语料，生成token词典。输入参数为list类
# X_train = tokenizer.texts_to_matrix()
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr):  # 第一个为位置参数，之后的作为一个tuple输入到*arr中
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features : continue  # 裁剪之后去掉索引值在设定的max_feature以后的（若总词数小于max_feature,则不会进行这一步）
    embedding_vector = embedding_index.get(word)  # 这里不直接用embedding_index[word]是因为若没有这个word，会报错
    if embedding_vector is not None : embedding_matrix[i] = embedding_vector  # 从而导致程序结束


# 创建一个回调函数
class RocAucEvaluation(Callback):  # 继承的keras的callback类，有属性params和model，model指代被训练模型
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):  # 需要重写特定方法，用于显示运行日志等
        # on_batch_end, on_train_begin, on_epoch_end等名称代表回调函数调用时刻；
        # 在训练时，相应的回调函数的方法就会被在各自的阶段被调用。
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    # 不同于dropout随机的取零，spatialdropout是随机地将某一列元素取零，即词向量的某个维度取零
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)  # 用units表示输出维度因为类比了最后一层神经元
    # return_sequence表示所有隐状态a都被返回；上面的x为时序长度为maxlen, 每个词对应输出为80+80=160的向量
    avg_pool = GlobalAveragePooling1D()(x)  # 加Global表示，160个维度，每个维度的所有单词对应值（某一列）做平均、最大pool
    max_pool = GlobalMaxPooling1D()(x)  # textCNN也是同理，只不过将160维换为filter个数了（所有形状filter个数和）
    conc = concatenate([avg_pool, max_pool])  # textCNN6个filter得到6维，这里得到160维
    outp = Dense(6, activation='sigmoid')(conc)  # 这里每个类别都有一个sigmoid激活函数，而不是一个softmax

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  # 这里用binary_crossentropy，因为文档中句子可能属于多个类别，也可能一个类别都不是
                  # 只有每个句子只属于一个类别的时候，使用categorical_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = get_model()

batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=2019)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batchsize=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)

y_pred = model.predict(x_test, batch_size=1024)
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = y_pred


## 就差多层的CNN没试过了


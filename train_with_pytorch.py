import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

import DeepNet.deeptoxic.models.pytorch.bgru as bgru
import DeepNet.deeptoxic.train.trainer as trn
from DeepNet.deeptoxic.config import dataset_config
from DeepNet.deeptoxic.config import model_config
from DeepNet.deeptoxic.data_helper.data_transformer import DataTransformer
from DeepNet.deeptoxic.data_helper.data_loader import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# # For jupyter notebook
# import importlib
# importlib.reload(bgru)
# importlib.reload(trn)


VOCAB_SIZE = model_config.VOCAB_SIZE
MAX_SEQUENCE_LENGTH = model_config.MAX_SEQUENCE_LENGTH
EMBEDDING_SIZE = model_config.EMBEDDING_SIZE
char_level = model_config.char_level

print("Loading data ...")
data_transformer = DataTransformer(max_num_words=VOCAB_SIZE, pad_seq=False,
                                   max_sequence_length=MAX_SEQUENCE_LENGTH, char_level=char_level, error_correct=False)
data_loader = DataLoader()

# prepare data
train_sequences, training_labels, test_sequences = data_transformer.prepare_data()

# load embedding matrix
embeddings_index = data_loader.load_embedding(dataset_config.FASTTEXT_PATH)  # 2,000,000 * 300
embedding_matrix = data_transformer.build_embedding_matrix(embeddings_index)  # 100,000 * 300
print("Loaded")


# Pytorch model
def get_bgru_network():
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    embedding.weight.requires_grad = False
    return bgru.BayesianGRUClassifier(input_size=EMBEDDING_SIZE, hidden_size=60, embedding=embedding)


trainer = trn.PyTorchModelTrainer(model_stamp="FASTTXT_BGRU_64_64", epoch_num=5, learning_rate=1e-3,
                                  verbose_round=40, shuffle_inputs=False, early_stopping_round=10)

models, best_logloss, avg_auc, best_val_pred = \
    trainer.train_folds(X=train_sequences, y=training_labels,
                        fold_count=10, batch_size=256, get_model_func=get_bgru_network, skip_fold=0)


# Compute average AUC of all folds
train_fold_preditcions = np.concatenate(best_val_pred, axis=0)
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

submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(best_logloss, avg_auc)
test_predicts.to_csv(submit_path, index=False)


print("Predicting training results...")
# Predict training data for future stacking
train_ids = train_df["id"].values
train_ids = train_ids.reshape((len(train_ids), 1))

train_predicts = pd.DataFrame(data=train_fold_preditcions, columns=CLASSES)
train_predicts["id"] = train_ids
train_predicts = train_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Train-L{:4f}-A{:4f}.csv".format(best_logloss, avg_auc)
train_predicts.to_csv(submit_path, index=False)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import DeepNet.deeptoxic.models.keras.model_class as model_zoo
from DeepNet.deeptoxic.train import trainer
from DeepNet.deeptoxic.data_helper.data_loader import DataLoader
from DeepNet.deeptoxic.data_helper.data_transformer import DataTransformer
from DeepNet.deeptoxic.config import dataset_config, model_config

VOCAB_SIZE = model_config.VOCAB_SIZE
MAX_SEQUENCE_LENGTH = model_config.MAX_SEQUENCE_LENGTH
EMBEDDING_SIZE = model_config.EMBEDDING_SIZE


data_transformer = DataTransformer(max_num_words=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH, char_level=False)
data_loader = DataLoader()

# prepare data
train_sequences, training_labels, test_sequences = data_transformer.prepare_data()

# load embedding matrix
embeddings_index = data_loader.load_embedding(dataset_config.FASTTEXT_PATH)
embedding_matrix = data_transformer.build_embedding_matrix(embeddings_index)

nb_words = embedding_matrix.shape[0]


def get_model():
    return model_zoo.get_av_rnn(nb_words, EMBEDDING_SIZE, embedding_matrix, MAX_SEQUENCE_LENGTH, out_size=6)


keras_model_trainer = trainer.KerasModelTrainer(model_stamp='kmax_text_cnn', epoch_num=50, learning_rate=1e-3)
models, val_loss, total_auc, fold_predictions = \
    keras_model_trainer.train_folds(train_sequences, training_labels, fold_count=10,
                                    batch_size=256, get_model_func=get_model)

print("Overall val-loss:", val_loss, "AUC", total_auc)

# Compute average AUC of all folds
train_fold_preditcions = np.concatenate(fold_predictions, axis=0)
training_auc = roc_auc_score(training_labels[:-1], train_fold_preditcions)
print("Training AUC", training_auc)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submit_path_prefix = "results/rnn/nds/fasttext-SC2-nds-randomNoisy-capNet-" + str(nb_words) + "-RST-lp-ct-" + str(MAX_SEQUENCE_LENGTH)

print("Predicting testing results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    test_predicts = model.predict(test_sequences, batch_size=256, verbose=False)
    test_predicts_list.append(test_predicts)
    np.save("predict_path/", test_predicts)

test_predicts = np.zeros(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts += fold_predict
test_predicts /= len(test_predicts_list)

test_ids = test_sequences["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(val_loss, total_auc)
test_predicts.to_csv(submit_path, index=False)

print("Predicting training results...")

train_ids = train_sequences["id"].values
train_ids = train_ids.reshape((len(train_ids), 1))

train_predicts = pd.DataFrame(data=train_fold_preditcions, columns=CLASSES)
train_predicts["id"] = train_ids
train_predicts = train_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Train-L{:4f}-A{:4f}.csv".format(val_loss, training_auc)
train_predicts.to_csv(submit_path, index=False)

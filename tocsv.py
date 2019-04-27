from DeepNet.deeptoxic.models.pytorch import bgru
from DeepNet.deeptoxic.config import model_config, dataset_config
from DeepNet.deeptoxic.data_helper.data_transformer import DataTransformer
from DeepNet.deeptoxic.data_helper.data_loader import DataLoader
import pandas as pd
import torch.nn as nn
import torch


MAX_SEQUENCE_LENGTH = model_config.MAX_SEQUENCE_LENGTH
VOCAB_SIZE = model_config.VOCAB_SIZE
EMBEDDING_SIZE = model_config.EMBEDDING_SIZE
char_level = model_config.char_level

data_loader = DataLoader()
data_transformer = DataTransformer(max_num_words=VOCAB_SIZE, pad_seq=False,
                                   max_sequence_length=MAX_SEQUENCE_LENGTH, char_level=char_level, error_correct=False)
train_seq, train_tag, test_seq = data_transformer.prepare_data()
embeddings_index = data_loader.load_embedding(dataset_config.FASTTEXT_PATH)
embedding_matrix = data_transformer.build_embedding_matrix(embeddings_index)

def get_bgru_network():
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    embedding.weight.requires_grad = False
    return bgru.BayesianGRUClassifier(input_size=EMBEDDING_SIZE, hidden_size=60, embedding=embedding)

model = get_bgru_network()
# model.load() return None
model.load(model_config.TEMPORARY_CHECKPOINTS_PATH + "FASTTXT_BGRU_64_64-TEMP.pt")
model = model.cuda()
test_tag = model.predict(test_seq)

path = 'Dataset/'
TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

test_ids = test_df['id'].values
test_ids = test_ids.reshape(len(test_ids), 1)

df = pd.DataFrame(data=test_tag, columns=CLASSES)
df['id'] = test_ids
df = df[["id"] + CLASSES]

df.to_csv("Dataset/submit.csv", index=False)




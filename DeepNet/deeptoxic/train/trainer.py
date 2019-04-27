import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import importlib

from sklearn.metrics import roc_auc_score
from DeepNet.deeptoxic.utils.score import log_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint

from DeepNet.deeptoxic.config import model_config
from DeepNet.deeptoxic import utils
importlib.reload(utils)


class ModelTrainer(object):

    def __init__(self, model_stamp, epoch_num, learning_rate=1e-3,
                 shuffle_inputs=False, verbose_round=40, early_stopping_round=8):
        """
        :param model_stamp: model文件储存名
        :param epoch_num: 最大epoch
        :param learning_rate: 学习率
        :param shuffle_inputs: 打乱顺序
        :param verbose_round: 基于epoch
        :param early_stopping_round: 基于epoch
        """
        self.models = []
        self.model_stamp = model_stamp
        self.val_loss = -1
        self.auc = -1
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.eps = 1e-10
        self.verbose_round = verbose_round
        self.early_stopping_round = early_stopping_round
        self.shuffle_inputs = shuffle_inputs

    def train_folds(self, X, y, fold_count, batch_size, get_model_func, skip_fold=0):
        """
        :param X: 样本特征矩阵
        :param y: 样本标记
        :param fold_count: 交叉验证的 fold 数
        :param batch_size: 批量SGD的 batch_size
        :param get_model_func: 输入的func需要是一个返回 keras 的 model 的函数
        :param skip_fold:
        :return:
        """
        fold_size = len(X) // fold_count
        models = []
        fold_predictions = []
        score = 0
        total_auc = 0

        for fold_id in range(0, fold_count):
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            # for the last fold
            if fold_id == fold_count - 1:
                fold_end = len(X)

            # cross validation
            train_x = np.concatenate([X[:fold_start], X[fold_end:]])  # 返回np.array
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x = X[fold_start:fold_end]
            val_y = y[fold_start:fold_end]

            if fold_id < skip_fold:
                model = get_model_func()
                model.load(model_config.MODEL_CHECKPOINT_FOLDER + self.model_stamp + str(fold_id) + ".pt")
                model = model.eval()
                model = model.cuda()
                fold_prediction = model.predict(val_x)
                auc = roc_auc_score(val_y, fold_prediction)
                bst_val_score = log_loss(y=val_y, y_pred=fold_prediction)
            else:
                model, bst_val_score, auc, fold_prediction = self._train_model_by_logloss(
                    get_model_func(), batch_size, train_x, train_y, val_x, val_y, fold_id)
            score += bst_val_score
            total_auc += auc
            models.append(model)
            fold_predictions.append(fold_prediction)

        self.models = models
        self.val_loss = score / fold_count
        self.auc = total_auc / fold_count
        return models, self.val_loss, self.auc, fold_predictions

    def keep_train_folds(self, X, y, fold_count, batch_size, old_models):
        fold_size = len(X) // fold_count
        models = []
        fold_predictions = []
        score = 0
        total_auc = 0

        for fold_id in range(0, fold_count):
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            if fold_id == fold_count - 1:
                fold_end = len(X)

            train_x = np.concatenate([X[:fold_start], X[fold_end:]])
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x = X[fold_start:fold_end]
            val_y = y[fold_start:fold_end]

            model, bst_val_score, auc, fold_prediction = self._train_model_by_logloss(
                old_models[fold_id], batch_size, train_x, train_y, val_x, val_y, fold_id)
            score += bst_val_score
            total_auc += auc
            models.append(model)
            fold_predictions.append(fold_prediction)

        self.models = models
        self.val_loss = score / fold_count
        self.auc = total_auc / fold_count
        return models, self.val_loss, self.auc, fold_predictions

    def _train_model_by_auc(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        raise NotImplementedError

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        raise NotImplementedError

    def evaluate(self, test_data, dataframe, submit_path_prefix):

        print("Predicting results...")
        test_predicts_list = []
        for fold_id, model in enumerate(self.models):
            test_predicts = model.predict(test_data, batch_size=512, verbose=False)
            test_predicts_list.append(test_predicts)
            np.save("predict_path/", test_predicts)

        test_predicts = np.zeros(test_predicts_list[0].shape)
        for fold_predict in test_predicts_list:
            test_predicts += fold_predict
        test_predicts /= len(test_predicts_list)

        ids = dataframe["id"].values
        ids = ids.reshape((len(ids), 1))
        CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
        test_predicts["id"] = ids
        test_predicts = test_predicts[["id"] + CLASSES]
        submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(self.val_loss, self.total_auc)
        test_predicts.to_csv(submit_path, index=False)


class KerasModelTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(KerasModelTrainer, self).__init__(*args, **kwargs)
        pass

    def _train_model_by_auc(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        print("Training on fold", fold_id + 1)
        best_auc = -1
        best_epoch = 0
        current_epoch = 1

        for epoch in range(self.epoch_num):
            model.fit(train_x, train_y, validation_data=[val_x, val_y],
                      epochs=1, batch_size=batch_size, shuffle=True)
            y_pred = model.predict(val_x, batch_size=batch_size)
            current_auc = roc_auc_score(val_y, y_pred)
            print("Epoch {}\tauc {:.6f}\tbest_auc {:.6f}".format(current_epoch, current_auc, best_auc))
            current_epoch += 1
            if best_auc < current_auc or best_auc == -1:
                best_auc = current_auc
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == self.early_stopping_round:
                    break

        return model, best_auc

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_round)
        bst_model_path = self.model_stamp + str(fold_id) + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        hist = model.fit(train_x, train_y,
                         validation_data=(val_x, val_y),
                         epochs=self.epoch_num, batch_size=batch_size, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint])
        bst_val_score = min(hist.history['val_loss'])
        predictions = model.predict(val_x)
        auc = roc_auc_score(val_y, predictions)
        print("AUC Score", auc)
        return model, bst_val_score, auc, predictions


class PyTorchModelTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        """
        Change loss_function here.
        只要是binary，都表示类别全二分类；只要是with_logits都表示内置激活函数
        """
        super(PyTorchModelTrainer, self).__init__(*args, **kwargs)
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.93

    def _train_model_by_auc(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        pass

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        print("Training on fold", fold_id + 1)

        # while using pytorch, model.cuda() is necessary if cuda be used
        if model_config.use_cuda:
            model = model.cuda()
        best_auc = -1
        best_logloss = -1
        best_epoch = 0
        current_epoch = 1
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)

        for epoch in range(self.epoch_num):
            epoch_logloss = 0
            # Generate mini_batch by generator
            for batch_id, (inputs_var, targets_var) in enumerate(
                    utils.generators.mini_batches_generator(train_x, train_y, batch_size, row_shuffle=self.shuffle_inputs)):
                loss = self._train_batch(model=model, inputs_var=inputs_var, targets_var=targets_var)

                # logging, TODO: add tensorboard for visualization
                epoch_logloss += loss
                if batch_id % self.verbose_round == 0:
                    print("Epoch: {}\tBatch: {}\tLog-loss {}".format(epoch + 1, batch_id, loss))

            # validation
            print("Epoch average log loss: {}".format(epoch_logloss / batch_id))
            val_pred = model.predict(val_x)

            current_logloss = log_loss(val_y, val_pred)
            current_epoch += 1
            if best_logloss > current_logloss or best_logloss == -1:
                best_logloss = current_logloss
                model.save(model_config.TEMPORARY_CHECKPOINTS_PATH + self.model_stamp + "-TEMP.pt")
                best_auc = roc_auc_score(val_y, val_pred)
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == self.early_stopping_round:
                    break
            print("In Epoch {}, val_loss: {}\tbest_val_loss: {}\tbest_auc: {}\n".format
                  (epoch + 1, current_logloss, best_logloss, best_auc))
            self.adjust_learning_rate()

        model.load(model_config.TEMPORARY_CHECKPOINTS_PATH + self.model_stamp + "-TEMP.pt")
        best_val_pred = model.predict(val_x)
        model.save(model_config.MODEL_CHECKPOINT_FOLDER + self.model_stamp + str(fold_id) + ".pt")
        return model, best_logloss, best_auc, best_val_pred

    def _train_batch(self, model, inputs_var, targets_var):
        self.optimizer.zero_grad()
        model.train()
        preds_var = model(inputs_var)
        # training
        loss = self.criterion(preds_var, targets_var)
        loss.backward()
        # to avoid gradient booming
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=3)
        self.optimizer.step()
        return loss.item()

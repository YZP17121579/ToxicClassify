import pandas as pd

train_df = pd.read_csv('Dataset/train.csv')

glove_rnn_predections = "results/rnn/ensemble/dropout-glove-bigru-attall-lp-ct-550-Train-L0.038997-A0.988497.csv"
prediction_name = 'glove_avrnn'
check_prediction = pd.read_csv(glove_rnn_predections)

print(check_prediction.head())


def get_error_term_pos(train, pred, check_column):
    sub_train = train[check_column]
    sub_pred = pred[check_column]
    sub_pred = sub_pred.round()
    diff = (sub_pred != sub_train) & (sub_train == 1)
    print("Wrong predections number:", diff.sum())
    pos = pd.DataFrame()
    pos['id'] = train[diff]['id']
    pos['text'] = train[diff]['comment_text']
    pos['pred_val'] = pred[diff][check_column]
    pos['label'] = train[diff][check_column]
    return pos


def get_error_term_neg(train, pred, check_column):
    sub_train = train[check_column]
    sub_pred = pred[check_column]
    sub_pred = sub_pred.round()
    diff = (sub_pred != sub_train) & (sub_train == 0)
    print("Wrong predections number:", diff.sum())
    neg = pd.DataFrame()
    neg['id'] = train[diff]['id']
    neg['text'] = train[diff]['comment_text']
    neg['pred_val'] = pred[diff][check_column]
    neg['label'] = train[diff][check_column]
    return neg


for term in train_df.columns.values[2:]:
    print("In term:", term)
    err_neg = get_error_term_neg(train_df, check_prediction, term)
    err_pos = get_error_term_pos(train_df, check_prediction, term)

    err_neg.to_csv("analysis/ERR-Neg"+term+"-"+prediction_name+".csv", index=False)
    err_pos.to_csv("analysis/ERR-Pos"+term+"-"+prediction_name+".csv", index=False)

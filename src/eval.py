from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np


def rnn_performance_eval(prediction, label, label_length):
    """prediction: logits"""
    prediction = prediction.detach().numpy()
    prediction = np.exp(prediction) / np.sum(np.exp(prediction), axis=2)[:, :, np.newaxis]
    prediction = prediction[:, :, 1]
    label = label.detach().numpy()
    label_length = np.array(label_length, dtype=np.int)

    result_list = []

    # general auc
    score = []
    true = []
    for pat_index in range(len(prediction)):
        for visit_idx in range(label_length[pat_index]):
            score.append(prediction[pat_index][visit_idx])
            true.append(label[pat_index][visit_idx])
    auc, tn, fp, fn, tp, optimal_cut = binary_performance_eval(score, true)
    result_list.append(['general', auc, tn, fp, fn, tp, optimal_cut])

    valid_idx_list = np.zeros(len(prediction))
    for i in range(4):
        for index in range(len(valid_idx_list)):
            if label_length[index] >= i:
                valid_idx_list[index] = 1
    for i in range(4):
        score = []
        true = []
        for pat_index in range(len(prediction)):
            if valid_idx_list[pat_index] == 1:
                score.append(prediction[pat_index][i])
                true.append(label[pat_index][i])
        auc, tn, fp, fn, tp, optimal_cut = binary_performance_eval(score, true)
        result_list.append(["visit {}".format(i+1), auc, tn, fp, fn, tp, optimal_cut])
    return result_list


def binary_performance_eval(pred_prob, label):
    # calc acc, precision, and recall by youden index
    optimal_cut = 0
    optimal_youden_index = -1
    pred_prob = np.array(pred_prob)
    label = np.array(label, dtype=np.bool)
    for cut in [i/100 for i in range(100)]:
        pred = pred_prob >= cut
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_index = sensitivity + specificity - 1
        if youden_index > optimal_youden_index:
            optimal_youden_index = youden_index
            optimal_cut = cut
    auc = roc_auc_score(label, pred_prob)
    tn, fp, fn, tp = confusion_matrix(label, pred_prob > optimal_cut).ravel()
    return auc, tn, fp, fn, tp, optimal_cut

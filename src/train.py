import torch
from model import PredictionModel
from read_data import read_label, read_dynamic_data, read_static_data, read_treatment, generate_train_test_data, \
    FiveFoldCrossValidation, data_shift
import numpy as np
import csv
from eval import rnn_performance_eval


def train(data_, hidden_size, training_step):
    s_input_size = len(data_[0][0][0])
    d_input_size = len(data_[0][1][0][0])
    result_list = []

    for fold in range(5):
        test_static, test_dynamic, test_label, test_valid_length, train_static, train_dynamic, train_label, \
            train_valid_length = generate_train_test_data(data_, fold, 'sequence')

        learning_rate = 1e-3
        model = PredictionModel(s_input_size, d_input_size, hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        train_mask = create_mask(train_label.shape, train_valid_length)
        for t in range(1, training_step+1):
            model.train()
            optimizer.zero_grad()
            # Forward pass: compute predicted y by passing x to the model.
            prediction = model(train_static, train_dynamic).squeeze()
            predict_eval = prediction.clone().detach()
            train_label_eval = train_label.clone().detach()
            prediction_train = prediction.reshape([-1, 2])
            train_label_train = train_label.reshape([-1]).long()
            train_mask_train = train_mask.reshape([-1])
            # Compute and print loss.
            loss = loss_fn(prediction_train, train_label_train)
            loss = torch.mean(train_mask_train * loss)
            if t % 100 == 0:
                model.eval()
                with torch.no_grad():
                    train_result = rnn_performance_eval(predict_eval, train_label_eval, train_valid_length)
                    test_pred = model(test_static, test_dynamic).squeeze()
                    test_result = rnn_performance_eval(test_pred, test_label, test_valid_length)
                    print("iter: {:>6}, pred loss: {:.4f}, train auc: {:.4f}, test_auc: {:.4f}"
                          .format(t, loss.item(), train_result[0][1], test_result[0][1]))
            loss.backward()
            optimizer.step()
        test_pred = model(test_static, test_dynamic).squeeze()
        result_list.append(rnn_performance_eval(test_pred, test_label, test_valid_length))
    print("accomplished")
    return result_list


def create_mask(size, seq_len):
    seq_len = np.array(seq_len, dtype=np.int)
    mask = np.zeros((size[0], size[1]))
    for index_1 in range(len(mask)):
        for index_2 in range(seq_len[index_1]):
            mask[index_1][index_2] = 1
    return torch.from_numpy(mask).float()


def main():
    static_feature_path = '../resource/data_v2/baseline.csv'
    dynamic_feature_path = '../resource/data_v2/dynamic.csv'
    label_path = '../resource/data_v2/label.csv'
    treatment_path = '../resource/data_v2/treatment.csv'
    side_effect_name_list = 'side_effect_1', 'side_effect_2', 'side_effect_3', 'side_effect_4'
    treatment_name_list = 'treatment_1', 'treatment_2', 'treatment_3', 'treatment_4'
    training_step = 2000
    hidden_size = 16

    output = [['rnn', 'i', 'side_effect_name', 'j', 'eval_info_g', 'auc_g', 'tn_g', 'fp_g', 'fn_g', 'tp_g',
               'optimal_cut_g', 'eval_info_1', 'auc_1', 'tn_1', 'fp_1', 'fn_1', 'tp_1', 'optimal_cut_1',
               'eval_info_2', 'auc_2', 'tn_2', 'fp_2', 'fn_2', 'tp_2', 'optimal_cut_2', 'eval_info_3', 'auc_3', 'tn_3',
               'fp_3', 'fn_3', 'tp_3', 'optimal_cut_3', 'eval_info_4', 'auc_4', 'tn_4', 'fp_4', 'fn_4', 'tp_4',
               'optimal_cut_4']]

    for i in range(2):
        for item in zip(side_effect_name_list, treatment_name_list):
            side_effect_name, treatment_name = item
            print('Iteration: {}'.format(i))
            print(side_effect_name)
            print(treatment_name)

            label = read_label(label_path)
            dynamic_data = read_dynamic_data(dynamic_feature_path)
            static_data = read_static_data(static_feature_path)
            treatment_data = read_treatment(treatment_path)
            dynamic_data, treatment_data, label = data_shift(dynamic_data, treatment_data, label)

            cross_validation = FiveFoldCrossValidation(label, dynamic_data, static_data, treatment_data)
            dataset = cross_validation.generate_five_fold(treatment_name, side_effect_name)
            result = train(dataset, hidden_size, training_step)

            for j in range(5):
                general, v1, v2, v3, v4 = result[j]
                eval_info_g, auc_g, tn_g, fp_g, fn_g, tp_g, optimal_cut_g = general
                eval_info_1, auc_1, tn_1, fp_1, fn_1, tp_1, optimal_cut_1 = v1
                eval_info_2, auc_2, tn_2, fp_2, fn_2, tp_2, optimal_cut_2 = v2
                eval_info_3, auc_3, tn_3, fp_3, fn_3, tp_3, optimal_cut_3 = v3
                eval_info_4, auc_4, tn_4, fp_4, fn_4, tp_4, optimal_cut_4 = v4
                output.append(['rnn', i, side_effect_name, j, eval_info_g, auc_g, tn_g, fp_g, fn_g, tp_g, optimal_cut_g,
                               eval_info_1, auc_1, tn_1, fp_1, fn_1, tp_1, optimal_cut_1, eval_info_2, auc_2, tn_2,
                               fp_2, fn_2, tp_2, optimal_cut_2, eval_info_3, auc_3, tn_3, fp_3, fn_3, tp_3,
                               optimal_cut_3, eval_info_4, auc_4, tn_4, fp_4, fn_4, tp_4, optimal_cut_4])
    with open('../resource/rnn.csv', 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(output)


if __name__ == '__main__':
    main()

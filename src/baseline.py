from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from eval import binary_performance_eval
from sklearn.ensemble import AdaBoostClassifier
from read_data import read_label, read_dynamic_data, read_static_data, read_treatment, data_shift, \
    FiveFoldCrossValidation, generate_train_test_data
import csv
import datetime
import numpy as np


def train(dataset, model_name, data_num, feature_list=None):
    result_list = []

    for fold in range(5):
        test_label, test_input, train_label, train_input = generate_train_test_data(dataset, fold, data_num,
                                                                                    'single', index_list=feature_list)
        if model_name == 'LR':
            model = LogisticRegression()
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=2000)
        elif model_name == 'MLP':
            model = MLPClassifier(max_iter=10000)
        elif model_name == 'Adaboost':
            model = AdaBoostClassifier()
        else:
            raise ValueError('')

        model.fit(train_input, train_label)
        predict = model.predict_proba(test_input)[:, 1]
        result = binary_performance_eval(predict, test_label), predict, test_label
        result_list.append(result)
    return result_list


def train_wrap(label_path, dynamic_feature_path, static_feature_path, treatment_path, treatment_name, side_effect_name,
               model_name, data_num, feature_list=None):
    label = read_label(label_path)
    dynamic_data = read_dynamic_data(dynamic_feature_path)
    static_data = read_static_data(static_feature_path)
    treatment_data = read_treatment(treatment_path)
    dynamic_data, treatment_data, label = data_shift(dynamic_data, treatment_data, label)

    cross_validation = FiveFoldCrossValidation(label, dynamic_data, static_data, treatment_data)
    dataset = cross_validation.generate_five_fold(treatment_name, side_effect_name)
    result = train(dataset, model_name, data_num, feature_list)
    return result


def main():
    static_feature_path = '../resource/data_v2/baseline.csv'
    dynamic_feature_path = '../resource/data_v2/dynamic.csv'
    label_path = '../resource/data_v2/label.csv'
    treatment_path = '../resource/data_v2/treatment.csv'
    side_effect_name_list = ('side_effect_4', )
    treatment_name_list = ('treatment_4', )
    data_size = [40*(10-i) for i in range(10)]

    # feature_dict = {
    #     'side_effect_1': [27, 35, 18, 44, 36, 4, 14, 34, 23, 17, 2, 45],
    #     'side_effect_3': [36, 23, 1, 26, 10, 30, 31, 35, 45, 34, 37, 9],
    #     'side_effect_4': [38, 21, 16, 10, 25, 20]
    # }

    output_result = [
        ['model', 'data_num', 'iter', 'side_effect', 'test_fold', 'auc', 'tn', 'fp', 'fn', 'tp', 'cutoff point']]
    for data_num in data_size:
        for model_name in ['RF', 'LR', 'Adaboost', 'MLP']:
            for item in zip(side_effect_name_list, treatment_name_list):
                label_list, pred_list = [], []
                side_effect_name, treatment_name = item
                for i in range(10):
                    result = train_wrap(label_path, dynamic_feature_path, static_feature_path, treatment_path,
                                        treatment_name, side_effect_name, model_name, data_num)
                    for j in range(5):
                        (auc, tn, fp, fn, tp, optimal_cut), predict, test_label = result[j]
                        print("model: {}, test auc: {}".format(model_name, auc))
                        output_result.append([model_name, data_num, i, side_effect_name, j, auc, tn, fp, fn, tp,
                                              optimal_cut])
                        for label_item in test_label:
                            label_list.append(label_item)
                        for pred_item in predict:
                            pred_list.append(pred_item)
                np.save('../resource/result/{}_{}_{}_pred.npy'.format(model_name, data_num, side_effect_name),
                        pred_list)
                np.save('../resource/result/{}_{}_{}_label.npy'.format(model_name, data_num, side_effect_name),
                        label_list)

    with open('../resource/baseline_{}.csv'.format(datetime.datetime.now().strftime('%d%m%Y%H%M%S')),
              'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(output_result)


if __name__ == '__main__':
    main()

import numpy as np
import os
from baseline import train_wrap


def main():
    """使用LR，做十折交叉验证，选最优顺序"""

    static_feature_path = '../resource/data_v2/baseline.csv'
    dynamic_feature_path = '../resource/data_v2/dynamic.csv'
    label_path = '../resource/data_v2/label.csv'
    treatment_path = '../resource/data_v2/treatment.csv'
    side_effect_name_list = 'side_effect_1', 'side_effect_3', 'side_effect_4'
    treatment_name_list = 'treatment_1', 'treatment_3', 'treatment_4'

    feature_order = dict()
    optimal_auc_dict = dict()

    for item in zip(side_effect_name_list, treatment_name_list):
        best_feature_idx_order = []
        valid_feature_idx = set([i for i in range(46)])
        side_effect_name, treatment_name = item
        optimal_auc_dict[side_effect_name] = list()

        while valid_feature_idx.__len__() > 0:
            auc_dict = dict()
            for feature_index in valid_feature_idx:
                feature_list = [item for item in best_feature_idx_order]
                feature_list.append(feature_index)
                auc_list = []
                for i in range(5):
                    result = train_wrap(label_path, dynamic_feature_path, static_feature_path, treatment_path,
                                        treatment_name, side_effect_name, 'LR', 'None', feature_list)
                    for j in range(5):
                        (auc, _, _, _, _, _), _, _ = result[j]
                        auc_list.append(auc)
                print('feature idx: {}. auc: {}, feature_list: {}'
                      .format(feature_index, np.mean(auc_list), feature_list))
                auc_dict[feature_index] = np.mean(auc_list)

            optimal_auc, optimal_idx = -1, -1
            for key in auc_dict:
                if auc_dict[key] > optimal_auc:
                    optimal_auc = auc_dict[key]
                    optimal_idx = key
            valid_feature_idx.remove(optimal_idx)
            best_feature_idx_order.append(optimal_idx)
            optimal_auc_dict[side_effect_name].append([optimal_idx, optimal_auc])
        print('{} feature order: {}'.format(side_effect_name, best_feature_idx_order))
        feature_order[side_effect_name] = best_feature_idx_order

    for side_effect in feature_order:
        print('{}, feature list: {}'.format(side_effect, feature_order[side_effect]))
    for side_effect in optimal_auc_dict:
        for item in optimal_auc_dict[side_effect]:
            print('{}, feature index: {}, auc: {}'.format(side_effect, item[0], item[1]))


if __name__ == '__main__':
    main()
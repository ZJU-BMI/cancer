import csv
from itertools import islice
import datetime
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


class FiveFoldCrossValidation(object):
    def __init__(self, label, dynamic_data, static_data, treatment_data):
        self._label = label
        self._dynamic_data = dynamic_data
        self._static_data = static_data
        self._treatment_data = treatment_data
        self._check_consistency()

    def _check_consistency(self):
        """保证所有index是一一对应的"""
        label_idx_set = set([pat_idx for pat_idx in self._label])
        dynamic_data_idx_set = set([pat_idx for pat_idx in self._dynamic_data])
        static_data_idx_set = set([pat_idx for pat_idx in self._static_data])
        treatment_data_idx_set = set([pat_idx for pat_idx in self._treatment_data])
        assert len(label_idx_set) == len(dynamic_data_idx_set)
        assert len(label_idx_set) == len(static_data_idx_set)
        assert len(label_idx_set) == len(treatment_data_idx_set)
        for pat_idx in label_idx_set:
            if not dynamic_data_idx_set.__contains__(pat_idx):
                raise ValueError('inconsistent')
            if not static_data_idx_set.__contains__(pat_idx):
                raise ValueError('inconsistent')
            if not treatment_data_idx_set.__contains__(pat_idx):
                raise ValueError('inconsistent')

    def generate_five_fold(self, t_name, l_name):
        pat_id_set = [pat_idx for pat_idx in self._label]
        permute = np.random.permutation(pat_id_set)

        data_len = len(permute) // 5
        s_data_1, d_data_1, label_1, valid_length_1 = self._reorganize(permute[0: data_len], t_name, l_name)
        s_data_2, d_data_2, label_2, valid_length_2 = self._reorganize(permute[data_len: data_len*2], t_name, l_name)
        s_data_3, d_data_3, label_3, valid_length_3 = self._reorganize(permute[data_len*2: data_len*3], t_name, l_name)
        s_data_4, d_data_4, label_4, valid_length_4 = self._reorganize(permute[data_len*3: data_len*4], t_name, l_name)
        s_data_5, d_data_5, label_5, valid_length_5 = self._reorganize(permute[data_len*4:], t_name, l_name)
        return (s_data_1, d_data_1, label_1, valid_length_1), \
               (s_data_2, d_data_2, label_2, valid_length_2), \
               (s_data_3, d_data_3, label_3, valid_length_3), \
               (s_data_4, d_data_4, label_4, valid_length_4), \
               (s_data_5, d_data_5, label_5, valid_length_5)

    def _reorganize(self, id_set, treatment_name, label_name):
        dynamic_data_list = []
        static_data_list = []
        label_list = []
        valid_length = []
        for pat_id in id_set:
            dynamic_data = np.array(self._dynamic_data[pat_id], dtype=np.float)
            label = [item[label_name] for item in self._label[pat_id]]
            treatment = np.array([item[treatment_name] for item in self._treatment_data[pat_id]], dtype=np.float)
            static_data = np.array(self._static_data[pat_id], dtype=np.float)
            valid_length.append(len(dynamic_data))
            fuse_dynamic = np.concatenate([dynamic_data, treatment[:, np.newaxis]], axis=1)
            static_data_list.append(static_data)
            label_list.append(np.array(label, dtype=np.int))
            dynamic_data_list.append(fuse_dynamic)
        return static_data_list, dynamic_data_list, label_list, valid_length


def pack_seq_data(data_):
    """
    处理数据的变长特点，参考
    https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
    """
    data_ = [torch.from_numpy(item).float() for item in data_]
    data_ = rnn_utils.pad_sequence(data_, batch_first=True, padding_value=0)
    return data_


def generate_train_test_data(data_, test_fold_idx, data_num, reorganize_type, index_list=None):
    test_static, test_dynamic, test_label, test_valid_length = data_[test_fold_idx]
    train_static = []
    train_dynamic = []
    train_label = []
    train_valid_length = []
    for index in range(5):
        if index == test_fold_idx:
            continue
        for item in data_[index][0]:
            train_static.append(item)
        for item in data_[index][1]:
            train_dynamic.append(item)
        for item in data_[index][2]:
            train_label.append(item)
        train_valid_length.append(data_[index][3])

    if reorganize_type == 'sequence':
        if data_num != 'None' and data_num is not None:
            raise ValueError('')
        test_dynamic = pack_seq_data(test_dynamic)
        test_static = torch.from_numpy(np.array(test_static, dtype=np.float)).float()
        test_label = pack_seq_data(test_label)
        test_valid_length = np.array(test_valid_length, dtype=np.float)

        train_static = torch.from_numpy(np.array(train_static, dtype=np.float)).float()
        train_valid_length = np.concatenate(train_valid_length, axis=0)
        train_dynamic = pack_seq_data(train_dynamic)
        train_label = pack_seq_data(train_label)
        return test_static, test_dynamic, test_label, test_valid_length, train_static, train_dynamic, train_label, \
            train_valid_length
    elif reorganize_type == 'single':
        test_label_out = []
        test_static_out = []
        test_dynamic_out = []
        train_label_out = []
        train_static_out = []
        train_dynamic_out = []
        for pat_id in range(len(test_label)):
            for item_id in range(len(test_label[pat_id])):
                test_label_out.append(test_label[pat_id][item_id])
                test_static_out.append(test_static[pat_id])
                test_dynamic_out.append(test_dynamic[pat_id][item_id])

        test_label_out = np.array(test_label_out)
        test_static_out = np.array(test_static_out)
        test_dynamic_out = np.array(test_dynamic_out)

        for pat_id in range(len(train_label)):
            for item_id in range(len(train_label[pat_id])):
                train_label_out.append(train_label[pat_id][item_id])
                train_static_out.append(train_static[pat_id])
                train_dynamic_out.append(train_dynamic[pat_id][item_id])

        train_label_out = np.array(train_label_out)
        train_static_out = np.array(train_static_out)
        train_dynamic_out = np.array(train_dynamic_out)

        train_input = np.concatenate([train_static_out, train_dynamic_out], axis=1)
        test_input = np.concatenate([test_static_out, test_dynamic_out], axis=1)

        if data_num is not None and data_num != 'None':
            permute = np.random.permutation(np.arange(len(train_input)))
            train_label_out = train_label_out[permute[: data_num]]
            train_input = train_input[permute[: data_num]]

        # 用于筛选变量
        if index_list is not None:
            train_input = train_input[:, index_list]
            test_input = test_input[:, index_list]

        return test_label_out, test_input, train_label_out, train_input
    else:
        raise ValueError('')


def read_label(file_path):
    label_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            pat_index, side_effect_1, side_effect_2, side_effect_3, side_effect_4 = line[0: 5]
            check_numeric((side_effect_1, side_effect_2, side_effect_3, side_effect_4))
            pat_index = pat_index.strip()
            if label_dict.__contains__(pat_index):
                label_dict[pat_index].append({'side_effect_1': side_effect_1, 'side_effect_2': side_effect_2,
                                              'side_effect_3': side_effect_3, 'side_effect_4': side_effect_4})
            else:
                label_dict[pat_index] = [{'side_effect_1': side_effect_1, 'side_effect_2': side_effect_2,
                                          'side_effect_3': side_effect_3, 'side_effect_4': side_effect_4}]

    return label_dict


def read_dynamic_data(file_path):
    dynamic_data_dict = dict()
    # 读取数据
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        # 读取数据
        for line in islice(csv_reader, 1, None):
            pat_index = line[0].strip()
            dynamic_data = line[1:]
            dynamic_data[0] = datetime.datetime.strptime(dynamic_data[0], '%Y/%m/%d')
            assert len(dynamic_data) == 32
            if dynamic_data_dict.__contains__(pat_index):
                dynamic_data_dict[pat_index].append(dynamic_data)
            else:
                dynamic_data_dict[pat_index] = [dynamic_data]
        # 计算和第一次入院的时间差
        for pat_index in dynamic_data_dict:
            first_admission_time = dynamic_data_dict[pat_index][0][0]
            for item in dynamic_data_dict[pat_index]:
                admission_time = item[0]
                item[0] = (admission_time-first_admission_time).days

    for pat_id in dynamic_data_dict:
        for item in dynamic_data_dict[pat_id]:
            check_numeric(item)
    return dynamic_data_dict


def read_static_data(file_path):
    static_data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            pat_index = line[0].strip()
            static_data = line[1:]
            check_numeric(static_data)
            # 根据原始数据判断
            assert len(static_data) == 13
            static_data_dict[pat_index] = static_data
    return static_data_dict


def read_treatment(file_path):
    treatment_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            pat_index, treatment_1, treatment_2, treatment_3, treatment_4 = line[0: 5]
            check_numeric((treatment_1, treatment_2, treatment_3, treatment_4))
            pat_index = pat_index.strip()
            if treatment_dict.__contains__(pat_index):
                treatment_dict[pat_index].append({'treatment_1': treatment_1, 'treatment_2': treatment_2,
                                                  'treatment_3': treatment_3, 'treatment_4': treatment_4})
            else:
                treatment_dict[pat_index] = [{'treatment_1': treatment_1, 'treatment_2': treatment_2,
                                              'treatment_3': treatment_3, 'treatment_4': treatment_4}]
    return treatment_dict


def data_shift(dynamic, treatment, label):
    """
    由于目标是预测未来事件，因此label和动态数据需要移位对应，并丢弃最后一次数据
    :return:
    """
    new_dynamic = dict()
    new_treatment = dict()
    new_label = dict()
    for pat_id in dynamic:
        new_dynamic[pat_id] = dynamic[pat_id][: len(dynamic[pat_id])-1]
        new_treatment[pat_id] = treatment[pat_id][: len(treatment[pat_id]) - 1]
        new_label[pat_id] = label[pat_id][1:]
    return new_dynamic, new_treatment, new_label


def check_numeric(data):
    for item in data:
        float(item)
    return 0


def main():
    static_feature_path = '../resource/data_v2/baseline.csv'
    dynamic_feature_path = '../resource/data_v2/dynamic.csv'
    label_path = '../resource/data_v2/label.csv'
    treatment_path = '../resource/data_v2/treatment.csv'
    label_dict = read_label(label_path)
    dynamic_dict = read_dynamic_data(dynamic_feature_path)
    _ = read_static_data(static_feature_path)
    treatment_dict = read_treatment(treatment_path)

    _ = data_shift(dynamic_dict, treatment_dict, label_dict)
    print('accomplish')


if __name__ == '__main__':
    main()

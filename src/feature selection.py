import os
from sklearn.calibration import calibration_curve
import numpy as np
import csv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

side_effects = {'side_effect_1': '骨髓抑制', 'side_effect_3': '恶液质', 'side_effect_4': '肝功能损害'}


def read_data():
    plot_data = {}
    with open('../resource/feature_selection_procedure.csv', 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            data = line[1: 47]
            plot_data[line[0]] = [float(data_item) for data_item in data]
    return plot_data


def main():
    plot_data = read_data()

    fig, axs = plt.subplots(figsize=(8, 4))
    data_fraction = [i for i in range(1, 47)]
    for side_effect in plot_data:
        data_list = plot_data[side_effect]
        axs.plot(data_fraction, data_list, "s-", label="%s" % (side_effects[side_effect],))
        axs.set_ylabel("")
        axs.set_ylim([0.72, 0.8])
    axs.set_ylabel('平均AUC')
    axs.set_xlabel('使用特征数量')
    axs.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

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
    for side_effect in side_effects:
        with open('../resource/{}_result.csv'.format(side_effect), 'r', encoding='utf-8-sig') as f:
            plot_data[side_effect] = {}
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data = line[1: 11]
                plot_data[side_effect][line[0]] = [float(data_item) for data_item in data]
    return plot_data


def main():
    plot_data = read_data()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    data_fraction = [i*10 for i in range(1, 11)]

    idx = 0
    for side_effect in plot_data:
        for model in plot_data[side_effect]:
            data_list = plot_data[side_effect][model]
            axs[idx].plot(data_fraction, data_list, "s-", label="%s" % (model,))
            axs[idx].set_ylabel("")
            axs[idx].set_ylim([0.5, 0.8])
            axs[idx].set_title(side_effects[side_effect])
        idx += 1
    axs[2].legend(loc="lower right")
    axs[2].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[0].set_ylabel('平均AUC')
    axs[0].set_xlabel('训练数据比例（%）')
    axs[1].set_xlabel('训练数据比例（%）')
    axs[2].set_xlabel('训练数据比例（%）')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

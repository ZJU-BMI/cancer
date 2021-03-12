import os
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def main():
    path_template = '../resource/result/{}_400_{}_{}.npy'
    models = 'RF', 'MLP', 'LR', 'Adaboost'

    side_effects = {'side_effect_1': '骨髓抑制', 'side_effect_3': '恶液质', 'side_effect_4': '肝功能损害'}

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    idx = 0
    for side_effect in side_effects:
        model_dict = {}
        for model in models:
            pred = np.load(path_template.format(model, side_effect, 'pred'))
            label = np.load(path_template.format(model, side_effect, 'label'))
            prob_true, prob_pred = calibration_curve(label, pred, n_bins=10)
            model_dict[model] = prob_true, prob_pred

        axs[idx].plot([0, 1], [0, 1], "k:", label="最优校准线")
        for model in model_dict:
            prob_true, prob_pred = model_dict[model]
            axs[idx].plot(prob_pred, prob_true, "s-", label="%s" % (model,))
            axs[idx].set_ylabel("")
            axs[idx].set_ylim([-0.05, 1.05])
            axs[idx].set_title(side_effects[side_effect])
        idx += 1
    axs[2].legend(loc="lower right")
    axs[2].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[0].set_ylabel('阳性样本比例')
    axs[0].set_xlabel('期望阳性样本比例')
    axs[1].set_xlabel('期望阳性样本比例')
    axs[2].set_xlabel('期望阳性样本比例')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

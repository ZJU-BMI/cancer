import os
from sklearn.metrics import roc_curve
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
            fpr_rf, tpr_rf, _ = roc_curve(label, pred)
            model_dict[model] = fpr_rf, tpr_rf

        for model in model_dict:
            fpr_rf, tpr_rf = model_dict[model]
            axs[idx].plot(fpr_rf, tpr_rf, label=model)
            axs[idx].set_title(side_effects[side_effect])
        idx += 1
    axs[2].legend(loc="lower right")
    axs[2].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_xlabel('False Positive Rate')
    axs[1].set_xlabel('False Positive Rate')
    axs[2].set_xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn.functional import relu
import numpy as np


class PredictionModel(nn.Module):
    """
    reference: wikipedia
    """
    def __init__(self, static_input_size, dynamic_input_size, hidden_size):
        super(PredictionModel, self).__init__()

        self._static_input_size = static_input_size
        self._dynamic_input_size = dynamic_input_size
        self._hidden_size = hidden_size
        # self._LSTM = LSTM(dynamic_input_size, hidden_size, batch_first=True)
        # self._output_layer_1 = nn.Linear(static_input_size+hidden_size, 2)

        self._LSTM = LSTM(dynamic_input_size+static_input_size, hidden_size, batch_first=True)
        self._dropout = nn.Dropout(0.5)
        self._output_layer_1 = nn.Linear(hidden_size, 2)

    def forward(self, static_input, dynamic_input):
        """
        :param static_input: [batch_size, data_size]
        :param dynamic_input: [batch_size, seq_len, data_size]
        :return:
        """
        # dynamic_state, (h, c) = self._LSTM(dynamic_input)
        # static_input = static_input.unsqueeze(1).repeat(1, dynamic_state.shape[1], 1)
        # state = torch.cat((static_input, dynamic_state), dim=2)
        # output = self._output_layer_1(state)

        static_input = static_input.unsqueeze(1).repeat(1, dynamic_input.shape[1], 1)
        state = torch.cat((static_input, dynamic_input), dim=2)
        dynamic_state, (h, c) = self._LSTM(state)
        output = self._output_layer_1(self._dropout(dynamic_state))
        return output


def main():
    s_input_size = 16
    d_input_size = 12
    hidden_size = 8
    seq_len = 4
    batch_size = 7

    dynamic_input = torch.from_numpy(np.random.uniform(-1, 1, (batch_size, seq_len, d_input_size))).float()
    static_input = torch.from_numpy(np.random.uniform(-1, 1, (batch_size, s_input_size))).float()

    model = PredictionModel(s_input_size, d_input_size, hidden_size)
    prediction = model(static_input, dynamic_input)
    print(prediction)
    print("accomplished")


if __name__ == '__main__':
    main()




import torch
from torch import nn
import numpy as np
from activation import SmileTanh


class SingleOptionVolatility(nn.Module):
    def __init__(self, hidden_size_list, input_size):
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.fc_act = nn.Linear(1,  self.hidden_size_list[0])
        self.fc_smile = nn.Linear(1, self.hidden_size_list[0])

        self.mh = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.th = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.new_hidden = nn.Linear(hidden_size_list[1], hidden_size_list[2])
        self.new_hidden2 = nn.Linear(hidden_size_list[2], hidden_size_list[3])

        self.new_hidden3 = nn.Linear(hidden_size_list[3], hidden_size_list[4])
        self.last_idden3 = torch.randn(hidden_size_list[4], 1, dtype=torch.float,
                                      requires_grad=True)
        self.relu = nn.ReLU()


    def forward(self, inp):

        m = torch.reshape(
            SmileTanh()(self.fc_smile(inp[:, :, 0])), (-1, self.hidden_size_list[0]))
        tau = torch.reshape(
            torch.sigmoid(self.fc_act(inp[:, :, 1])), (-1, self.hidden_size_list[0]))
        m = self.relu(self.mh(m))
        tau = self.relu(self.th(tau))
        tm = (m*tau)
        

        etm = self.relu(self.new_hidden2(self.relu(self.new_hidden(tm))))

        etm = self.relu(self.new_hidden3(etm))
        res = etm.mm(torch.exp(self.last_idden3))

        return res

class MultiOptionVolatility(nn.Module):
    def __init__(self, input_size, model_count, single_model_hidden_size,
                 hidden_weight_size):
        super().__init__()
        self.hidden_weight_size = hidden_weight_size
        self.model_count = model_count
        self.single_model_hidden_size = single_model_hidden_size
        self.input_size = input_size
        self.fc_act = nn.Linear(1, hidden_weight_size)
        self.fc_smile = nn.Linear(1, hidden_weight_size)

        self.w_m = nn.Linear(1, hidden_weight_size)
        self.w_t = nn.Linear(1, hidden_weight_size)
        self.w_k = nn.Linear(hidden_weight_size, input_size)
        self.b_i = torch.randn(1, input_size, dtype=torch.float,
                               requires_grad=True)
        self.single_model_list = []
        for i in range(model_count):
            torch.manual_seed(i)
            self.single_model_list.append(
                SingleOptionVolatility(self.single_model_hidden_size,
                                       self.input_size))

    def single_model_weight(self, m, tau):
        a = torch.sigmoid(self.w_m(m) + self.w_t(tau))

        aa = self.w_k(a)
        aaa = torch.exp(aa)
        return aaa / torch.sum(aaa)

    def forward(self, inp):
        sum_res = 0
        output_list = []
        for model in self.single_model_list:
            single_model_output = model(inp)

            single_model_output = torch.reshape(single_model_output, (-1, 1))

            weight = self.single_model_weight(inp[:, :, 0], inp[:, :, 1])

            output = single_model_output*weight

            output_list.append(torch.reshape(single_model_output*weight, (-1, 1, 1)))
            
        output_tensor = torch.cat(output_list, dim=2)
        return torch.sum(output_tensor, dim=2)


import torch.nn as nn
import torch
import numpy as np


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size

        self.padding = 0
        self.bias = bias

        self.conv = nn.Conv1d(in_channels=2 * self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.convf = torch.nn.Conv1d(self.input_dim, 128, 1)

    def forward(self, input_tensor, cur_state):

        input_tensor = self.convf(input_tensor)

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)



        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        lenth = 280
        return (torch.zeros(batch_size, self.hidden_dim, lenth, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, lenth, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.fc1 = nn.Linear(896, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        self.act = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, input_tensor, hidden_state=None):


        b, _, _, l = input_tensor.size()


        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b)

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor


        for layer_idx in range(self.num_layers):

            batch_size = input_tensor.size(0)

            h, c = hidden_state[layer_idx]

            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                 cur_state=[h, c])
                # print(h.shape)
                
                output_result = h.view(batch_size, 128, -1, 7)
                output_result = torch.max(output_result, dim = 2, keepdim= True)[0]
                output_result = output_result.view(batch_size, 896)
                output_result = self.act(self.fc1(output_result))
                output_result = self.act(self.fc2(output_result))
                output_result = self.fc3(output_result)

                output_inner.append(output_result)

        result = torch.stack(output_inner, dim = 1)

        return result

    def _init_hidden(self, batch_size):

        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


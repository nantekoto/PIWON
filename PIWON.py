
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0, **kwargs):
        super(EncoderRNN, self).__init__( **kwargs)

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, input, *args):

        input = input.permute(1, 0, 2)
        output, state = self.gru(input)
        return output, state
    

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0, **kwargs):
        super(DecoderRNN, self).__init__(**kwargs)

        self.GRU = nn.GRU(input_size , hidden_size, n_layers,
                          dropout=dropout)

        self.dense1 = nn.Linear(hidden_size, 128)
        self.dense2 = nn.Linear(128, 1)
        #self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, input, init_state):

        input = input.repeat(1, 2, 1) # 2 times duration of rainfall
        input = input.permute(1, 0, 2)
        input = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.arange(-12,12,0.1).to(device),1),1).repeat(1,input.shape[1],1), input), dim = 2)
        #context = init_state[-1].repeat(input.shape[0], 1, 1)
        #X_and_context = torch.cat((input, context), 2)
        output, final_state = self.GRU(input, init_state)
        output = self.dense1(output).permute(1, 0, 2)
        output = self.dense2(output)

        #output = self.dropout(output)
        output = self.relu(output)


        return output, final_state
    


class BuildUp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dense1 = nn.Linear(input_size, 64)
        self.dense2 = nn.Linear(64, 128)
        #self.dropout = nn.Dropout()
        self.dense3 = nn.Linear(128, 1)
        self.act = nn.ReLU()

    def forward(self, input):

        output = self.dense1(input)
        output = self.dense2(output)
        #output = self.dropout(output)
        output = self.dense3(output)
        output = self.act(output)
        return output

class WashOffCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x, h, para1, para2):
        # define your own calculation for each time step here
        # x is the input tensor at this time step, with shape (batch_size, input_size)
        # h is the hidden state at the previous time step, with shape (batch_size, hidden_size)
        h_new = torch.zeros_like(h)
        h_new = h - (para1 * x ** para2) * h
        h_new = self.act(h_new)

        return h_new

class WashOffPara(nn.Module):
    def __init__(self, n_factor):
        super().__init__()
        self.para_dense1 = nn.Linear(n_factor, 32)
        self.para_dense2 = nn.Linear(32, 64)
        #self.dropout = nn.Dropout()
        self.para_dense3 = nn.Linear(64, 2)
        self.para_act = nn.ReLU()

    def forward(self, factor):
        output = self.para_dense1(factor)

        #output = self.dropout(output)
        output = self.para_dense2(output)
        output = self.para_act(output)
        output = torch.where(output == 0, torch.finfo().tiny, output)
        return output


class WashOff(nn.Module):
    def __init__(self):
        super().__init__()
        self.washoffcell = WashOffCell()

    def forward(self, runoff,  init_buildup, para):

        n_steps = runoff.shape[1]

        h = init_buildup  # initialize hidden state
        #print(inith)
        res = h # Record load as time step

        for i in range(n_steps):
            new_h = self.washoffcell(runoff[:,i,:], h, para[:,0].unsqueeze(-1), para[:,1].unsqueeze(-1)) # apply RNN cell to each time step
            res = torch.cat((res, new_h), 1)
            h = new_h
        return h, res
    


class Finetune(nn.Module):
    def __init__(self, input_size):
      super().__init__()
      self.fc1 = nn.Linear(input_size, 64)
      self.fc2 = nn.Linear(64, 128)
      #self.dropout = nn.Dropout()
      self.fc3 = nn.Linear(128, 1)

    def forward(self, data):
      output = self.fc1(data)
      output = self.fc2(output)
      #output = self.dropout(output)
      output = self.fc3(output)

      return output
    

class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 2, dropout = 0.1, para_flag = True):
      super().__init__()

      self.encoder = EncoderRNN(input_size = input_size + 1, hidden_size = hidden_size, n_layers = n_layers, dropout = dropout)
      self.decoder = DecoderRNN(input_size = input_size + 1, hidden_size = hidden_size, n_layers = n_layers, dropout = dropout)
      self.buildup = BuildUp(input_size = input_size)
      self.washoff = WashOff()
      self.finetune = Finetune(input_size)


      if para_flag:
        self.para_flag = para_flag
        self.para_pipeline = WashOffPara(input_size)
      else:
        self.para_flag = False


    def forward(self, test_data):
      if self.para_flag:
        para = self.para_pipeline(test_data[:,0,1:])
      else:
        para = nn.Parameter(torch.Tensor([[0.1, 1]]).repeat(test_data.shape[0], 1))


      output, state = self.encoder(test_data)
      runoff, final_state = self.decoder(test_data[:,:,1:], state)
      if runoff.isnan().any():
        print('nan runoff')
      total_runoff = runoff.sum(axis = 1)
      init_buildup = self.buildup(test_data[:,0,1:])
      if init_buildup.isnan().any():
        print('nan init_buildup')

      # print(runoff.shape, para.shape)
      final_load, load_record = self.washoff(runoff, init_buildup, para)
      # mismatch = self.finetune(test_data[:,0,1:])
      # mismatch = torch.cat((torch.zeros(test_data.shape[0], 240).to(device), mismatch), 1)

      # load_record = load_record + mismatch

      if load_record.isnan().any():
        print('nan load_record')

      return runoff/10, total_runoff/10, load_record
    



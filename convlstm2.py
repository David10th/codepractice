import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell,self).__init__()
        self.input_dim = input_dim #雪深通道数
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding =kernel_size[0]//2,kernel_size[1]//2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                              out_channels=self.hidden_dim*4,
                              kernel_size=self.padding,
                              bias = self.bias)

    def forward(self,input_tensor,cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor,h_cur],dim = 1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self,batch_size,image_size):
        latitude,longitude = image_size
        return (torch.zeros(batch_size, self.hidden_dim, latitude, longitude, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, latitude, longitude, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self,input_dim,hidden_dim,kernel_size,bias=True):
        super(ConvLSTM,self).__init__()
        self.cell = ConvLSTMCell(input_dim,hidden_dim,kernel_size,bias)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.snow_out = nn.Conv2d(in_channels = hidden_dim,
                                  out_channels = input_dim,
                                  kernel_size = 1,
                                  bias = bias)

    def forward(self,x):
        #输入：(time, batch_size, latitude, longitude, snow_depth)
        #输出：(time, batch_size, latitude, longitude, snow_depth)
        time_steps, batch_size, latitude, longitude, _ = x.size()
        h, c = self.cell.init_hidden(batch_size, (latitude, longitude))
        outputs = []
        for t in range(time_steps):
            input_t = x[t]  # (batch_size, latitude, longitude, snow_depth)
            input_t = input_t.permute(0, 3, 1, 2)  # (batch_size, snow_depth, latitude, longitude)
            h, c = self.cell(input_t, (h, c))
            output_t = self.snow_out(h)  # 雪深
            output_t = output_t.permute(0, 2, 3, 1)  # (batch_size, latitude, longitude, snow_depth)
            outputs.append(output_t)

        outputs = torch.stack(outputs)  # (time, batch_size, latitude, longitude, snow_depth)
        return outputs
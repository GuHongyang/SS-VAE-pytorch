import torch
import torch.nn as nn
import torch.nn.functional as F



class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid,self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

class EXP(nn.Module):
    def __init__(self):
        super(EXP,self).__init__()

    def forward(self, x):
        return torch.exp(x)

class MLP(nn.Module):
    def __init__(self,dim_input,dim_output,hidden_layers=[500],hidden_nonlinearity=nn.Softplus(),output_nonlinearity=[None]):
        super(MLP,self).__init__()

        self.output_num=len(output_nonlinearity)

        layers=[]
        pre_dim=dim_input
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(pre_dim,hidden_layers[i]))
            layers.append(hidden_nonlinearity)
            pre_dim=hidden_layers[i]

        self.base=nn.Sequential(*layers)


        self.output_layers=nn.ModuleList()
        for i in range(self.output_num):
            if output_nonlinearity[i] is None:
                self.output_layers.append(
                    nn.Sequential(
                        nn.Linear(pre_dim,dim_output),
                    )
                )
            else:
                self.output_layers.append(
                    nn.Sequential(
                        nn.Linear(pre_dim, dim_output),
                        output_nonlinearity[i]
                    )
                )


    def forward(self, x):
        f0=self.base(x)

        outputs=[]
        for i in range(self.output_num):
            outputs.append(self.output_layers[i](f0))

        return outputs


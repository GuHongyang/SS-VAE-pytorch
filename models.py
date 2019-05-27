import torch
import torch.nn as nn
from modules import *
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable


def log_Gaussian_pro(x,mean,var):
    d=x.size(1)

    return -d/2*np.log(2*np.pi)-1/2*torch.sum(torch.log(var+1e-10),1)-1/2*torch.sum((x-mean)**2/var,1)

class M2(nn.Module):
    def __init__(self,alpha=0.1*200):
        super(M2,self).__init__()

        self.encode_y=MLP(dim_input=784,dim_output=10,hidden_layers=[500],output_nonlinearity=[None])

        self.encode_z=MLP(dim_input=10+784,dim_output=50,hidden_layers=[500],output_nonlinearity=[None,EXP()])

        self.decoder=MLP(dim_input=10+50,dim_output=784,hidden_layers=[500],output_nonlinearity=[None])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


        self.alpha=alpha

    def forward(self, lx,ly,ux):

        #labeled data
        z_mu,z_var=self.encode_z(torch.cat([lx,ly],1))  # bl * 50
        #z~ q(z|x,y)
        z=torch.randn_like(z_mu)*torch.sqrt(z_var)+z_mu # bl * 50
        log_q_z=log_Gaussian_pro(z,z_mu,z_var) # bl
        #x ~ p(x|y,z)
        x=self.decoder(torch.cat([z,ly],1))[0] # bl * 784
        log_p_x=-torch.sum(F.binary_cross_entropy_with_logits(x,lx,reduction='none'),1) # bl
        log_p_y=np.log(0.1)
        log_p_z=log_Gaussian_pro(z,torch.zeros_like(z),torch.ones_like(z)) # bl
        Labled_loss=torch.mean(log_q_z-log_p_x-log_p_y-log_p_z)

        #supervised learning
        y_=self.encode_y(lx)[0] # bl * 10

        Sup_loss=self.alpha*F.cross_entropy(y_,torch.argmax(ly,1),reduction='mean')

        #unlabeled data
        uq_y=F.softmax(self.encode_y(ux)[0],dim=-1) # bu *10
        Unlabled_loss=0

        uy_=torch.Tensor(ux.size(0),10).cuda()
        for i in range(10):
            # uy=Variable(torch.zeros(1,10).scatter_(1,torch.Tensor([[i]]).long(),1).repeat(ux.size(0),1).cuda(),requires_grad=False) # bu * 10
            uy=torch.zeros_like(uy_)
            uy[:,i]=1

            uz_mu,uz_var=self.encode_z(torch.cat([ux,uy],1)) # bu * 50
            #z ~ q(z|x,y)
            uz=torch.randn_like(uz_mu)*torch.sqrt(uz_var)+uz_mu # bu * 50
            ulog_q_z=log_Gaussian_pro(uz,uz_mu,uz_var) # bu
            # x ~ p(x|y,z)
            xx=self.decoder(torch.cat([uz,uy],1))[0] # bu * 784
            ulog_p_x=-torch.sum(F.binary_cross_entropy_with_logits(xx,ux,reduction='none'),1) # bu
            ulog_p_y=np.log(0.1)
            ulog_p_z=log_Gaussian_pro(uz,torch.zeros_like(uz),torch.ones_like(uz)) # bu
            Unlabled_loss = Unlabled_loss + (ulog_q_z-ulog_p_x-ulog_p_y-ulog_p_z)*uq_y[:,i] + uq_y[:,i] * torch.log(uq_y[:,i]+1e-10)

        Unlabled_loss=torch.mean(Unlabled_loss)

        Loss=Labled_loss+Unlabled_loss+Sup_loss

        return Loss, Labled_loss, Unlabled_loss, Sup_loss


    def predict(self,x):
        return torch.argmax(self.encode_y(x)[0],1)
            




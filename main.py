from data import get_dataloader
from models import M2
from tqdm import tqdm
import torch
import os
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import MultiStepLR
os.environ['CUDA_VISIBLE_DEVICES']='5'




batch_size=200


model=M2(alpha=0.1)
model=model.cuda()
opti=Adam(model.parameters(),lr=1e-3)
lr=MultiStepLR(opti,milestones=[100],gamma=0.1)
dataloader=get_dataloader(num_training=50000,num_labeled=3000,batch_size=batch_size)


epoch_bar=tqdm(range(500))
for epoch in epoch_bar:
    Loss=0
    L_loss=0
    U_loss=0
    S_loss=0
    model.train()
    lr.step()
    batch_bar=tqdm(zip(dataloader['labeled'],dataloader['unlabeled']))
    for label_batch,unlabel_batch in batch_bar:


        lx,ly=label_batch
        lx=lx.cuda()
        ly=ly.cuda()

        ux,uy=unlabel_batch
        ux=ux.cuda()
        uy=uy.cuda()

        loss, L_loss_mean, U_loss_mean, S_loss_=model(lx, ly, ux)
        Loss+=loss
        L_loss+=L_loss_mean
        U_loss+=U_loss_mean
        S_loss+=S_loss_

        opti.zero_grad()
        loss.backward()
        opti.step()

        batch_bar.set_description('[Loss={:.4f}], [L_Loss={:.4f}], [U_Loss={:.4f}], [S_Loss={:.4f}]'.format(loss,
                                                                                                            L_loss_mean,
                                                                                                            U_loss_mean,
                                                                                                               S_loss_))
    epoch_bar.set_description('[Loss={:.4f}], [L_Loss={:.4f}], [U_Loss={:.4f}], [S_Loss={:.4f}]'.format(Loss/len(dataloader['labeled']),
                                                                                                        L_loss/len(dataloader['labeled']),
                                                                                                        U_loss/len(dataloader['labeled']),
                                                                                                        S_loss/len(dataloader['labeled'])))

    model.eval()
    acc = 0
    N=0
    batch_bar = tqdm(dataloader['test'])
    with torch.no_grad():
        for label_batch in batch_bar:
            lx,ly=label_batch
            x=lx.cuda()
            y=ly.cuda()

            y_=model.predict(x)

            acc+=torch.nonzero(y==y_).size(0)
            N+=y_.size(0)
        epoch_bar.write('[ACC={:.4f}%]'.format(acc/N*100))





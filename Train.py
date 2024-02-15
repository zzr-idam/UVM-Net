import os
import torch
from unet_model import UNet 
import dataset
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
#from kornia.filters import laplacian

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train(args):

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3)
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])  
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    
    mse = nn.L1Loss().cuda()
    
    content_folder1 = 'test/hazy'
    information_folder = 'test/GT'
    
    train_loader = dataset.style_loader(content_folder1, information_folder, args.size, 8)
    
    num_batch = len(train_loader)
    for epoch in range(args.epoch):
      for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            total_iter = epoch*num_batch  + idx
               
            content = batch[0].float().cuda()
            information = batch[1].float().cuda()
            
            optimizer.zero_grad()
 
            #content = torch.exp(content)
            
            output = model(content)
             
                     
            total_loss =  mse(output , information) 

            total_loss.backward()
            

            optimizer.step()


            
            if np.mod(total_iter+1, 1) == 0:
                print('{}, Epoch:{} Iter:{} total loss: {}'.format(args.save_dir, epoch, total_iter, total_loss.item()))
                
                
                
                
            if not os.path.exists(args.save_dir+'/image'):
                os.mkdir(args.save_dir+'/image')

      if epoch % 5 ==0:
        #content = torch.log(content)
        #output = torch.log(output)
        out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
        save_image(out_image, args.save_dir+'/image/iter{}_1.jpg'.format(total_iter+1))
        torch.save(model.state_dict(), 'model' +'/our_dehaze{}.pth'.format(epoch))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    train(args)

import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset




def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' or '.JPG' or '.png' or '.PNG' in name]
    name_list.sort()
    return name_list


class StyleDataset(Dataset):
    def __init__(self, content1, information ,size):
        self.content1_list = content1

        self.label = information
        self.size = 512
        self.len = len(self.content1_list)

    def __getitem__(self, index):

        c1_path = self.content1_list[index]
        

        f_path = self.label[index]

        
        content = cv2.imread(c1_path)[:, :, ::-1]
        
        information = cv2.imread(f_path)[:, :, ::-1]
        
        
        try:

            content = cv2.resize(content, (512, 512))

            
        except:
            content = cv2.resize(content, (512, 512))

            
        try:

            information = cv2.resize(information, (512, 512))
        except:
            information = cv2.resize(information, (512, 512))
        
        
        content = content.transpose((2, 0, 1))/255.0

        
 
        information = information.transpose((2, 0, 1))/255.0
        
        return content,  information 

    def __len__(self):
        return self.len




    

def style_loader(content1_folder,  information_folder, size, batch_size):
    content_list1 = load_simple_list(content1_folder)

    
    information_list = load_simple_list(information_folder)
    

    dataset = StyleDataset(content_list1,  information_list, size)
    num_workers = 8 if batch_size > 8 else batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    return dataloader
    

if __name__ == '__main__':
    pass
        
    


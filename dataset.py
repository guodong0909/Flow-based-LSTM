import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from function import quaternion2rotation, rotation2quaternion


class DFdata(Dataset):
    def __init__(self, list_file, img_file):
        

        dataDir= os.path.split(list_file)[0]

        with open(list_file, 'r') as f:
            self.list_files = f.readlines()
            self.gr_list = [dataDir + '/gr_pose/' + x.strip() for x in self.list_files]
            self.mask_list = [dataDir + '/mask/' + x.strip() for x in self.list_files]
        with open(img_file, 'r') as f:
            self.img_files = f.readlines() 
            self.img_list = [dataDir + '/img/' + x.strip() for x in self.img_files]


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        prev = cv2.imread(self.img_list[index])
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)#image1
        after = cv2.imread(self.img_list[index+1])
        after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)#image2

        inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(prev, after, None)#flow of image1&image2

        # flow = cv2.calcOpticalFlowFarneback(prev=prevgray, next=gray, flow=None, pyr_scale=0.5, levels=5,
        #                                         winsize=15,
        #                                         iterations=3, poly_n=3, poly_sigma=1.2,
        #                                         flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)        #Farneback
                                            
        # flow = cv2.optflow.calcOpticalFlowSparseToDense(prevgray, gray)  #SparseToDense

        #flow = cv2.optflow.calcOpticalFlowSF(prevgray, gray, 2, 2, 4)  #simpleflow

        # inst = cv2.optflow.createOptFlow_PCAFlow()
        # flow = inst.calc(prevgray, gray, None) #pcaflow

        prev_gr_qt = torch.from_numpy(np.load(self.gr_list[index])).view(1,7)
        after_gr_qt = torch.from_numpy(np.load(self.gr_list[index+1])).view(1,7)

        prev_gr_q = prev_gr_qt[:, 0:4]
        after_gr_q = after_gr_qt[:, 0:4]
        prev_gr_t = prev_gr_qt[:, 4:7].T
        after_gr_t = after_gr_qt[:, 4:7].T

        prev_gr_r = quaternion2rotation(prev_gr_q).view(3, 3)
        after_gr_r = quaternion2rotation(after_gr_q).view(3, 3)

        delta_r = torch.mm(after_gr_r, torch.inverse(prev_gr_r))


        delta_t = after_gr_t - torch.mm(delta_r, prev_gr_t)

        mask = np.load(self.mask_list[index+1]).astype(np.float32)  # from FCN and deepflow

        flow = np.transpose(flow, (2, 1, 0))
        
        return flow, delta_r, delta_t, mask
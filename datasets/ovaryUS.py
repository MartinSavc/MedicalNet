'''
Dataset 
Written by Savc
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
from scipy import ndimage
import h5py

class OvaryUSDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        self.data = h5py.File(img_list, 'r')
        print("Processing {} datas".format(len(self.data['Images'])))
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.random_flip = sets.random_flip
        self.phase = sets.phase

    def __del__(self):
        #self.data.close()
        pass

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.data['Images'])

    def original_size(self, idx):
        return self.data[f'Images/{idx+1}'].shape

    def __getitem__(self, idx):

        
        if self.phase == "train":
            img = self.data[f'Images/{idx+1}']
            mask = self.data[f'Labels/{idx+1}']

            ## data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            ## 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, mask_array
        
        elif self.phase == "test":
            img = self.data[f'Images/{idx+1}']
            if 'Labels' in self.data:
                mask = self.data[f'Labels/{idx+1}']
            else:
                mask = np.zeros(img.shape)

            ## data processing
            img_array, mask_array = self.__testing_data_process__(img, mask)

            ## 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            return img_array, mask_array
            

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label>0)
        img_d, img_h, img_w = data.shape
        in_d, in_h, in_w = self.input_D, self.input_H, self.input_W

        data_cropped = np.zeros((in_d, in_h, in_w))
        label_cropped = np.zeros((in_d, in_h, in_w))

        Z_min = int(max(img_d-in_d, 0)*random()+0.5)
        Y_min  = int(max(img_h-in_h, 0)*random()+0.5)
        X_min = int(max(img_w-in_w, 0)*random()+0.5)

        Z_max = min(Z_min+in_d, img_d)
        Y_max = min(Y_min+in_h, img_h)
        X_max = min(X_min+in_w, img_w)

        d = Z_max-Z_min
        h = Y_max-Y_min
        w = X_max-X_min

        data_cropped[:d, :h, :w] = data[Z_min:Z_max, Y_min:Y_max, X_min:X_max]
        label_cropped[:d, :h, :w] = label[Z_min:Z_max, Y_min:Y_max, X_min:X_max]

        return data_cropped, label_cropped

    def __random_flip__(self, data, label):
        from random import random
        flip_axis = []
        if random() > 0.5:
            flip_axis.append(0)
        if random() > 0.5:
            flip_axis.append(1)
        if random() > 0.5:
            flip_axis.append(2)

        if len(flip_axis) != 0:
            data = np.flip(data, flip_axis)
            label = np.flip(label, flip_axis)

        return data, label



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        
        return data, label

    def __pad_data__(self, data, label):
        """
        Pad data to have dimension a multiple of 4.
        """
        d, h, w = data.shape

        d_pad = int(np.ceil(d/4)*4)-d
        h_pad = int(np.ceil(h/4)*4)-h
        w_pad = int(np.ceil(w/4)*4)-w
        
        if d_pad>0 or h_pad>0 or w_pad>0:
            data = np.pad(data, ((0, d_pad), (0, h_pad), (0, w_pad)))
            label = np.pad(label, ((0, d_pad), (0, h_pad), (0, w_pad)))

        return data, label

    def __training_data_process__(self, data, label): 
        # crop data according net input size
        data = np.array(data)
        label = np.array(label)
        
        # crop data
        data, label = self.__crop_data__(data, label) 
        if self.random_flip:
            data, label = self.__random_flip__(data, label)
        data, label = self.__pad_data__(data, label)


        # resize data
        #data = self.__resize_data__(data)
        #label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label


    def __testing_data_process__(self, data, label): 
        # crop data according net input size
        data = np.array(data)
        label = np.array(label)

        data, label = self.__pad_data__(data, label)
        # resize data
        #data = self.__resize_data__(data)
        #label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label

import torch.utils.data as data
import os
from .data_utils import *
import torchio as tio
import torch
import numpy as np


class mae_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'
        all_img = 0
        candidate_dir = list_mae_domains(cfg.data.mae_root)

        self.path_dic = {}
        for i in range(len(candidate_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(candidate_dir[i], self.cfg.data.extension))
            all_img += len(self.path_dic[str(i)])
        self.num_domain = len(candidate_dir)
        print('num of mae domains: '+str(self.num_domain))
        print('num of mae scans: ' + str(all_img))

        self.all_img = all_img

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        index = np.random.randint(0, len(tmp_path))
        # index = index % len(tmp_path)
        # scans should be first warped to the RAS space
        # see preprocessing for more details
        tmp_scans = nib.load(tmp_path[index])
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        tmp_scans[tmp_scans < 0] = 0
        tmp_scans = random_flip(tmp_scans)

        # normalization
        # we do a little trick to normalize the scan at random percentile
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100-self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc-perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)

        # whether to pad the image to match the patch size
        # and then cast to torch.tensor
        # print(tmp_scans.shape)
        x, y, z = self.cfg.data.patch_size
        pad_h, pad_w = max(0, x - tmp_scans.shape[0]), max(0, y - tmp_scans.shape[1])
        # print(pad_h, pad_w)
        # if min(tmp_scans.shape) < min(x, y, z):
        #     x_diff = x-tmp_scans.shape[0]
        #     y_diff = y-tmp_scans.shape[1]
        #     z_diff = z-tmp_scans.shape[2]
        #     tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
        #         y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location
        #     tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        # else:
        #     tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        if pad_h > 0 or pad_w > 0:
            tmp_scans = np.pad(tmp_scans, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0),), 
                         constant_values=1e-4)  # Avoid zero-padding
        tmp_scans = np.expand_dims(tmp_scans, axis=-1)

        tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        # print(tmp_scans.shape)
        # if len(tmp_scans.shape) == 3:
        # # Add a missing dimension (e.g., batch/channel dimension)
        #     tmp_scans = tmp_scans.unsqueeze(0)  # Adds a dimension at index 0
        if len(tmp_scans.shape) != 4:
            return self.__getitem__((index + 1) % len(self))  # Try next valid sample


        _, x1, y1, z1 = tmp_scans.shape
        # _, x1, y1 = tmp_scans.shape
        transforms = tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.75, 1.5), degrees=40,
                                      isotropic=False,
                                      default_pad_value=0, image_interpolation='linear')
        tmp_scans = transforms(tmp_scans)

        # if remove_bg, the patch will only be sampled from the foreground (non-zero) region

        bound = get_bounds(tmp_scans.data)
        x1, y1, _ = tmp_scans.shape[1:]  # Get updated shape

        if self.cfg.data.remove_bg:
            x_idx = max(0, min(np.random.randint(bound[0], bound[1] - x), x1 - x))
            y_idx = max(0, min(np.random.randint(bound[2], bound[3] - y), y1 - y))
        else:
            x_idx = np.random.randint(0, x1 - x) if x1 - x > 0 else 0
            y_idx = np.random.randint(0, y1 - y) if y1 - y > 0 else 0

        # location indicates the sampled patch location
        location = torch.zeros_like(tmp_scans.data)
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, ] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3], :]))
        transforms = tio.transforms.Resize(target_shape=(x, y, 1))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data[:,:, :,0]
        # print(down_scan.shape)

        input_dict = {'local_patch': tmp_scans.data[:, x_idx:x_idx + x, y_idx:y_idx + y, 0],
                      'global_images': down_scan}


        return input_dict

    def __len__(self):
        # we used fixed 2000 as the number of samples in each epoch
        # one can choose max(2000, self.all_img)
        # return max(2000, self.all_img)
        return 2000


class mpl_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'

        # data from target domain, only img (folder name should end with '_train')

        tgt_dir, src_dir1 = list_finetune_domains(
            cfg.data.tgt_data, cfg.data.src_data)

        self.path_dic = {}
        for i in range(len(tgt_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(tgt_dir[i], self.cfg.data.extension))
        self.num_domain = len(tgt_dir)

        # data from source domain,  img + label (folder name should end with '_img' for img and '_label' for label)

        self.path_dic_B1 = {}
        self.path_dic_B2 = {}
        for i in range(len(src_dir1)):
            self.path_dic_B1[str(i)] = sorted(
                list_scans(src_dir1[i], self.cfg.data.extension))
            self.path_dic_B2[str(i)] = [i.replace(
                '_img', '_label') for i in self.path_dic_B1[str(i)]]
            self.path_dic_B2[str(i)] = [i.replace(
                '_te1_split', '_ref_mask_split') for i in self.path_dic_B2[str(i)]]

        self.num_domain_B = len(src_dir1)

        print('num of target domain: ' + str(self.num_domain))
        print('num of source domain: ' + str(self.num_domain_B))

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        indexA = np.random.randint(0, len(tmp_path))

        idx = int(np.random.random_sample() // (1 / self.num_domain_B))
        tmp_path_B1 = self.path_dic_B1[str(idx)]
        tmp_path_B2 = self.path_dic_B2[str(idx)]

        indexB = np.random.randint(0, len(tmp_path_B1))
        x, y, z = self.cfg.data.patch_size
        '''
        getitem for training/validation
        '''

        '''
        load non-labeled data
        '''
        tmp_scansA = nib.load(tmp_path[indexA])
        # print("tmp_scansA shape: ", tmp_scansA.get_fdata().shape)
        tmp_scansA = np.squeeze(tmp_scansA.get_fdata())
        tmp_scansA[tmp_scansA < 0] = 0

        # normalization
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scansA = norm_img(tmp_scansA, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scansA = norm_img(tmp_scansA, self.cfg.data.norm_perc)
        # padding
        pad_h, pad_w = max(0, x - tmp_scansA.shape[0]), max(0, y - tmp_scansA.shape[1])
        # print("shape of tmp_scansA: ", tmp_scansA.shape)
        if pad_h > 0 or pad_w > 0:
            
            tmp_scansA = np.pad(tmp_scansA, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0)), 
                         constant_values=1e-4)  # Avoid zero-padding
        tmp_scansA = np.expand_dims(tmp_scansA, axis=-1)
        tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        if len(tmp_scansA.shape) != 4:
            return self.__getitem__((index + 1) % len(self))  # Try next valid sample        
        _, x1, y1, z1 = tmp_scansA.shape
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.3), degrees=30,
                                                       isotropic=False,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest')
                                      ])
            # tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = transforms(tmp_scansA)
        else:
            # tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = tmp_scansA
        # randomly select patch
        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)
        else:
            bound = [0, x1, 0, y1]
            x_idx = 0 if x1 - x == 0 else np.random.randint(0, x1 - x)
        #     y_idx = 0 if y1 - y == 0 else np.random.randint(0, y1 - y)
        # print('tmp_scans shape (A): ', tmp_scans.data.shape)
        # print('bound (A): ', bound)
            
        location = torch.zeros_like(tmp_scans.data).float()
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, ] = 1
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],:]), a_segmentation=tio.LabelMap(
            tensor=location[:, bound[0]:bound[1],
                            bound[2]:bound[3], :]
        ))
        transforms = tio.transforms.Resize(target_shape=(x, y, 1))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data[:,:,:,0]
        # print('sbj shape (A): ', sbj['one_image'].data.shape)
        locA = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locA)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),

                                  ]).astype(int)

        patchA = tmp_scans.data[:, x_idx:x_idx + x,
                                y_idx:y_idx + y,0].float()
        downA = down_scan.float()

        '''
        load annotated data
        '''

        tmp_scans = nib.load(tmp_path_B1[indexB])
        # print("tmp_scansB1 shape: ", tmp_scans.get_fdata().shape)
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        '''
        WARNING: HERE WE ONLY USE POSITIVE INTENSITY 
        FOR CT, USE PREPROCESSING TO turn negatives to positives 
        
        '''
        tmp_scans[tmp_scans < 0] = 0
        # print("tmp_scansB2 shape: ", nib.load(tmp_path_B2[indexB]).get_fdata().shape)
        tmp_label = np.squeeze(
            np.round(nib.load(tmp_path_B2[indexB]).get_fdata()))
        assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'

        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)
        
        pad_h, pad_w = max(0, x - tmp_scans.shape[0]), max(0, y - tmp_scans.shape[1])
        if pad_h > 0 or pad_w > 0:
            tmp_scans = np.pad(tmp_scans, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0),), 
                         constant_values=1e-4)  # Avoid zero-padding
            tmp_label = np.pad(tmp_label, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0)), 
                         constant_values=1e-4)  # Avoid zero-padding
        tmp_scans = np.expand_dims(tmp_scans, axis=-1)
        tmp_label = np.expand_dims(tmp_label, axis=-1)            
        tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        tmp_label = torch.unsqueeze(torch.from_numpy(tmp_label), 0)        

        if len(tmp_scans.shape) != 4:
            return self.__getitem__((index + 1) % len(self))  # Try next valid sample

        _, x1, y1, z1 = tmp_scans.shape
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_label = tio.LabelMap(tensor=tmp_label)
        sbj = tio.Subject(one_image=tmp_scans, a_segmentation=tmp_label)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.4), degrees=30,
                                                       isotropic=False,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest'),
                                      tio.RandomBiasField(
                                      p=self.cfg.data.aug_prob),
                                      tio.RandomGamma(
                                      p=self.cfg.data.aug_prob, log_gamma=(-0.4, 0.4))
                                      ])
            sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

        else:
            bound = [0, x1, 0, y1]
            x_idx = 0 if x1 - x == 0 else np.random.randint(0, x1 - x)
            y_idx = 0 if y1 - y == 0 else np.random.randint(0, y1 - y)
        # print('tmp_scans shape (B): ', tmp_scans.data.shape)
        # print('bound (B): ', bound)
        location_B = torch.zeros_like(tmp_scans.data).float()
        location_B[:, x_idx:x_idx + x,
                   y_idx:y_idx + y, ] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],:]),
                          a_segmentation=tio.LabelMap(
            tensor=location_B[:, bound[0]:bound[1], bound[2]:bound[3], :])
        )
        transforms = tio.transforms.Resize(target_shape=(x, y, 1))
        sbj = transforms(sbj)
        # print('sbj shape (B): ', sbj['one_image'].data.shape)
        down_scan = sbj['one_image'].data[:,:,:,0].float()
        locB = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locB)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],:]),
                          a_segmentation=tio.LabelMap(
            tensor=tmp_label[:, bound[0]:bound[1], bound[2]:bound[3], :])
        )
        sbj = transforms(sbj)
        aux_label = sbj['a_segmentation'].data[:,:,:,0] 

        coordinates_B = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                ]).astype(int)
        
        input_dict = {'imgB': tmp_scans[:, x_idx:x_idx + x, y_idx:y_idx + y, 0],
                      'labelB': torch.squeeze(tmp_label[:, x_idx:x_idx + x, y_idx:y_idx + y, 0]),
                      'label_B_aux': torch.squeeze(aux_label),
                      'downB': down_scan,
                      'cord_B': coordinates_B,
                      'imgA': patchA,
                      'downA': downA,
                      'cord_A': coordinates_A}
        # print('imgB shape: ', input_dict['imgB'].shape)
        # print('labelB shape: ', input_dict['labelB'].shape)
        # print('downB shape: ', input_dict['downB'].shape)
        # print('cord_B shape: ', input_dict['cord_B'].shape)
        # print('imgA shape: ', input_dict['imgA'].shape)
        # print('downA shape: ', input_dict['downA'].shape)
        # print('cord_A shape: ', input_dict['cord_A'].shape)

        return input_dict

    def __len__(self):

        # we used fixed 100 steps for each epoch in finetuning
        # THIS PARAM WAS NEVER TUNED
        return 100

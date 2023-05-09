# Import libraries
import torch, cv2, glob, os, shutil, random, numpy as np, scipy.io as io
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

im_dim = 128
class SHHA(Dataset):
    def __init__(self, data_root, transform = None, train = False, patch = False, flip = False, im_dim = im_dim):
        
        self.root_path, self.transform, self.train, self.patch, self.flip, self.im_dim = data_root, transform, train, patch, flip, im_dim
        
        # Get train images and gts
        if train:
            self.im_paths = sorted(glob.glob("/home/ubuntu/workspace/bekhzod/DM-Count/datasets/ShanghaiTech/*/train_data/*/*.jpg"))
            self.gt_paths = sorted(glob.glob("/home/ubuntu/workspace/bekhzod/DM-Count/datasets/ShanghaiTech/*/train_data/*/*.mat"))
            print(f"There are {len(self.im_paths)} number of images in train dataset")
        
        # Get validation images and gts
        else:
            self.im_paths = sorted(glob.glob("/home/ubuntu/workspace/bekhzod/DM-Count/datasets/ShanghaiTech/*/test_data/*/*.jpg"))
            self.gt_paths = sorted(glob.glob("/home/ubuntu/workspace/bekhzod/DM-Count/datasets/ShanghaiTech/*/test_data/*/*.mat"))
            print(f"There are {len(self.im_paths)} number of images in validation dataset")

    # Get length of the images in the dataset
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, index):
        
        """
        
        This function gets an index and returns metadata information.
        
        Parameter:
        
            index   - index of the dataset, int.
            
        Outputs:
            
            img     - an image, tensor;
            target  - corresponding ground information of the image, dict. 
        
        """
        
        # Assertion
        assert index <= len(self), "index range error"

        # Get image and gt paths
        img_path, gt_path = self.im_paths[index], self.gt_paths[index]
        
        # Get image and gt
        img, point = load_data((img_path, gt_path), self.train)
            
        # Ppply augumentations
        if self.transform is not None: img = self.transform(img)
            
        # Upsample images and points if their dimensions are smaller than the cropped image dimension
        if img.shape[1] < im_dim or img.shape[2] < im_dim:
            
            img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor = 2).squeeze(0)
            point *= 2

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # Scale the image and points
            if scale * min_size > self.im_dim:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        
        # Random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point): point[i] = torch.Tensor(point[i])
        
        # Random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # Random flip
            img = torch.Tensor(img[:, :, ::-1].copy())
            for i, _ in enumerate(point): point[i][:, 0] = self.im_dim - point[i][:, 0]

        if not self.train: point = [point]

        # Pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
            
        return img, target

def load_data(img_gt_path, train):
    
    """
    
    This function gets several parameters and loads the data.
    
    Parameters:
    
        img_gt_path     - path to ground truth data, str;
        train           - train option, bool.
        
    Outputs:
    
        img             - an output image, PIL Image object;
        points          - points to be detected, array.
    
    """
    
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    coords = io.loadmat(gt_path)["image_info"][0][0][0][0][0]
    points = []
    for coord in coords:
        points.append(coord)
    # print(np.array(points).shape)
    return img, np.array(points)

# random crop augumentation
def random_crop(img, den, num_patch=2):
    half_h = im_dim
    half_w = im_dim
    result_img = torch.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        # print(img.size(1) - half_h)
        # print(img.size(2) - half_w)
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)
        
    return result_img, result_den

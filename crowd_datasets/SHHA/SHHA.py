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
    
    # Get image and gt paths
    img_path, gt_path = img_gt_path
    
    # Load a image from the path
    img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    
    # Load ground truth points
    coords = io.loadmat(gt_path)["image_info"][0][0][0][0][0]
    
    # Initialize list to store gt points
    points = [coord for coord in coords]
    
    return img, np.array(points)

def random_crop(img, den, num_patch = 2):
    
    """
    
    This function gets several parameters and performs random crop operation.
    
    Parameters:
    
        img          - image to be cropped, tensor;
        den          - ground truth points, array;
        num_patch    - number of patches to be cropped, int.
        
    Outputs:
    
        result_img   - cropped image, tensor
        result_den   - gt points of the cropped image, list.
    
    """
    
    # Get dimensions to crop an input image
    half_h, half_w = im_dim, im_dim
    
    # Create an image filled with zeros
    result_img = torch.zeros([num_patch, img.shape[0], half_h, half_w])
    # Create a list to store gt points after cropping
    result_den = []
    
    # Crop num_patch for each image
    for i in range(num_patch):
        
        # Get start dimensions to crop
        start_h, start_w = random.randint(0, img.size(1) - half_h), random.randint(0, img.size(2) - half_w)
        
        # Get end dimensions to crop
        end_h, end_w = start_h + half_h, start_w + half_w
        
        # Copy the cropped part
        result_img[i] = img[:, start_h : end_h, start_w : end_w]
        
        # Copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        
        # Shift the coordinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        # Add the cropped gt points to the list
        result_den.append(record_den)
        
    return result_img, result_den

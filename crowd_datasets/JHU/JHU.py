# Import libraries
import torch, cv2, glob, os, random, numpy as np, scipy.io as io
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

def pp(var_name, var, shape = False):
    
    """
    
    This function gets several parameters and prints the variable metadata.
    
    
    """
    if shape:
        print(f"{var_name} -> {var.shape}\n")        
    else:
        print(f"{var_name} -> {var}\n")

im_dim = 200
class JHU(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, im_dim=im_dim):
        
        self.root_path = data_root
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        self.im_dim = im_dim
        folder = "train" if train else "val"
        broken = ["1564", "1495"]
        
        # there may exist multiple list files
        if train:
            self.im_paths = sorted(glob.glob(f"{data_root}/{folder}/images/*.jpg"))
            self.gt_paths = sorted(glob.glob(f"{data_root}/{folder}/gt/*.txt"))
            self.im_paths = [fname for fname in self.im_paths if os.path.splitext(os.path.basename(fname))[0] not in broken]
            self.gt_paths = [fname for fname in self.gt_paths if os.path.splitext(os.path.basename(fname))[0] not in broken]
            print(f"There are {len(self.im_paths)} number of images in train dataset")
        else:
            self.im_paths = sorted(glob.glob(f"{data_root}/{folder}/images/*.jpg"))
            self.gt_paths = sorted(glob.glob(f"{data_root}/{folder}/gt/*.txt"))
            print(f"There are {len(self.im_paths)} number of images in validation dataset")
        
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, index):
        
        assert index <= len(self), 'index range error'

        img_path = self.im_paths[index]
        gt_path = self.gt_paths[index]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
            
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)
            
        if img.shape[1] < im_dim or img.shape[2] < im_dim:
            
            # print("YESSSSSSS SMALLER THAN IM_DIM")
            img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=2).squeeze(0)
            point *= 2

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > self.im_dim:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = self.im_dim - point[i][:, 0]

        if not self.train:
            point = [point]

        # pack up related infos
        img = torch.Tensor(img)
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

#         pp("img", img, True)
#         pp("point", target[0]['point'], True)
#         pp("image_id", target[0]['image_id'], True)
#         pp("labels", target[0]['labels'], True)
        
#         pp("point1", target[1]['point'], True)
#         pp("image_id1", target[1]['image_id'], True)
#         pp("labels1", target[1]['labels'], True)
        
#         pp("point2", target[2]['point'], True)
#         pp("image_id2", target[2]['image_id'], True)
#         pp("labels2", target[2]['labels'], True)
        
        # (patch_num, 3, im_w, im_h); (patch_num, dict)
        return img, target

def load_data(img_gt_path, train):
    
    img_path, gt_path = img_gt_path
    # load the images
    img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    
    # load ground truth points
    with open(gt_path, 'r') as f:
        coords = [line.split('\n')[0] for line in f.readlines()]
    new_coords = []
    for coord in coords:
        new_coords.append([float(split) for split in coord.split(" ")[:2]])
    return img, np.array(new_coords)

# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = im_dim
    half_w = im_dim
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
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

# root = "/home/ubuntu/workspace/bekhzod/DM-Count/datasets/jhu_crowd_v2.0"
# tfs = T.Compose([T.ToTensor()])
# ds = JHU(root, transform=tfs, train=True, patch=False, flip=False)
# print(ds[0][0].shape)
# ds = JHU(root, transform=tfs, train=False, patch=False, flip=False)
# print(ds[0][0].shape)
# print(len(ds[0][1]))
# print(ds[0][1][0].keys())
# print(ds[0][1][0]['image_id'])
# print(ds[0][1][0]['point'])
# print(ds[0][1][0]['labels'])

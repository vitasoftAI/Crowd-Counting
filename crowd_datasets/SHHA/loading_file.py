# Import libraries
import torchvision.transforms as standard_transforms
from .SHHA import SHHA

class DeNormalize(object):
    
    """
    
    This class gets an object and denormalizes it.
    
    Parameter:
    
        object   -  an image object, tensor.
        
    Output:
    
        tensor   - a denormalized image object, tensor.
    
    """
    
    def __init__(self, mean, std): self.mean, self.std = mean, std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std): t.mul_(s).add_(m)
        
        return tensor

def loading_data(data_root):
    
    """
    
    This function gets a path to the data and loads data.
    
    Parameter:
    
        data_root     - a path to the data, str;
        
    Output:
    
        train_set     - train dataset, torch dataset object;
        val_set       - validation dataset, torch dataset object.
    
    """
    
    # Initialize transformations
    transform = standard_transforms.Compose([ standard_transforms.ToTensor(), standard_transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])
    
    # Get train dataset
    train_set = SHHA(data_root, train = True, transform = transform, patch = True, flip = False)
    
    # Get validation dataset
    val_set = SHHA(data_root, train = False, transform = transform)

    return train_set, val_set

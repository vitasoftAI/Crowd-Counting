from .p2pnet import build

# build the P2PNet model
# set training to 'True' during training
def build_model(args, device, training=False):
    return build(args, device, training)

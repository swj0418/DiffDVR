import torch
from torchvision.transforms import functional as F


def _clip_preprocess(n_px):
    def preprocess(image):
        # Assuming `image` is a PyTorch tensor of shape [C, H, W] and in the range [0, 1]
        # Resize and center crop
        image = F.resize(image, n_px, interpolation=F.InterpolationMode.BICUBIC)
        image = F.center_crop(image, n_px)

        # Normalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image = F.normalize(image, mean=mean, std=std)

        return image

    return preprocess

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


def create_tf_indices(rows):
    indices = []
    for i in range(rows):
        tmp = []
        for j in range(5):
            if j == 4:
                tmp.append(-1)
            else:
                tmp.append(4 * i + j)
        indices.append(tmp)
    indices = torch.tensor(indices, dtype=torch.int32).unsqueeze(0)
    return indices


def random_initial_tf(seed=0, cp=12):
    torch.manual_seed(seed)

    tf = torch.randint(low=0, high=255, size=(1, cp, 5), dtype=torch.float32)

    # RGB [0, 1]
    tf[:, :, 0:3] = tf[:, :, 0:3] / 255

    # Opacity [0, 100]
    tf[:, :, 3] = tf[:, :, 3] * (100 / 255)

    # Control point [0, 255], in ascending order. Sort every TF points based on control points.
    # control_points = tf[:, :, 4]
    # _, sorted_indices = torch.sort(control_points, dim=1)
    # sorted_tensor = torch.gather(tf, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, tf.size(2)))

    # Linearly spaced
    tf[:, :, 4] = torch.linspace(0, 255, steps=cp, dtype=torch.float32)

    return tf


# derivative_tf_indices = torch.tensor([[
#     [-1, -1, -1, -1, -1],
#     [0, 1, 2, 3, -1],
#     [4, 5, 6, 7, -1],
#     [8, 9, 10, 11, -1],
#     [12, 13, 14, 15, -1],
#     [16, 17, 18, 19, -1],
#     [20, 21, 22, 23, -1],
#     [24, 25, 26, 27, -1],
#     [28, 29, 30, 31, -1],
#     [32, 33, 34, 35, -1],
#     [36, 37, 38, 39, -1],
#     [-1, -1, -1, -1, -1]
# ]], dtype=torch.int32)

# initial_tf = torch.tensor([[
#         # r,g,b,a,pos
#         [0.23, 0.30, 0.75, 0.0 * opacity_scaling, 0],
#         [0.39, 0.52, 0.92, 0.0 * opacity_scaling, 10],
#         [0.39, 0.52, 0.92, 0.0 * opacity_scaling, 25],
#         [0.86, 0.86, 0.86, 0.4 * opacity_scaling, 50],
#         [0.86, 0.86, 0.86, 0.4 * opacity_scaling, 75],
#         [0.86, 0.86, 0.86, 0.4 * opacity_scaling, 100],
#         [0.86, 0.86, 0.86, 0.4 * opacity_scaling, 125],
#         [0.96, 0.75, 0.65, 0.8 * opacity_scaling, 150],
#         [0.96, 0.75, 0.65, 0.8 * opacity_scaling, 175],
#         [0.87, 0.39, 0.31, 0.99 * opacity_scaling, 200],
#         [0.87, 0.39, 0.31, 0.99 * opacity_scaling, 225],
#         [0.70, 0.015, 0.15, 0.99 * opacity_scaling, 255]
#     ]], dtype=dtype, device=device)

import argparse
import json

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import open_clip
import imageio
import OpenVisus as ov
import torchvision.utils

from utils import _clip_preprocess, create_tf_indices, random_initial_tf, histo_initial_tf, flat_initial_tf
from tf_transforms import TransformCamera, TransformTFHSL

sys.path.insert(0, os.getcwd())

import pyrenderer
from concurrent.futures import ProcessPoolExecutor
from data_loader import VolumeDatasetLoader
from vis import tfvis
from histogram import find_peaks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', type=str)
    parser.add_argument('--pitch', type=int)
    parser.add_argument('--yaw', type=int)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--seed', type=int)

    return parser.parse_args()

args = parse_args()

torch.set_printoptions(sci_mode=False, precision=3)
lr = 0.5
opacity_lr = 1.0
step_size = 200
gamma = 0.1
lamb = 0
num_peaks = 5
cp = 15
steepest = True
iterations = 600  # Optimization iterations
B = 1  # batch dimension
H = 224  # screen height
W = 224 # screen width

experiment_name = f'{args.volume}_{args.prompt}_{args.pitch}_{args.yaw}_{args.seed}'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-g-14')

grad_preprocess = _clip_preprocess(224)

clipmodel = clipmodel.cuda()
text = tokenizer([args.prompt]).cuda()

dataset = VolumeDatasetLoader(args.volume)
volume_dataset = ov.load_dataset(dataset.get_url(), cache_dir='./cache')
data = volume_dataset.read(x=(0, dataset.get_xyz()[0]), y=(0, dataset.get_xyz()[1]), z=(0, dataset.get_xyz()[2]))
peaks = find_peaks(data, num_peaks=num_peaks, steepest=steepest)

dtype = torch.float32
# data = data.astype(float)
volume = torch.from_numpy(data).unsqueeze(0)
volume = torch.tensor(volume, dtype=dtype, device=device)
X, Y, Z = dataset.get_xyz()

# initialize initial TF and render
print("Render initial")
# initial_tf = flat_initial_tf(args.seed, cp)
initial_tf = random_initial_tf(args.seed, cp)
# initial_tf = histo_initial_tf(peaks, seed=args.seed)
initial_tf = initial_tf.to(device)
print(initial_tf)

# Camera settings
fov_radians = np.radians(45.0)
camera_orientation = pyrenderer.Orientation.Ym
camera_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
camera_initial_pitch = torch.tensor([[np.radians(args.pitch)]], dtype=dtype, device=device)
camera_initial_yaw = torch.tensor([[np.radians(args.yaw)]], dtype=dtype, device=device)
camera_initial_distance = torch.tensor([[2.0]], dtype=dtype, device=device)


if __name__ == '__main__':
    viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, camera_initial_yaw, camera_initial_pitch, camera_initial_distance, camera_orientation)
    ray_start, ray_dir = pyrenderer.Camera.generate_rays(viewport, fov_radians, W, H)

    # TF settings
    tf_mode = pyrenderer.TFMode.Linear

    print("Create renderer inputs")
    inputs = pyrenderer.RendererInputs()
    inputs.screen_size = pyrenderer.int2(W, H)
    inputs.volume = volume.clone()
    inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
    inputs.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
    inputs.box_size = pyrenderer.real3(1, 1, 1)
    inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
    inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
    inputs.step_size = 0.5 / X
    inputs.tf_mode = tf_mode
    inputs.blend_mode = pyrenderer.BlendMode.BeerLambert

    print("Create forward difference settings")
    differences_settings = pyrenderer.ForwardDifferencesSettings()

    # differences_settings.D = 4 * num_peaks  # TF + camera
    # derivative_tf_indices = create_tf_indices(num_peaks + 2)

    differences_settings.D = 4 * (cp - 2)  # TF + camera
    derivative_tf_indices = create_tf_indices(cp)

    differences_settings.d_tf = derivative_tf_indices.to(device=device)
    differences_settings.d_rayStart = pyrenderer.int3(0, 1, 2)
    differences_settings.d_rayDir = pyrenderer.int3(3, 4, 5)
    differences_settings.has_tf_derivatives = True

    print("Create renderer outputs")
    output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
    output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
    outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
    gradients_out = torch.empty(1, H, W, differences_settings.D, 4, dtype=dtype, device=device)

    class RendererDeriv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, transformed_tf):
            inputs.tf = transformed_tf

            # Allocate output tensors
            output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
            output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
            outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
            gradients_out = torch.empty(1, H, W, differences_settings.D, 4, dtype=dtype, device=device)

            # Render
            pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings, outputs, gradients_out)
            ctx.save_for_backward(gradients_out, transformed_tf)
            return output_color

        @staticmethod
        def backward(ctx, grad_output_color):
            gradients_out, transformed_tf = ctx.saved_tensors

            grad_output_color = grad_output_color.unsqueeze(3)  # for broadcasting over the derivatives
            gradients = torch.mul(gradients_out, grad_output_color)  # adjoint-multiplication

            gradients = torch.sum(gradients, dim=[1, 2, 4])  # reduce over screen height, width and channel

            # TF map
            grad_tf = torch.zeros_like(transformed_tf)
            for R in range(grad_tf.shape[1]):
                for C in range(grad_tf.shape[2]):
                    idx = derivative_tf_indices[0, R, C]
                    if idx >= 0:
                        grad_tf[:, R, C] = gradients[:, idx]

            return grad_tf
    rendererDeriv = RendererDeriv.apply

    class OptimModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_transform = TransformTFHSL()

        def forward(self, current_tf_color, current_tf_opacity):
            # TF transform - activation
            transformed_tf = self.tf_transform(current_tf_color)
            transformed_tf_opacity = self.tf_transform(current_tf_opacity)
            transformed_tf[:, :, 3:4] = transformed_tf_opacity[:, :, 3:4]

            # Forward
            color = rendererDeriv(transformed_tf)
            return viewport, transformed_tf, color
    model = OptimModel()

    # run optimization
    reconstructed_color = []
    reconstructed_tf = []
    reconstructed_loss = []
    reconstructed_sparsity = []
    reconstructed_cliploss = []

    # Working parameters
    current_tf = initial_tf.clone()[0]
    current_tf_opacity = initial_tf.clone()[0]
    current_tf.requires_grad_()
    current_tf_opacity.requires_grad_()

    # optimizer = torch.optim.Adam([current_tf], lr=lr)
    # optimizer_opacity = torch.optim.Adam([current_tf_opacity], lr=opacity_lr)

    optimizer = torch.optim.SGD([current_tf], lr=lr)
    optimizer_opacity = torch.optim.SGD([current_tf_opacity], lr=opacity_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler_opacity = torch.optim.lr_scheduler.StepLR(optimizer_opacity, step_size=step_size, gamma=gamma)

    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr, end_factor=0.001, total_iters=iterations)
    # scheduler_opacity = torch.optim.lr_scheduler.LinearLR(optimizer_opacity, start_factor=opacity_lr, end_factor=0.001, total_iters=iterations)

    for iteration in range(iterations):
        optimizer.zero_grad()
        optimizer_opacity.zero_grad()

        viewport, transformed_tf, color = model(current_tf, current_tf_opacity)

        # preprocess and embed
        # Tensor [C, H, W]
        tmpimg = color[:, :, :, :3][0]
        tmpimg = torch.swapdims(tmpimg, 0, 2)  # [C, W, H]
        tmpimg = torch.swapdims(tmpimg, 1, 2)  # [C, H, W]

        prep_img = grad_preprocess(tmpimg)
        prep_img = prep_img.float()

        # Embed
        embedding = clipmodel.encode_image(prep_img.unsqueeze(0).cuda())[0]
        text_features = clipmodel.encode_text(text)
        nembedding = embedding / embedding.norm(dim=-1, keepdim=True)
        ntext_features = text_features / text_features.norm(dim=-1, keepdim=True)

        score = 1 - nembedding @ ntext_features.T

        # Sparsity
        l1 = torch.sum(torch.abs(current_tf_opacity[:, 1:-1, 3:4] / 255))  # Sparsity in opacity only
        loss = score + (lamb * l1)

        # compute loss
        # if iteration % 4 == 0:
        reconstructed_color.append(color.detach().cpu().numpy()[0, :, :, 0:3])
        reconstructed_cliploss.append(score.item())
        reconstructed_sparsity.append(l1.item())
        reconstructed_tf.append(transformed_tf.detach().cpu().numpy()[0])

        loss.backward()
        optimizer.step()
        optimizer_opacity.step()
        scheduler.step()
        scheduler_opacity.step()
        print("Iteration % 4d, CD: %7.5f, L1: %7.5f" % (iteration, score.item(), l1.item()))

    print("Visualize Optimization")
    tmp_fig_folder = 'tmp_figure'
    retain_fig_folder = f'experiment_figs/figure_{experiment_name}'
    os.makedirs(tmp_fig_folder, exist_ok=True)
    os.makedirs(retain_fig_folder, exist_ok=True)

    num_frames = len(reconstructed_color)  # Assuming reconstructed_color holds the data for each frame
    def generate_frame(frame):
        # Your existing logic to generate and save a single frame
        fig, axs = plt.subplots(2, 2, figsize=(6, 9))

        axs[0, 0].imshow(reconstructed_color[frame])
        tfvis.renderTfLinear(reconstructed_tf[frame], axs[0, 1])

        # Update other plots as needed
        axs[1, 0].imshow(reconstructed_color[0])  # Initialization
        axs[1, 1].plot(reconstructed_cliploss)

        # Titles
        axs[0, 0].set_title("Optimized Image")
        axs[0, 1].set_title("TF")
        axs[1, 0].set_title("Initial Image")
        axs[1, 1].set_title("Cosine Distance")

        fig.suptitle(
            "Iteration % 4d, CD: %7.5f, L1: %7.5f" % (
                frame, reconstructed_cliploss[frame], reconstructed_sparsity[frame]
            ))
        fig.tight_layout()

        # Save the frame
        frame_filename = f"{tmp_fig_folder}/frame_{frame:04d}.png"
        fig.savefig(frame_filename)

        if frame % 100 == 0 or frame == iterations - 1:
            fig.savefig(f"{retain_fig_folder}/frame_{frame:04d}.png")

        plt.close(fig)  # Close the figure to free memory

    # Parallelize frame generation
    with ProcessPoolExecutor() as executor:
        executor.map(generate_frame, range(num_frames))

    # Compile frames into a GIF
    frame_files = [f"{tmp_fig_folder}/frame_{frame:04d}.png" for frame in range(num_frames)]
    images = [imageio.v2.imread(frame_file) for frame_file in frame_files]
    imageio.mimsave(f'{retain_fig_folder}/test_tf_optimization.gif', images, loop=10, fps=10)  # Adjust fps as needed
    for frame_file in frame_files:  # Cleanup
        os.remove(frame_file)

    # Camera setting & TF export
    settings = {}
    current_tf = current_tf.detach().cpu()
    current_tf_opacity = current_tf_opacity.detach().cpu()
    current_tf[:, :, 3:4] = current_tf_opacity[:, :, 3:4]
    settings['tf'] = current_tf
    settings['camera'] = {
        "orientation": "Ym",
        "center": camera_center.detach().cpu(),
        "pitch": camera_initial_pitch.detach().cpu(),
        "yaw": camera_initial_yaw.detach().cpu(),
        "distance": camera_initial_distance.detach().cpu()
    }
    torch.save(settings, f'{retain_fig_folder}/settings.pt')

    # Final render
    viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, camera_initial_yaw, camera_initial_pitch, camera_initial_distance, camera_orientation)
    ray_start, ray_dir = pyrenderer.Camera.generate_rays(viewport, fov_radians, 1024, 1024)
    inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)

    color = torch.empty(1, 1024, 1024, 4, dtype=dtype, device=device)
    output_termination_index = torch.empty(1, 1024, 1024, dtype=torch.int32, device=device)
    outputs = pyrenderer.RendererOutputs(color, output_termination_index)
    gradients_out = torch.empty(1, 1024, 1024, differences_settings.D, 4, dtype=dtype, device=device)
    inputs.screen_size = pyrenderer.int2(1024, 1024)
    # Render
    pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings, outputs, gradients_out)

    tmpimg = color[:, :, :, :3][0]
    tmpimg = torch.swapdims(tmpimg, 0, 2)  # [C, W, H]
    tmpimg = torch.swapdims(tmpimg, 1, 2)  # [C, H, W]
    torchvision.utils.save_image(tmpimg, f"{retain_fig_folder}/final.png", normalize=True)

    pyrenderer.cleanup()

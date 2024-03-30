import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm
import open_clip
import imageio
import OpenVisus as ov
from PIL import Image

from utils import _clip_preprocess

sys.path.insert(0, os.getcwd())

# load pyrenderer
from diffdvr import make_real3, Settings
import pyrenderer
from concurrent.futures import ProcessPoolExecutor

from vis import tfvis

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
grad_preprocess = _clip_preprocess(224)
clipmodel = clipmodel.cuda()
text = tokenizer(["A CT scan of a teapot"]).cuda()

# Load data
dataset = ov.load_dataset('https://klacansky.com/open-scivis-datasets/boston_teapot/boston_teapot.idx', cache_dir='./cache')
data = dataset.read(x=(0, 256), y=(0, 256), z=(0, 178))

# dataset = ov.load_dataset('https://klacansky.com/open-scivis-datasets/carp/carp.idx', cache_dir='./cache')
# data = dataset.read(x=(0, 256), y=(0, 256), z=(0, 512))

dtype = torch.float32
data = data.astype(float)
volume = torch.from_numpy(data).unsqueeze(0)
volume = torch.tensor(volume, dtype=dtype, device=device)
X, Y, Z = 256, 256, 178
camera_gradient_discount_factor = 2

torch.set_printoptions(sci_mode=False, precision=3)
lr = 5.0
step_size = 100
gamma = 0.1
iterations = 300  # Optimization iterations
B = 1  # batch dimension
H = 224  # screen height
W = 224  # screen width


class TransformCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pitch, yaw):
        return torch.cat([
            self.sigmoid(pitch),
            self.sigmoid(yaw)
        ])


class InverseTransformCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pitch, yaw):
        def inverseSigmoid(y):
            y = y * 2
            return torch.log(-y / (y - 1))

        return torch.cat([
            inverseSigmoid(pitch),
            inverseSigmoid(yaw)
        ])


class TransformTFParameterization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
        self.cp_min, self.cp_max = 0, 255

    def _check_start_condition(self, start):
        # 0 < starting point < 254
        if start < 0:
            start = torch.tensor([0], dtype=dtype, device=device)
        elif start > 254:
            start = torch.tensor([254], dtype=dtype, device=device)
        return start

    def _check_width_condition(self, start, width, fix_width=True):
        # start + width < 255
        if start + width > 255:
            if fix_width:
                width = torch.tensor([(start + width) - 255], dtype=dtype, device=device)
            else:
                start = torch.tensor([(start + width) - 255], dtype=dtype, device=device)
        return start, width

    def _check_height_condition(self, height):
        # height < 100 (?)
        if height > 100:
            height = torch.tensor([100], dtype=dtype, device=device)
        if height < 0:
            height = torch.tensor([0], dtype=dtype, device=device)
        return height

    def _build_tf(self, start, width, height, rgb):
        # Convert LAB into RGB
        tf = torch.zeros(size=(1, 3, 5), dtype=dtype, device=device)
        tf[:, 0, 4] = start
        tf[:, 1, 4] = start + (width / 2.)
        tf[:, 1, 3] = height
        tf[:, 2, 4] = start + width

        # RGB
        tf[:, 1, 0] = rgb[3]
        tf[:, 1, 1] = rgb[4]
        tf[:, 1, 2] = rgb[5]

        return tf

    def forward(self, param_tf):
        # L A B
        # print("PR:", param_tf.detach().cpu().numpy())

        # Opacity, Control Point
        start = self._check_start_condition(param_tf[0])
        start, width = self._check_width_condition(param_tf[0], param_tf[1])
        height = self._check_height_condition(param_tf[2])
        tf = self._build_tf(start, width, height, param_tf)

        return torch.cat([
            self.sigmoid(tf[:, :, 0:3]),  # color
            self.softplus(tf[:, :, 3:4]),  # opacity
            tf[:, :, 4:5]  # position
        ], dim=2)


# class InverseTransformTFParameterization(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, tf):
#         # Tf -> start, width, height
#         def inverseSigmoid(y):
#             return torch.log(-y / (y - 1))
#
#         def inverseSoftplus(y, beta=1, threshold=20):
#             # if y*beta>threshold: return y
#             return torch.log(torch.exp(beta * y) - 1) / beta
#
#         assert len(tf.shape) == 3
#         assert tf.shape[2] == 5
#         tf = torch.cat([
#             inverseSigmoid(tf[:, :, 0:3]),  # color
#             inverseSoftplus(tf[:, :, 3:4]),  # opacity
#             tf[:, :, 4:5]  # position
#         ], dim=2)
#
#         # start width and height
#         start = tf[:, 0, 4]
#         width = tf[:, 2, 4] - start
#         height = tf[:, 1: 3]
#         r, g, b = tf[:, 1, 0], tf[:, 1, 1], tf[:, 1, 2]
#         return torch.tensor([start, width, height, r, g, b], dtype=dtype, device=device)


if __name__ == '__main__':
    # Camera settings
    fov_radians = np.radians(45.0)
    camera_orientation = pyrenderer.Orientation.Ym
    camera_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)

    # [0, 2pi]
    camera_initial_pitch = torch.tensor([[np.radians(0)]], dtype=dtype,
                                        device=device)  # torch.tensor([[np.radians(-14.5)]], dtype=dtype, device=device)
    camera_initial_yaw = torch.tensor([[np.radians(0)]], dtype=dtype,
                                      device=device)  # torch.tensor([[np.radians(113.5)]], dtype=dtype, device=device)
    camera_initial_roll = torch.tensor([[np.radians(0)]], dtype=dtype,
                                      device=device)  # torch.tensor([[np.radians(113.5)]], dtype=dtype, device=device)
    camera_initial_distance = torch.tensor([[2.0]], dtype=dtype, device=device)

    viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, camera_initial_yaw, camera_initial_pitch, camera_initial_distance, camera_orientation)
    ray_start, ray_dir = pyrenderer.Camera.generate_rays(viewport, fov_radians, W, H)

    # TF settings
    # Triangular TF: start, width, height, L, A, B
    tf_mode = pyrenderer.TFMode.Linear
    opacity_scaling = 25.0

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
    differences_settings.D = 15  # TF + camera
    # derivative_tf_indices = torch.tensor([[[0, 1, 2, 3, 4, 5]]], dtype=torch.int32)
    derivative_tf_indices = torch.tensor([[
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14]
    ]], dtype=torch.int32)
    differences_settings.d_tf = derivative_tf_indices.to(device=device)
    differences_settings.d_rayStart = pyrenderer.int3(0, 1, 2)
    differences_settings.d_rayDir = pyrenderer.int3(3, 4, 5)
    differences_settings.has_tf_derivatives = True

    print("Create renderer outputs")
    output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
    output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
    outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
    gradients_out = torch.empty(1, H, W, differences_settings.D, 4, dtype=dtype, device=device)

    # initialize initial TF and render
    print("Render initial")
    initial_tf = torch.tensor([55, 10, 0.8 * opacity_scaling, 0.2, 0.2, 0.2], dtype=dtype, device=device)
    initial_transformed_tf = TransformTFParameterization()(initial_tf)

    class RendererDeriv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ray_start, ray_end, transformed_tf):
            inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
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
            # print("Gradient size: ", gradients.shape)  # [1, 224, 224, 22 (D), 4 (Channel)]

            # I don't know how to aggregate if I were to compute gradients for camera and TF
            c_gradients = torch.sum(gradients, dim=4)  # reduce over channel
            gradients = torch.sum(gradients, dim=[1, 2, 4])  # reduce over screen height, width and channel
            # print(c_gradients.shape, gradients.shape)

            # Map to output variables
            # grad_ray_start = c_gradients[..., 15:18]
            # grad_ray_dir = c_gradients[..., 18:21]
            # print(grad_ray_dir.sum(), grad_ray_start.sum())

            grad_ray_start = c_gradients[..., 0:3]
            grad_ray_dir = c_gradients[..., 3:6]

            # TF map
            grad_tf = torch.zeros_like(transformed_tf)
            for R in range(grad_tf.shape[1]):
                for C in range(grad_tf.shape[2]):
                    idx = derivative_tf_indices[0, R, C]  # Indices already offset by camera grad indices
                    if idx >= 0:
                        grad_tf[:, R, C] = gradients[:, idx]

            return grad_ray_start, grad_ray_dir, grad_tf
    rendererDeriv = RendererDeriv.apply

    class OptimModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_transform = TransformTFParameterization()
            self.camera_transform = TransformCamera()

        def forward(self, current_pitch, current_yaw, current_distance, current_tf):
            # Camera transform = activation
            # transformed_pitch, transformed_yaw = self.camera_transform(current_pitch, current_yaw)
            # transformed_pitch, transformed_yaw = transformed_pitch.unsqueeze(0), transformed_yaw.unsqueeze(0)
            print(current_yaw, current_pitch)

            # Camera
            viewport = pyrenderer.Camera.viewport_from_sphere(
                camera_center, current_yaw, current_pitch, current_distance, camera_orientation)
            ray_start, ray_dir = pyrenderer.Camera.generate_rays(viewport, fov_radians, W, H)

            # TF transform - activation
            transformed_tf = self.tf_transform(current_tf)

            # Forward
            color = rendererDeriv(ray_start, ray_dir, transformed_tf)

            return viewport, transformed_tf, color
    model = OptimModel()

    # run optimization
    reconstructed_color = []
    reconstructed_viewport = []
    reconstructed_tf = []
    reconstructed_loss = []
    reconstructed_cliploss = []
    reconstructed_pitchyaw = []

    # Working parameters
    current_pitch = camera_initial_pitch.clone()
    current_yaw = camera_initial_yaw.clone()
    current_roll = camera_initial_roll.clone()
    current_distance = camera_initial_distance.clone()
    current_pitch.requires_grad_()
    current_yaw.requires_grad_()
    current_distance.requires_grad_()

    current_tf = initial_tf.clone()
    current_tf.requires_grad_()

    optimizer = torch.optim.Adam([current_pitch, current_yaw, current_distance, current_tf], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for iteration in range(iterations):
        optimizer.zero_grad()

        viewport, transformed_tf, color = model(current_pitch, current_yaw, current_distance, current_tf)
        # print("Current: ", transformed_tf.detach().cpu().numpy())

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

        # compute loss
        # if iteration % 4 == 0:
        reconstructed_color.append(color.detach().cpu().numpy()[0, :, :, 0:3])
        reconstructed_cliploss.append(score.item())
        reconstructed_tf.append(transformed_tf.detach().cpu().numpy()[0])
        reconstructed_pitchyaw.append((current_pitch.cpu(), current_distance.cpu()))

        score.backward()
        optimizer.step()
        scheduler.step()
        print("Iteration % 4d, Cosine Distance: %7.5f" % (iteration, score.item()))

    print("Visualize Optimization")
    tmp_fig_folder = 'tmp_figure'
    os.makedirs(tmp_fig_folder, exist_ok=True)

    num_frames = len(reconstructed_color)  # Assuming reconstructed_color holds the data for each frame
    print(num_frames)
    def generate_frame(frame):
        # Your existing logic to generate and save a single frame
        fig, axs = plt.subplots(2, 2, figsize=(6, 9))

        axs[0, 0].imshow(reconstructed_color[frame])
        tfvis.renderTfLinear(reconstructed_tf[frame], axs[0, 1])

        # Update other plots as needed
        axs[1, 1].plot(reconstructed_cliploss)
        fig.suptitle(
            "Iteration % 4d, Cosine Distance: %7.5f" % (
                frame, reconstructed_cliploss[frame]
            ))
        fig.tight_layout()

        # Save the frame
        frame_filename = f"{tmp_fig_folder}/frame_{frame:04d}.png"
        fig.savefig(frame_filename)
        plt.close(fig)  # Close the figure to free memory

    # Parallelize frame generation
    with ProcessPoolExecutor() as executor:
        executor.map(generate_frame, range(num_frames))

    # Compile frames into a GIF
    frame_files = [f"{tmp_fig_folder}/frame_{frame:04d}.png" for frame in range(num_frames)]
    images = [imageio.v2.imread(frame_file) for frame_file in frame_files]
    imageio.mimsave('test_tf_optimization.gif', images, loop=10, fps=10)  # Adjust fps as needed
    for frame_file in frame_files:  # Cleanup
        os.remove(frame_file)

    pyrenderer.cleanup()

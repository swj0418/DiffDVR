import torch

from pytests.nonparam_camera_ import dtype, device


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
        tf = torch.zeros(size=(1, 5, 5), dtype=dtype, device=device)
        tf[:, 1, 4] = start
        tf[:, 2, 4] = start + (width / 2.)
        tf[:, 2, 3] = height
        tf[:, 3, 4] = start + width
        tf[:, 4, 4] = 255

        # RGB
        tf[:, 2, 0] = rgb[3]
        tf[:, 2, 1] = rgb[4]
        tf[:, 2, 2] = rgb[5]

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


class InverseTransformTFParameterization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tf):
        # Tf -> start, width, height
        def inverseSigmoid(y):
            return torch.log(-y / (y - 1))

        def inverseSoftplus(y, beta=1, threshold=20):
            # if y*beta>threshold: return y
            return torch.log(torch.exp(beta * y) - 1) / beta

        assert len(tf.shape) == 3
        assert tf.shape[2] == 5
        tf = torch.cat([
            inverseSigmoid(tf[:, :, 0:3]),  # color
            inverseSoftplus(tf[:, :, 3:4]),  # opacity
            tf[:, :, 4:5]  # position
        ], dim=2)

        # start width and height
        start = tf[:, 0, 4]
        width = tf[:, 2, 4] - start
        height = tf[:, 1: 3]
        r, g, b = tf[:, 1, 0], tf[:, 1, 1], tf[:, 1, 2]
        return torch.tensor([start, width, height, r, g, b], dtype=dtype, device=device)


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


class TransformCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

    def forward(self, pitch, yaw):
        return torch.cat([
            self.tanh(pitch) * 2,
            self.tanh(yaw) * 2
        ])

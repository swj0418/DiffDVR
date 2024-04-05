import torch


# def hue_to_rgb(p, q, t):
#     new_t = t.clone()
#     if t < 0:
#         new_t = t + 1
#     if new_t > 1:
#         new_t = new_t - 1
#     if new_t < 1 / 6:
#         return p + (q - p) * 6 * new_t
#     if new_t < 1 / 2:
#         return q
#     if new_t < 2 / 3:
#         return p + (q - p) * (2 / 3 - new_t) * 6
#     return p

def hue_to_rgb(p, q, t):
    t = torch.where(t < 0, t + 1, t)
    t = torch.where(t > 1, t - 1, t)
    r = torch.where(t < 1/6, p + (q - p) * 6 * t, q)
    r = torch.where((t >= 1/6) & (t < 1/2), q, r)
    r = torch.where((t >= 1/2) & (t < 2/3), p + (q - p) * (2/3 - t) * 6, r)
    return torch.where(t >= 2/3, p, r)


def hsl_to_rgb(hsl):
    h, s, l = hsl[:, 0], hsl[:, 1], hsl[:, 2]

    q = torch.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    result = torch.stack((r, g, b), dim=1)
    return result

class TransformTFHSL(torch.nn.Module):
    def __init__(self, lab=False):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()

    def forward(self, tf):
        assert len(tf.shape) == 3
        assert tf.shape[2] == 5

        new_tf = torch.cat([
            self.sigmoid(tf[:, :, 0:1]), # Hue
            self.sigmoid(tf[:, :, 1:2]) * 0.5 + 0.5, # Saturation - modulate
            self.sigmoid(tf[:, :, 2:3]), # Luminance
            tf[:, :, 3:4],  # opacity
            tf[:, :, 4:5]  # position
        ], dim=2)

        # Convert HSL to RGB
        converted_tf = new_tf.clone()
        for i in range(tf.shape[1]):
            converted_tf[:, i, 0:3] = hsl_to_rgb(new_tf[:, i, 0:3])

        return torch.cat([
            converted_tf[:, :, 0:3],  # color in RGB
            self.softplus(converted_tf[:, :, 3:4]),  # opacity
            tf[:, :, 4:5]  # position
        ], dim=2)


# class TransformTF(torch.nn.Module):
#     def __init__(self, lab=False):
#         super().__init__()
#         self.sigmoid = torch.nn.Sigmoid()
#         self.softplus = torch.nn.Softplus()
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, tf):
#         assert len(tf.shape) == 3
#         assert tf.shape[2] == 5
#         return torch.cat([
#             self.sigmoid(tf[:, :, 0:3]),  # color
#             self.softplus(tf[:, :, 3:4]),  # opacity
#             tf[:, :, 4:5]  # position
#         ], dim=2)


class TransformTF(torch.nn.Module):
    def __init__(self, lab=False, max_softplus_output=20):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()
        self.max_softplus_output = max_softplus_output  # Max expected value from Softplus

    def forward(self, tf):
        assert len(tf.shape) == 3
        assert tf.shape[2] == 5

        # Apply Softplus and then clamp values above 100 to be exactly 100
        opacity = self.softplus(tf[:, :, 3:4])
        opacity = torch.clamp(opacity, max=100)  # Clamp opacity to a maximum of 100

        return torch.cat([
            self.sigmoid(tf[:, :, 0:3]),  # color
            opacity,  # opacity, clamped to not exceed 100
            tf[:, :, 4:5]  # position
        ], dim=2)

class TransformTFParameterization(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
        self.cp_min, self.cp_max = 0, 255
        self.dtype, self.device = dtype, device

    def _check_start_condition(self, start):
        # 0 < starting point < 254
        if start < 0:
            start = torch.tensor([0], dtype=self.dtype, device=self.device)
        elif start > 254:
            start = torch.tensor([254], dtype=self.dtype, device=self.device)
        return start

    def _check_width_condition(self, start, width, fix_width=True):
        # start + width < 255
        if start + width > 255:
            if fix_width:
                width = torch.tensor([(start + width) - 255], dtype=self.dtype, device=self.device)
            else:
                start = torch.tensor([(start + width) - 255], dtype=self.dtype, device=self.device)
        return start, width

    def _check_height_condition(self, height):
        # height < 100 (?)
        if height > 100:
            height = torch.tensor([100], dtype=self.dtype, device=self.device)
        if height < 0:
            height = torch.tensor([0], dtype=self.dtype, device=self.device)
        return height

    def _build_tf(self, start, width, height, rgb):
        # Convert LAB into RGB
        tf = torch.zeros(size=(1, 5, 5), dtype=self.dtype, device=self.device)
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
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype, self.device = dtype, device

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
        return torch.tensor([start, width, height, r, g, b], dtype=self.dtype, device=self.device)


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


if __name__ == '__main__':
    hsl = torch.tensor([[
        [0.2, 0.2, 0.2, 45, 255],
        [0.2, 0.2, 0.2, 45, 255],
        [0.2, 0.2, 0.2, 45, 255],
    ]])
    trans = TransformTFHSL()

    # test_data = hsl[0, 0, 0:3]
    # print(test_data)
    # transformed = hsl_to_rgb(test_data)
    # print(transformed)

    transformed = trans(hsl)
    print(transformed)
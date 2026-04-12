import torch
import torch.nn.functional as F
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch.nn as nn


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, class_idx=None):

        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        loss = output[:, class_idx]

        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.cpu().numpy()

        return cam


# ==========================
# Overlay GradCAM
# ==========================

def overlay_gradcam(image, cam):

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = heatmap * 0.4 + image

    return np.uint8(overlay)


# ==========================
# Convertir imagen a base64
# ==========================

def convert_to_base64(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


# ==========================
# CBAM
# ==========================

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.size()

        avg_pool = torch.mean(x, dim=(2,3)).view(b,c)
        max_pool = torch.amax(x, dim=(2,3)).view(b,c)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        out = avg_out + max_out

        out = self.sigmoid(out).view(b,c,1,1)

        return x * out


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_pool, max_pool], dim=1)

        out = self.conv(concat)

        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):

    def __init__(self, channels):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x
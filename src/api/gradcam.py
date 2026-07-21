import cv2
import numpy as np
import torch

from PIL import Image


class GradCAMGenerator:

    def __init__(self, model):

        self.model = model

        self.gradients = None
        self.activations = None

        self.target_layer = (self.model.model.features[-1])


        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor,
        original_image,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[0, class_index]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= (
            cam.max() + 1e-8
        )

        cam = cam.cpu().numpy()

        cam = cv2.resize(
            cam,
            (
                original_image.width,
                original_image.height
            )
        )

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        original = cv2.cvtColor(
            np.array(original_image),
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(
            overlay
        )

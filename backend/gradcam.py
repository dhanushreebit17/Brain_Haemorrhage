import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Fix for "BackwardHookFunction is a view..." error:
        # in-place ReLUs conflict with full_backward_hook tracking.
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: [1, 3, H, W], already normalized, requires no grad set externally
        class_idx: which class to explain. If None, uses the predicted class.
        Returns: (heatmap [H, W] in range 0-1, predicted_class_idx, probs)
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)          # [1, num_classes]
        probs = F.softmax(output, dim=1)[0]

        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # [C, h, w]
        activations = self.activations[0]  # [C, h, w]

        weights = gradients.mean(dim=(1, 2))  # [C] global average pool

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam, class_idx, probs.detach().cpu().numpy()


def overlay_heatmap(original_bgr_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """
    original_bgr_img: uint8 image, shape [H, W, 3], BGR (as read by cv2)
    cam: float heatmap in range [0, 1], shape [H, W]
    Returns: uint8 BGR image with heatmap overlay
    """
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
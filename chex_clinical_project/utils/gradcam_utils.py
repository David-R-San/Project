# gradcam_utils.py
# gradcam_utils.py
import torch
import numpy as np
import cv2

def register_hooks(model, target_layer, grads, acts):
    def save_grad(_, __, grad_output):
        grads.append(grad_output[0])

    def save_act(_, __, output):
        acts.append(output)

    target_layer.register_forward_hook(save_act)
    target_layer.register_full_backward_hook(save_grad)

def compute_gradcam(grads, acts):
    grad = grads[0].cpu().detach().numpy()[0]
    act = acts[0].cpu().detach().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import matplotlib.cm as cm


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize(tensor_img):
    img = tensor_img.cpu().clone() * STD + MEAN
    arr = img.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(arr, 0, 1)


def apply_gradcam_batch(cam_extractor, inputs, labels, outputs,
                        class_names, mode='incorrect',
                        top_k=4, save_path='gradcam.png'):

    probs = torch.nn.functional.softmax(outputs, dim=1)
    confs, preds = probs.max(dim=1)
    mask = (preds != labels) if mode == 'incorrect' else (preds == labels)
    if mask.sum() == 0:
        print(f"No {mode} predictions in this batch.")
        return

    selected = confs[mask].argsort(descending=True)[:top_k]
    real_idxs = mask.nonzero().squeeze(1)[selected]

    k = real_idxs.size(0)
    fig, axs = plt.subplots(1, k, figsize=(5*k, 5))
    if k == 1:
        axs = [axs]

    for ax, idx in zip(axs, real_idxs):
        i = idx.item()
        img_t = inputs[i]
        true, pred, conf = labels[i].item(), preds[i].item(), confs[i].item()

        # get CAM
        target_class = pred
        activation_map = cam_extractor(
            target_class, outputs[i].unsqueeze(0), retain_graph=True)[0]

        cam_np = activation_map.detach().cpu().numpy()
        cam_np = cam_np.squeeze()

        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            print(f"Warning: Flat CAM map detected for image {i}")

        if cam_np.ndim != 2:
            if cam_np.ndim > 2:
                cam_np = np.mean(cam_np, axis=0)
            else:
                cam_np = np.ones((7, 7)) * 0.5

        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        img_pil = to_pil_image(unnormalize(img_t))
        img_np = np.array(img_pil)
        heatmap_rgb = (cm.jet(cam_np)[:, :, :3] * 255).astype(np.uint8)

        if heatmap_rgb.shape[:2] != img_np.shape[:2]:
            from PIL import Image
            heatmap_pil = Image.fromarray(heatmap_rgb)
            heatmap_pil = heatmap_pil.resize(
                img_pil.size, resample=Image.BICUBIC)
            heatmap_rgb = np.array(heatmap_pil)

        alpha = 0.4
        overlay = img_np * (1-alpha) + heatmap_rgb * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title(
            f"True: {class_names[true]}\n"
            f"Pred: {class_names[pred]} ({conf:.2f})"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=400)
    plt.close(fig)
    print(f"Saved {mode} Grad-CAM to {save_path}")

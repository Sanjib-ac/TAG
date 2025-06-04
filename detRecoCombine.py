import os

os.environ["DOCTR_CACHE_DIR"] = "./"

import importlib
import torch
import torch.nn as nn
import numpy as np

# === PATCH 1: Override DocumentFile.from_images with a dummy version ===
from doctr.io import DocumentFile


def dummy_from_images(cls, pages):
    # Dummy implementation; this branch should not be used at inference
    return None


# Mark the dummy function so that TorchScript ignores its body.
dummy_from_images.__torch_script_ignore__ = True
DocumentFile.from_images = classmethod(dummy_from_images)

# === PATCH 2: Override _remove_padding to force tensor arithmetic ===
# Import the module where _remove_padding is defined.
detection_utils = importlib.import_module("doctr.models.detection._utils.base")


def patched_remove_padding(loc_pred, pages, *args, **kwargs):
    """
    This patched version ensures that loc_pred is converted to a torch.Tensor,
    performs the arithmetic entirely in tensor-space (with factor converted to a tensor),
    and then converts back to a NumPy array if the original loc_pred was a NumPy array.
    """
    # Determine height and width from pages.
    if isinstance(pages, (list, tuple)) and len(pages) > 0:
        sample = pages[0]
        if hasattr(sample, "shape"):
            h = float(sample.shape[0])
            w = float(sample.shape[1])
        else:
            h, w = 720.0, 1280.0  # Fallback defaults.
    else:
        # If pages is a tensor in the shape [B, 3, H, W]
        h = float(pages.shape[-2])
        w = float(pages.shape[-1])
    factor = w / h  # A plain Python float.

    # Check if loc_pred is already a torch.Tensor.
    converted = False
    if not isinstance(loc_pred, torch.Tensor):
        # Convert loc_pred to a tensor.
        loc_pred = torch.as_tensor(loc_pred)
        converted = True

    # Ensure that the multiplication factor is a torch scalar of the same dtype and device.
    factor_tensor = torch.tensor(factor, dtype=loc_pred.dtype, device=loc_pred.device)

    # Perform the arithmetic fully in tensor-space.
    new_vals = (loc_pred[:, :, 1] - 0.5) * factor_tensor + 0.5
    loc_pred[:, :, 1] = new_vals

    # If we converted from a NumPy array, convert back.
    if converted:
        return loc_pred.cpu().numpy()
    else:
        return loc_pred


# Patch the original _remove_padding function.
detection_utils._remove_padding = patched_remove_padding

# === Now import the OCR predictor (after our patches) ===
from doctr.models import ocr_predictor


# === Define a TorchScript/tracing-friendly wrapper module ===
class ScriptableOCR(nn.Module):
    def __init__(self, device):
        super(ScriptableOCR, self).__init__()
        # Create the full OCR predictor with all submodels.
        self.ocr = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='sar_resnet31',
            pretrained=True,
            assume_straight_pages=False
        ).to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Expect the caller to provide a tensor of shape [B, 3, H, W].
        return self.ocr(images)


if __name__ == '__main__':
    # Choose the appropriate device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScriptableOCR(device)

    # Create a dummy input tensor—for instance, one image of size 720x1280.
    dummy_input = torch.randn(1, 3, 720, 1280, device=device)

    # We use tracing to record the model's computation given the dummy input.
    traced_model = torch.jit.trace(model, dummy_input)

    traced_model.save("adsTag7201280_traced.pt")
    print("Saved traced OCR model as adsTag7201280_traced.pt")

import warnings

import torch


def load_state_dict_flexible(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If checkpoint is a wrapper (like {"state_dict": ...})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state_dict = model.state_dict()
    loaded_state_dict = {}

    for name, param in state_dict.items():
        if name not in model_state_dict:
            warnings.warn(f"Skipping '{name}' as it is not in the model.")
            continue

        if model_state_dict[name].shape != param.shape:
            warnings.warn(
                f"Shape mismatch for '{name}': "
                f"model={model_state_dict[name].shape}, "
                f"checkpoint={param.shape}. Skipping."
            )
            continue

        loaded_state_dict[name] = param

    # Load only the matching parameters
    missing_keys, unexpected_keys = model.load_state_dict(
        loaded_state_dict, strict=False
    )

    if missing_keys:
        warnings.warn(f"Missing keys in checkpoint (not loaded): {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")

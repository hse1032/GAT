import torch


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    if resolution == 128:
        resolution = 256
    if resolution not in (256, 512):
        raise ValueError(f"Unsupported resolution: {resolution}")

    encoders, encoder_types, architectures = [], [], []
    for enc_name in enc_type.split(","):
        encoder_type, architecture, model_config = enc_name.split("-")
        if "dinov2" not in encoder_type:
            raise NotImplementedError("This refactor keeps only DINOv2 encoders.")

        import timm

        model_name = f"dinov2_vit{model_config}14_reg" if "reg" in encoder_type else f"dinov2_vit{model_config}14"
        encoder = torch.hub.load("facebookresearch/dinov2", model_name)
        del encoder.head
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data,
            [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(device)
        encoder.eval()

        encoders.append(encoder)
        encoder_types.append(encoder_type)
        architectures.append(architecture)

    return encoders, encoder_types, architectures


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "decoder_blocks" in key:
            parts = key.split(".")
            parts[0] = "blocks"
            parts[1] = str(int(parts[1]) + encoder_depth)
            new_state_dict[".".join(parts)] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

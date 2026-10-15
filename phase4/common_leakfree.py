from contextlib import nullcontext
import torch
import torch.nn.functional as F
from temporal_predictor import multi_horizon_loss

INPUT_FRAMES = 16
REAL_PAST_FRAMES = 8
REAL_FUTURE_FRAMES = 6

IMAGE_SIZE = 384
PATCH_SIZE = 16
TUBELET_SIZE = 2
TEMPORAL_TOKENS = INPUT_FRAMES // TUBELET_SIZE
SPATIAL_TOKENS = IMAGE_SIZE // PATCH_SIZE

CONTEXT_TOKENS = 4
FUTURE_TOKENS = 3

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def prepare_window(sample, start, length):
    video = sample["video"].numpy()
    end = start + length

    if end > video.shape[0]:
        raise RuntimeError(
            f"Requested frames [{start}:{end}], video only has {video.shape[0]} frames"
        )

    video = torch.from_numpy(video[start:end])

    if length < INPUT_FRAMES:
        padding = INPUT_FRAMES - length
        last = video[-1:].repeat(padding, 1, 1, 1)
        video = torch.cat([video, last], dim=0)

    video = video.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    b, c, t, h, w = video.shape

    x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = F.interpolate(
        x,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    x = x.reshape(b, t, c, IMAGE_SIZE, IMAGE_SIZE)
    x = x.permute(0, 2, 1, 3, 4)

    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)

    return (x - mean) / std


def prepare_past_future(sample):
    past = prepare_window(
        sample,
        start=0,
        length=REAL_PAST_FRAMES,
    )

    future = prepare_window(
        sample,
        start=REAL_PAST_FRAMES,
        length=REAL_FUTURE_FRAMES,
    )

    return past, future


def shift_clip(video, dx):
    if dx == 0:
        return video

    b, c, t, h, w = video.shape
    x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    theta = torch.tensor(
        [
            [1.0, 0.0, -2.0 * dx / (w - 1)],
            [0.0, 1.0, 0.0],
        ],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0).repeat(b * t, 1, 1)

    grid = F.affine_grid(
        theta,
        x.size(),
        align_corners=True,
    )

    x = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )

    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def encode_video(encoder, video):
    z = encoder(video)
    b, n, d = z.shape

    expected = TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS

    if n != expected:
        raise RuntimeError(
            f"Unexpected token count {n}, expected {expected}"
        )

    return z.reshape(
        b,
        TEMPORAL_TOKENS,
        SPATIAL_TOKENS,
        SPATIAL_TOKENS,
        d,
    )


def latent_resample(latent, dx_pixels):
    if dx_pixels == 0:
        return latent

    b, t, h, w, d = latent.shape
    x = latent.permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w)

    dx_tokens = dx_pixels / PATCH_SIZE
    normalized_shift = 2.0 * dx_tokens / (w - 1)

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )

    grid = torch.stack(
        [xs + normalized_shift, ys],
        dim=-1,
    )

    grid = grid.unsqueeze(0).repeat(b * t, 1, 1, 1)

    x = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return x.reshape(b, t, d, h, w).permute(0, 1, 3, 4, 2)


def extract_z(encoder, video, phase):
    shifted = shift_clip(video, phase)
    z = encode_video(encoder, shifted)
    return latent_resample(z, phase)


def temporal_pair(past_repr, future_repr):
    context = past_repr[:, :CONTEXT_TOKENS]

    targets = [
        future_repr[:, i].detach()
        for i in range(FUTURE_TOKENS)
    ]

    return context, targets


def representation_loss(prediction, target):
    l1 = F.smooth_l1_loss(prediction, target)

    pred = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = 1.0 - (pred * target).sum(dim=-1).mean()

    return cosine + l1


def world_consistency_loss(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1).mean()


def world_variance_loss(world, target_std=0.5):
    x = world.reshape(-1, world.shape[-1]).float()
    std = x.std(dim=0, unbiased=False)
    return F.relu(target_std - std).mean()


def copy_last_loss(context, targets):
    last = context[:, -1]
    predictions = [last for _ in targets]
    return multi_horizon_loss(predictions, targets)


def relative_gain(pred_loss, copy_loss):
    return (copy_loss - pred_loss) / max(copy_loss, 1e-8)


def cosine_delta(a, b, border=2):
    if border > 0:
        a = a[:, :, border:-border, border:-border]
        b = b[:, :, border:-border, border:-border]

    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)

    return (1.0 - (a * b).sum(dim=-1)).mean()


def freeze_encoder_except_last_blocks(encoder, num_blocks=4):
    for p in encoder.parameters():
        p.requires_grad = False

    if not hasattr(encoder, "blocks"):
        raise RuntimeError("Encoder does not expose encoder.blocks")

    for block in encoder.blocks[-num_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def set_partial_encoder_train_mode(encoder, num_blocks=4):
    encoder.eval()

    for block in encoder.blocks[-num_blocks:]:
        block.train()


def trainable_parameters(model):
    return [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


def trainable_state_dict(model):
    names = {
        name
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
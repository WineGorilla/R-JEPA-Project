import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldGridFactorizer(nn.Module):
    """
    Input:
        z: [B, T, H, W, 768]

    Output:
        world: [B, T, H, W, world_dim]
        grid:  [B, T, H, W, grid_dim]
        grid_logits: [B, num_grid_classes]
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=768,
        world_dim=512,
        grid_dim=128,
        num_grid_classes=4,
        dropout=0.0,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.world_head = nn.Sequential(
            nn.Linear(hidden_dim, world_dim),
            nn.LayerNorm(world_dim),
        )

        self.grid_head = nn.Sequential(
            nn.Linear(hidden_dim, grid_dim),
            nn.LayerNorm(grid_dim),
        )

        self.grid_classifier = nn.Sequential(
            nn.Linear(grid_dim, grid_dim),
            nn.GELU(),
            nn.Linear(grid_dim, num_grid_classes),
        )

    def forward(self, z):

        if z.ndim != 5:
            raise ValueError(
                f"Expected [B,T,H,W,D], got {z.shape}"
            )

        x = self.input_norm(z)

        h = self.trunk(x)

        world = self.world_head(h)
        grid = self.grid_head(h)

        # Pool grid information across space + time
        pooled_grid = grid.mean(
            dim=(1, 2, 3)
        )

        grid_logits = self.grid_classifier(
            pooled_grid
        )

        return world, grid, grid_logits


def world_consistency_loss(
    world_list,
    eps=1e-6,
):
    """
    Different patch-grid phases should produce
    similar World representations.
    """

    reference = F.normalize(
        world_list[0],
        dim=-1,
        eps=eps,
    )

    losses = []

    for world in world_list[1:]:

        world = F.normalize(
            world,
            dim=-1,
            eps=eps,
        )

        cosine = (
            reference * world
        ).sum(dim=-1)

        loss = (
            1.0 - cosine
        ).mean()

        losses.append(loss)

    return torch.stack(
        losses
    ).mean()


def world_variance_loss(
    world,
    target_std=0.5,
    eps=1e-4,
):
    """
    Prevent World branch collapse.

    Encourages feature dimensions to keep
    non-zero variance across tokens.
    """

    x = world.reshape(
        -1,
        world.shape[-1],
    )

    variance = x.var(
        dim=0,
        unbiased=False,
    )

    std = torch.sqrt(
        variance + eps
    )

    loss = F.relu(
        target_std - std
    ).mean()

    return loss
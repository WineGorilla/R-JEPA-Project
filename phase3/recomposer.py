import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldGridRecomposer(nn.Module):
    """
    Combine World + Grid factors
    to reconstruct original V-JEPA latent.

    Input:
        world: [B,T,H,W,Dw]
        grid:  [B,T,H,W,Dg]

    Output:
        z_hat: [B,T,H,W,768]
    """

    def __init__(
        self,
        world_dim=512,
        grid_dim=128,
        hidden_dim=768,
        output_dim=768,
        dropout=0.0,
    ):
        super().__init__()

        input_dim = (
            world_dim
            + grid_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

        self.output_norm = nn.LayerNorm(
            output_dim
        )

    def forward(
        self,
        world,
        grid,
    ):

        if (
            world.shape[:-1]
            != grid.shape[:-1]
        ):
            raise ValueError(
                "World and Grid spatial shapes "
                "must match."
            )

        x = torch.cat(
            [
                world,
                grid,
            ],
            dim=-1,
        )

        z_hat = self.net(x)

        z_hat = self.output_norm(
            z_hat
        )

        return z_hat


def reconstruction_loss(
    pred,
    target,
):
    """
    Smooth-L1 + cosine reconstruction.
    """

    loss_l1 = F.smooth_l1_loss(
        pred,
        target,
    )

    pred_norm = F.normalize(
        pred,
        dim=-1,
    )

    target_norm = F.normalize(
        target,
        dim=-1,
    )

    cosine = (
        pred_norm
        * target_norm
    ).sum(dim=-1)

    loss_cos = (
        1.0 - cosine
    ).mean()

    return (
        loss_l1
        + loss_cos
    )


def cross_recomposition_loss(
    recomposer,
    world_list,
    grid_list,
    target_list,
):
    """
    Core idea:

        R(W_a, G_b) -> Z_b

    If World really stores scene/world information
    and Grid stores tokenizer/grid condition,
    World from one intervention should work together
    with Grid from another intervention.
    """

    losses = []

    world_ref = world_list[0]
    grid_ref = grid_list[0]
    target_ref = target_list[0]

    for i in range(
        1,
        len(world_list),
    ):

        # ----------------------------------------
        # World from reference
        # Grid from phase i
        # -> reconstruct phase i
        # ----------------------------------------

        pred_i = recomposer(
            world_ref,
            grid_list[i],
        )

        loss_i = reconstruction_loss(
            pred_i,
            target_list[i],
        )

        losses.append(loss_i)

        # ----------------------------------------
        # World from phase i
        # Grid from reference
        # -> reconstruct reference
        # ----------------------------------------

        pred_ref = recomposer(
            world_list[i],
            grid_ref,
        )

        loss_ref = reconstruction_loss(
            pred_ref,
            target_ref,
        )

        losses.append(loss_ref)

    return torch.stack(
        losses
    ).mean()
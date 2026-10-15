import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPredictor(nn.Module):
    """
    Temporal predictor applied independently at every spatial position.

    Input:
        x: [B, T_context, H, W, D]

    Output:
        list of predictions:
        [
            [B, H, W, output_dim],   # t+1
            [B, H, W, output_dim],   # t+2
            ...
        ]
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        max_context=8,
        num_horizons=3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.temporal_pos_embed = nn.Parameter(
            torch.randn(
                1,
                max_context,
                hidden_dim,
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.horizon_embed = nn.Parameter(
            torch.randn(
                num_horizons,
                hidden_dim,
            ) * 0.02
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x:
            [B, T, H, W, D]
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,H,W,D], got {x.shape}"
            )

        B, T, H, W, D = x.shape

        if T > self.temporal_pos_embed.shape[1]:
            raise ValueError(
                f"T={T} exceeds max_context="
                f"{self.temporal_pos_embed.shape[1]}"
            )

        # [B,T,H,W,D]
        # -> [B,H,W,T,D]
        # -> [B*H*W,T,D]
        x = x.permute(
            0,
            2,
            3,
            1,
            4,
        ).contiguous()

        x = x.reshape(
            B * H * W,
            T,
            D,
        )

        x = self.input_proj(x)

        x = (
            x
            + self.temporal_pos_embed[:, :T]
        )

        x = self.transformer(x)
        x = self.norm(x)

        # use last context state
        h = x[:, -1]

        predictions = []

        for k in range(self.num_horizons):

            hk = (
                h
                + self.horizon_embed[k]
            )

            pred = self.pred_head(hk)

            pred = pred.reshape(
                B,
                H,
                W,
                self.output_dim,
            )

            predictions.append(pred)

        return predictions


def prediction_loss(
    prediction,
    target,
    cosine_weight=1.0,
    smooth_l1_weight=1.0,
):
    """
    prediction:
        [B,H,W,D]

    target:
        [B,H,W,D]
    """

    loss_l1 = F.smooth_l1_loss(
        prediction,
        target,
    )

    pred_norm = F.normalize(
        prediction,
        dim=-1,
    )

    target_norm = F.normalize(
        target,
        dim=-1,
    )

    cosine_similarity = (
        pred_norm
        * target_norm
    ).sum(dim=-1)

    loss_cos = (
        1.0
        - cosine_similarity
    ).mean()

    total = (
        cosine_weight * loss_cos
        + smooth_l1_weight * loss_l1
    )

    return {
        "total": total,
        "cosine": loss_cos,
        "smooth_l1": loss_l1,
    }


def multi_horizon_loss(
    predictions,
    targets,
):
    total_list = []
    cosine_list = []
    l1_list = []

    for pred, target in zip(
        predictions,
        targets,
    ):
        losses = prediction_loss(
            pred,
            target,
        )

        total_list.append(
            losses["total"]
        )

        cosine_list.append(
            losses["cosine"]
        )

        l1_list.append(
            losses["smooth_l1"]
        )

    return {
        "total": torch.stack(
            total_list
        ).mean(),

        "cosine": torch.stack(
            cosine_list
        ).mean(),

        "smooth_l1": torch.stack(
            l1_list
        ).mean(),
    }
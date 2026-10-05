import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18
from common import CFG, ROIDataset, fit_normalization, load_manifest, set_seed


class TubeMAE(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=4), nn.GELU(),
            nn.ConvTranspose3d(256, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1),
        )

    @staticmethod
    def tube_mask(volume):
        batch, _, _, height, width = volume.shape
        grid = 8
        ratios = torch.empty(batch, 1, 1, device=volume.device).uniform_(0.75, 0.90)
        coarse = torch.rand(batch, grid, grid, device=volume.device) < ratios
        mask = coarse.repeat_interleave(height // grid, 1).repeat_interleave(width // grid, 2)
        mask = mask[:, None, None].expand_as(volume)
        return volume.masked_fill(mask, 0), mask

    def forward(self, volume):
        masked, mask = self.tube_mask(volume)
        latent = self.encoder(masked).squeeze(2)
        reconstruction = self.decoder(latent)
        return ((reconstruction - volume[:, :1]) ** 2)[mask[:, :1]].mean()


def train(train_frame, output: Path, epochs=40):
    set_seed()
    loader = DataLoader(
        ROIDataset(train_frame, fit_normalization(train_frame), load_volume=True),
        batch_size=8, shuffle=True, num_workers=4, drop_last=True,
    )
    model = TubeMAE().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        for batch in loader:
            loss = model(batch["volume"].to(CFG.device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-wafers", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest()
    train(manifest[manifest.wafer.astype(str).isin(args.train_wafers)], args.output)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from common import CFG, ROIDataset, CSVLogger, load_split, set_seed

class TubeMAE(nn.Module):
    def __init__(self, mask_ratio=0.85):
        super().__init__()
        self.mask_ratio = mask_ratio
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=4),        
            nn.GELU(),
            nn.ConvTranspose3d(256, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1),  
        )

    def tube_mask(self, x):
        B = x.shape[0]
        g = 8
        grid = torch.rand(B, g, g, device=x.device) < self.mask_ratio  
        mask = grid.repeat_interleave(64 // g, 1).repeat_interleave(64 // g, 2)
        mask = mask.unsqueeze(1).unsqueeze(1)                          
        mask = mask.expand(-1, 3, 64, -1, -1)                            
        return x * (~mask), mask

    def forward(self, x):
        xm, mask = self.tube_mask(x)
        feat = self.encoder(xm)          
        feat = feat.squeeze(2)            
        recon = self.decoder(feat)        
        target = x[:, :1]
        loss = ((recon - target) ** 2)[mask[:, :1]].mean()  
        return loss

def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    import pandas as pd
    df_all = pd.concat([df_tr, df_vi])
    ds = ROIDataset(df_all, norm, load_volume=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    model = TubeMAE().to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    logger = CSVLogger("mae")

    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 41):                 
        model.train()
        tot, n = 0.0, 0
        for batch in dl:
            loss = model(batch["volume"].to(CFG.DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        logger.log(epoch=epoch, recon_mse=tot / n)
        print(f"[MAE] epoch {epoch:02d}/40  recon_mse={tot/n:.5f}")
    torch.save(model.encoder.state_dict(), CFG.CKPT_DIR / "mae_r3d18.pt")

if __name__ == "__main__":
    main()

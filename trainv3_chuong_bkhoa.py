import argparse, os, math, random, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def series_id(lat, lon, prec=3):
    return f"{round(float(lat),prec)}_{round(float(lon),prec)}"

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Latitude", "Longitude", "date"]).reset_index(drop=True)
    doy = df["date"].dt.dayofyear.values
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    mon = df["date"].dt.month.values
    df["mon_sin"] = np.sin(2*np.pi*mon/12.0)
    df["mon_cos"] = np.cos(2*np.pi*mon/12.0)
    df["sid"] = [series_id(a,b) for a,b in zip(df["Latitude"], df["Longitude"])]
    return df

def time_split_by_date(df: pd.DataFrame, date_col="date",
                       train_end="2022-12-31", valid_end="2024-12-31"):
    train = df[df[date_col] <= pd.to_datetime(train_end)].copy()
    valid = df[(df[date_col] > pd.to_datetime(train_end)) &
               (df[date_col] <= pd.to_datetime(valid_end))].copy()
    test  = df[df[date_col] > pd.to_datetime(valid_end)].copy()
    return train, valid, test

class WindowDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_cols: List[str],
                 window: int,
                 horizon: int,
                 sid_col: str = "sid"):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols  = target_cols
        self.window = window
        self.horizon = horizon
        self.sid_col = sid_col

        self.groups = []
        for sid, g in df.groupby(sid_col):
            g = g.sort_values("date")
            values = g.reset_index(drop=True)
            for i in range(len(values) - (window + horizon) + 1):
                self.groups.append((sid, i, values))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        sid, i, g = self.groups[idx]
        sl = slice(i, i + self.window)
        tl = slice(i + self.window, i + self.window + self.horizon)

        x = g.loc[sl, self.feature_cols].values.astype(np.float32)  
        y = g.loc[tl, self.target_cols].values.astype(np.float32)   
        y_in = np.zeros_like(y)
        y_in[1:] = y[:-1]
        return torch.from_numpy(x), torch.from_numpy(y_in), torch.from_numpy(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 3700, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        T = x.size(0)
        x = x + self.pe[:T]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.src_proj = nn.Linear(d_input, d_model)
        self.tgt_proj = nn.Linear(d_output, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.pos_dec = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=False)  
        self.head = nn.Linear(d_model, d_output)

    def generate_square_subsequent_mask(self, size: int, device):
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        src = self.src_proj(src)
        tgt = self.tgt_proj(tgt_in)
        src = src.transpose(0,1)
        tgt = tgt.transpose(0,1)
        src = self.pos_enc(src)
        tgt = self.pos_dec(tgt)
        device = src.device
        H = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(H, device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.head(out)         
        return out.transpose(0,1)    


@dataclass
class Config:
    csv: str
    target: List[str]
    window: int
    horizon: int
    batch_size: int
    d_model: int
    heads: int
    layers: int
    ff: int
    dropout: float
    lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    num_workers: int

def make_feature_target_cols(df: pd.DataFrame, targets: List[str]) -> Tuple[List[str], List[str]]:
    cols = ["T2M","QV2M","PS","WS10M","PRECTOTCORR","CLRSKY_SFC_SW_DWN",
            "Latitude","Longitude","hour","day","month","doy_sin","doy_cos","mon_sin","mon_cos"]
    cols = [c for c in cols if c in df.columns]
    target_cols = targets
    feature_cols = cols
    return feature_cols, target_cols

def build_scalers(train_df, feature_cols, target_cols):
    Xs, Ys = train_df[feature_cols].values, train_df[target_cols].values
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit(Xs); y_scaler.fit(Ys)
    return x_scaler, y_scaler

def apply_scalers(df, feature_cols, target_cols, x_scaler, y_scaler):
    df = df.copy()
    df[feature_cols] = x_scaler.transform(df[feature_cols].values)
    df[target_cols]  = y_scaler.transform(df[target_cols].values)
    return df

def collate(batch):
    xs, ys_in, ys = zip(*batch)
    return torch.stack(xs,0), torch.stack(ys_in,0), torch.stack(ys,0)

def mae_rmse(y_true, y_pred):
    mae = (y_true - y_pred).abs().mean().item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
    return mae, rmse

def train_loop(model, loader, opt, scaler_amp, device):
    model.train()
    total_loss = 0.0
    crit = nn.MSELoss()
    for x, y_in, y in loader:
        x, y_in, y = x.to(device), y_in.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(x, y_in)
            loss = crit(out, y)
        scaler_amp.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    crit = nn.MSELoss()
    total_loss = 0.0
    mae_total, rmse_total, n = 0.0, 0.0, 0
    for x, y_in, y in loader:
        x, y_in, y = x.to(device), y_in.to(device), y.to(device)
        out = model(x, y_in)
        loss = crit(out, y)
        total_loss += loss.item() * x.size(0)
        mae, rmse = mae_rmse(y, out)
        mae_total += mae * x.size(0)
        rmse_total += rmse * x.size(0)
        n += x.size(0)
    return total_loss / len(loader.dataset), mae_total/n, rmse_total/n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to CSV (e.g., datatrainai.csv)")
    p.add_argument("--target", type=str, default="T2M", help="Comma-separated targets, e.g., T2M or T2M,PRECTOTCORR")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--window", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    cfg = Config(
        csv=args.csv,
        target=[t.strip() for t in args.target.split(",") if t.strip()],
        window=args.window,
        horizon=args.horizon,
        batch_size=args.batch_size,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff=args.ff,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers
    )

    os.makedirs("checkpoints", exist_ok=True)
    df = pd.read_csv(cfg.csv)
    df = add_time_features(df)
    train_df, valid_df, test_df = time_split_by_date(df, "date",
                                                     train_end="2022-12-31",
                                                     valid_end="2024-12-31")
    feature_cols, target_cols = make_feature_target_cols(df, cfg.target)
    x_scaler, y_scaler = build_scalers(train_df, feature_cols, target_cols)
    train_df = apply_scalers(train_df, feature_cols, target_cols, x_scaler, y_scaler)
    valid_df = apply_scalers(valid_df, feature_cols, target_cols, x_scaler, y_scaler)
    test_df  = apply_scalers(test_df,  feature_cols, target_cols, x_scaler, y_scaler)
    ds_tr = WindowDataset(train_df, feature_cols, target_cols, cfg.window, cfg.horizon)
    ds_va = WindowDataset(valid_df, feature_cols, target_cols, cfg.window, cfg.horizon)
    ds_te = WindowDataset(test_df,  feature_cols, target_cols, cfg.window, cfg.horizon)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqTransformer(
        d_input=len(feature_cols),
        d_output=len(target_cols),
        d_model=cfg.d_model,
        nhead=cfg.heads,
        num_encoder_layers=cfg.layers,
        num_decoder_layers=cfg.layers,
        dim_feedforward=cfg.ff,
        dropout=cfg.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        warmup = 2000
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, cfg.max_epochs*len(dl_tr) - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val = float("inf"); patience = cfg.patience; best_path = "checkpoints/best_transformer.pt"
    global_step = 0

    for epoch in range(1, cfg.max_epochs+1):
        train_loss = train_loop(model, dl_tr, opt, scaler_amp, device)
        sch.step()
        val_loss, val_mae, val_rmse = eval_loop(model, dl_va, device)

        print(f"Epoch {epoch:03d} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f} | "
              f"val MAE {val_mae:.5f} | val RMSE {val_rmse:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "x_scaler_mean": x_scaler.mean_.tolist(),
                        "x_scaler_scale": x_scaler.scale_.tolist(),
                        "y_scaler_mean": y_scaler.mean_.tolist(),
                        "y_scaler_scale": y_scaler.scale_.tolist(),
                        "feature_cols": feature_cols,
                        "target_cols": target_cols,
                        "cfg": vars(cfg)}, best_path)
            patience = cfg.patience
            print(f"  ✅ Saved best to {best_path}")
        else:
            patience -= 1
            if patience == 0:
                print("  ⏹ Early stopping.")
                break
        global_step += len(dl_tr)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_mae, test_rmse = eval_loop(model, dl_te, device)
    print(f"\nTEST | MSE {test_loss:.5f} | MAE {test_mae:.5f} | RMSE {test_rmse:.5f}")

if __name__ == "__main__":
    main()

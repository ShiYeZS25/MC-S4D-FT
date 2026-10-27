
import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from s4d_ft import S4DFT, NSELoss, setup_optimizer, calc_nse, count_parameters
from s4d_ft_mc import S4DFT_MC
from s4d_ft_softplus import S4DFT_Softplus
from data_pipeline import (
    CamelsH5, CamelsTXT,
    add_camels_attributes, create_h5_files,
    get_basin_list, rescale_features,
)


def build_model(cfg_dict: Dict, device=None):
    d_input = 32
    model_type = cfg_dict.get("model", "siso")

    if model_type == "siso":
        ssm_cfg = {
            "lr_min": cfg_dict["lr_min"], "lr": cfg_dict["lr"],
            "d_state": cfg_dict["d_state"],
            "min_dt": 0.001, "max_dt": 1.0, "lr_dt": 0.0,
            "cfr": cfg_dict["cfr"], "cfi": cfg_dict["cfi"],
            "wd": cfg_dict.get("wd", 0.0),
        }
        model = S4DFT(
            d_input=d_input, d_output=1,
            d_model=cfg_dict["d_model"], n_layers=cfg_dict["n_layers"],
            dropout=cfg_dict["ssm_dropout"], cfg=ssm_cfg, prenorm=False,
        )
        label = f"S4D-FT (SISO, {cfg_dict['n_layers']}层)"


    elif model_type == "mc":
        ssm_cfg = {
            "lr_min": cfg_dict["lr_min"], "lr": cfg_dict["lr"],
            "d_state": cfg_dict["d_state"],
            "min_dt": 0.001, "max_dt": 1.0, "lr_dt": 0.0,
            "cfr": cfg_dict["cfr"], "cfi": cfg_dict["cfi"],
            "wd": cfg_dict.get("wd", 0.0),
        }
        model = S4DFT_MC(
            d_input=d_input, d_output=1,
            d_model=cfg_dict["d_model"], n_layers=cfg_dict["n_layers"],
            dropout=cfg_dict["ssm_dropout"], cfg=ssm_cfg, prenorm=False,
            k_max=cfg_dict.get("k_max", 0.1),
        )
        label = f"S4D-FT MC ({cfg_dict['n_layers']}层, k_max={cfg_dict.get('k_max', 0.1)})"


    elif model_type == "softplus":
        ssm_cfg = {
            "lr_min": cfg_dict["lr_min"], "lr": cfg_dict["lr"],
            "d_state": cfg_dict["d_state"],
            "min_dt": 0.001, "max_dt": 1.0, "lr_dt": 0.0,
            "cfr": cfg_dict["cfr"], "cfi": cfg_dict["cfi"],
            "wd": cfg_dict.get("wd", 0.0),
        }

        model = S4DFT_Softplus(
            d_input=d_input, d_output=1,
            d_model=cfg_dict["d_model"], n_layers=cfg_dict["n_layers"],
            dropout=cfg_dict["ssm_dropout"], cfg=ssm_cfg, prenorm=False,
        )
        label = f"S4D-FT + Softplus ({cfg_dict['n_layers']}层)"


    else:
        raise ValueError(f'Unsupported model: {model_type}')

    if device is not None:
        model = model.to(device)

    n_params = count_parameters(model)
    print(f"\n{label}: {n_params:,} 可训练参数")
    return model




TIME_SETTINGS = {
    'seq_length':  365,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end':   pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start':   pd.to_datetime('01101984', format='%d%m%Y'),
    'val_end':     pd.to_datetime('30091989', format='%d%m%Y'),
    'test_start':  pd.to_datetime('01101989', format='%d%m%Y'),
    'test_end':    pd.to_datetime('30091999', format='%d%m%Y'),
 }


def get_args() -> Dict:
    parser = argparse.ArgumentParser(description="S4D-FT 降雨径流模拟")
    parser.add_argument('mode', choices=["train", "evaluate"],
                        help="运行模式: train 或 evaluate")
    parser.add_argument('--camels_root', type=str,
                        default='./data/basin_dataset_public_v1p2',
                        help="CAMELS 数据集根目录 (含 basin_mean_forcing/ 和 usgs_streamflow/)")

    parser.add_argument('--seed', type=int, default=300, help="随机种子")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID (-1=CPU)")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="DataLoader 线程数 (Windows 建议 0，Linux 可设 8)")
    parser.add_argument('--cache_data', action='store_true', default=True,
                        help="将训练数据全部加载到内存")

    parser.add_argument('--run_dir', type=str, default=None, help="评估时指定的 run 目录")
    parser.add_argument('--epoch_num', type=int, default=49, help="评估的 epoch 编号")
    parser.add_argument('--auto_eval', action='store_true', help="训练后自动评估")

    parser.add_argument('--basin_file', type=str, default=None,
                        help="流域 ID 列表文件 (默认 data/basin_list.txt)")

    parser.add_argument('--lr', type=float, default=0.0004, help="全局学习率")
    parser.add_argument('--lr_min', type=float, default=0.0004, help="SSM 参数学习率")
    parser.add_argument('--weight_decay', type=float, default=0.03, help="权重衰减")
    parser.add_argument('--wd', type=float, default=0.0, help="SSM 参数权重衰减")

    parser.add_argument('--epochs', type=int, default=50, help="训练轮数")
    parser.add_argument('--epochs_scheduler', type=int, default=50, help="调度器总轮数")
    parser.add_argument('--warmup', type=int, default=0, help="预热轮数")

    parser.add_argument('--batch_size', type=int, default=128, help="训练 batch size")

    parser.add_argument('--model', type=str, default='siso',
                        choices=['siso', 'mc', 'softplus'],
                        help="Model: siso, mc, or softplus")
    parser.add_argument('--n_layers', type=int, default=None,
                        help="层数 (默认: siso=6, mimo_va=4)")
    parser.add_argument('--d_model', type=int, default=128, help="模型隐藏维度")
    parser.add_argument('--d_state', type=int, default=None,
                        help="SSM 状态维度 (默认: siso=128, mimo_va=256)")
    parser.add_argument('--ssm_dropout', type=float, default=0.12, help="Dropout 率")
    parser.add_argument('--cfr', type=float, default=10.0, help="频率调谐因子-实部")
    parser.add_argument('--cfi', type=float, default=10.0, help="频率调谐因子-虚部")
    parser.add_argument('--k_max', type=float, default=0.1,
                        help="MC readout 基流系数上限 (用于 --model mc/lstm_mc/transformer_mc)")

    parser.add_argument('--use_mse', action='store_true', help="使用 MSE 替代 NSE Loss")
    parser.add_argument('--clip_value', type=float, default=1.0, help="梯度裁剪阈值")

    args = parser.parse_args()
    cfg = vars(args)

    model_defaults = {'siso': {'n_layers': 6, 'd_state': 128}, 'mc': {'n_layers': 6, 'd_state': 128}, 'softplus': {'n_layers': 6, 'd_state': 128}}
    defaults = model_defaults[cfg["model"]]
    if cfg["n_layers"] is None:
        cfg["n_layers"] = defaults["n_layers"]
    if cfg["d_state"] is None:
        cfg["d_state"] = defaults["d_state"]

    cfg.update(TIME_SETTINGS)

    if cfg["gpu"] >= 0 and torch.cuda.is_available():
        cfg["device"] = torch.device(f"cuda:{cfg['gpu']}")
    else:
        cfg["device"] = torch.device("cpu")

    cfg["camels_root"] = Path(cfg["camels_root"])

    return cfg



def train(cfg: Dict):

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = cfg["device"]
    print(f"\n设备: {device}")

    basins = get_basin_list(cfg["basin_file"])
    print(f"流域数: {len(basins)}")

    now = datetime.now()
    run_name = f"run_{now:%Y%m%d_%H%M%S}_seed{cfg['seed']}"
    cfg["run_dir"] = Path("runs") / run_name
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    (cfg["run_dir"] / "data" / "train").mkdir(parents=True, exist_ok=True)
    (cfg["run_dir"] / "data" / "val").mkdir(parents=True, exist_ok=True)
    print(f"Run 目录: {cfg['run_dir']}")

    db_path = str(cfg["run_dir"] / "attributes.db")
    cfg["db_path"] = db_path
    add_camels_attributes(cfg["camels_root"], db_path=db_path)

    train_h5 = cfg["run_dir"] / "data" / "train" / "train_data.h5"
    val_h5 = cfg["run_dir"] / "data" / "val" / "val_data.h5"
    cfg["train_file"] = train_h5
    cfg["val_file"] = val_h5
    _save_config(cfg)

    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=train_h5,
        basins=basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"],
    )
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=val_h5,
        basins=basins,
        dates=[cfg["val_start"], cfg["val_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"],
    )

    train_ds = CamelsH5(
        h5_file=str(train_h5),
        basins=basins,
        db_path=db_path,
        concat_static=True,
        cache=cfg["cache_data"],
        no_static=False,
    )
    val_ds = CamelsH5(
        h5_file=str(val_h5),
        basins=basins,
        db_path=db_path,
        concat_static=True,
        cache=cfg["cache_data"],
        no_static=False,
    )
    n_train = len(train_ds)
    n_val = len(val_ds)
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1024, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=use_cuda,
    )
    print(f"训练样本: {n_train}, 验证样本: {n_val}")

    model = build_model(cfg, device=device)

    optimizer, scheduler = setup_optimizer(
        model, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs_scheduler"], warmup_epochs=cfg["warmup"],
    )

    if cfg["use_mse"]:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = NSELoss(eps=0.1)

    print(f"\n{'='*60}")
    print(f"开始训练: {cfg['epochs']} 轮, batch_size={cfg['batch_size']}")
    print(f"{'='*60}")

    best_val = float('inf')
    loss_log = []

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y, q_std in tqdm(train_loader, desc=f"Epoch {epoch:2d} [train]",
                                 leave=False):
            x, y, q_std = x.to(device), y.to(device), q_std.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y) if cfg["use_mse"] else loss_fn(out, y, q_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, q_std in val_loader:
                x, y, q_std = x.to(device), y.to(device), q_std.to(device)
                out = model(x)
                loss = loss_fn(out, y) if cfg["use_mse"] else loss_fn(out, y, q_std)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        lr_now = optimizer.param_groups[0]['lr']
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        scheduler.step()

        loss_log.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'lr': lr_now,
        })

        marker = " ★" if is_best else ""
        print(f"Epoch {epoch:3d}/{cfg['epochs']-1}  "
              f"Train: {train_loss:.5f}  Val: {val_loss:.5f}  "
              f"LR: {lr_now:.6f}{marker}")

        torch.save(model.state_dict(), cfg["run_dir"] / f"model_epoch{epoch}.pt")

    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(cfg["run_dir"] / "loss_log.csv", index=False)
    print(f"\n训练完成。模型保存于 {cfg['run_dir']}")
    print(f"最佳验证损失: {best_val:.5f}")

    if cfg.get("auto_eval", False):
        print("\n" + "=" * 60)
        print("自动评估最后一个 epoch...")
        cfg["epoch_num"] = cfg["epochs"] - 1
        evaluate(cfg)

    return cfg["run_dir"]



def evaluate(cfg: Dict):

    device = cfg["device"]
    run_dir = Path(cfg["run_dir"])

    with open(run_dir / "cfg.json", 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list(cfg.get("basin_file"))
    print(f"\n评估 {len(basins)} 个流域, epoch={cfg['epoch_num']}")

    db_path = str(run_dir / "attributes.db")
    train_h5 = str(run_dir / "data" / "train" / "train_data.h5")
    ds_train = CamelsH5(
        h5_file=train_h5, basins=basins, db_path=db_path,
        concat_static=True, cache=False,
    )
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    model = build_model(run_cfg, device=device)

    weight_file = run_dir / f"model_epoch{cfg['epoch_num']}.pt"
    model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
    model.eval()
    print(f"已加载权重: {weight_file}")

    date_range = pd.date_range(
        start=TIME_SETTINGS["test_start"],
        end=TIME_SETTINGS["test_end"],
    )

    results = {}
    nse_values = []
    failed_basins = []

    for basin in tqdm(basins, desc="逐流域评估"):
        try:
            ds_test = CamelsTXT(
                camels_root=cfg["camels_root"],
                basin=basin,
                dates=[TIME_SETTINGS["test_start"], TIME_SETTINGS["test_end"]],
                is_train=False,
                seq_length=run_cfg.get("seq_length", 365),
                with_attributes=True,
                attribute_means=means,
                attribute_stds=stds,
                concat_static=True,
                db_path=db_path,
            )
        except Exception as e:
            failed_basins.append((basin, str(e)))
            continue

        loader = DataLoader(ds_test, batch_size=1024, shuffle=False,
                            num_workers=cfg.get("num_workers", 0))

        preds_list, obs_list = [], []
        with torch.no_grad():
            for x, y, q_std in loader:
                x = x.to(device)
                p = model(x).cpu()
                preds_list.append(p)
                obs_list.append(y)

        preds = torch.cat(preds_list, dim=0).numpy()
        obs = torch.cat(obs_list, dim=0).numpy()

        preds = rescale_features(preds, variable='output')
        preds[preds < 0] = 0

        try:
            nse = calc_nse(obs.flatten(), preds.flatten())
            nse_values.append(nse)
        except Exception:
            nse = float('nan')
            nse_values.append(nse)

        n = min(len(date_range), len(obs), len(preds))
        df = pd.DataFrame(
            {'qobs': obs.flatten()[:n], 'qsim': preds.flatten()[:n]},
            index=date_range[:n],
        )
        results[basin] = df

    nse_arr = np.array([v for v in nse_values if not np.isnan(v)])
    print(f"\n{'='*60}")
    print(f"评估结果 (epoch {cfg['epoch_num']}):")
    print(f"  有效流域数: {len(nse_arr)} / {len(basins)}")
    print(f"  失败流域数: {len(failed_basins)}")
    print(f"{'='*60}")
    if len(nse_arr) > 0:
        print(f"  NSE 均值:   {np.mean(nse_arr):.4f}")
        print(f"  NSE 中位数: {np.median(nse_arr):.4f}")
        print(f"  NSE > 0:    {np.sum(nse_arr > 0)} ({100*np.mean(nse_arr > 0):.1f}%)")
        print(f"  NSE > 0.5:  {np.sum(nse_arr > 0.5)} ({100*np.mean(nse_arr > 0.5):.1f}%)")
        print(f"  NSE > 0.7:  {np.sum(nse_arr > 0.7)} ({100*np.mean(nse_arr > 0.7):.1f}%)")
        print(f"  NSE > 0.8:  {np.sum(nse_arr > 0.8)} ({100*np.mean(nse_arr > 0.8):.1f}%)")
        for q in [10, 25, 50, 75, 90]:
            print(f"  NSE P{q:02d}:    {np.percentile(nse_arr, q):.4f}")
    print(f"{'='*60}")

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    pkl_file = results_dir / f"results_epoch{cfg['epoch_num']}.pkl"
    with open(pkl_file, 'wb') as fp:
        pickle.dump(results, fp)

    nse_df = pd.DataFrame({
        'basin': [b for b, _ in zip(basins, nse_values) if b not in [fb[0] for fb in failed_basins]],
        'nse': [v for v in nse_values],
    })
    nse_file = results_dir / f"nse_epoch{cfg['epoch_num']}.csv"
    nse_df.to_csv(nse_file, index=False)

    summary = {
        'epoch': cfg['epoch_num'],
        'n_basins': len(nse_arr),
        'nse_mean': float(np.mean(nse_arr)) if len(nse_arr) > 0 else None,
        'nse_median': float(np.median(nse_arr)) if len(nse_arr) > 0 else None,
    }
    with open(results_dir / f"summary_epoch{cfg['epoch_num']}.json", 'w') as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n结果已保存至 {results_dir}")

    if failed_basins:
        print(f"\n失败流域:")
        for b, e in failed_basins[:10]:
            print(f"  {b}: {e}")
        if len(failed_basins) > 10:
            print(f"  ... 共 {len(failed_basins)} 个")



def _save_config(cfg: Dict):
    serial = {}
    for k, v in cfg.items():
        if isinstance(v, (Path,)):
            serial[k] = str(v)
        elif isinstance(v, pd.Timestamp):
            serial[k] = v.strftime("%d%m%Y")
        elif isinstance(v, torch.device):
            serial[k] = str(v)
        else:
            serial[k] = v
    with open(cfg["run_dir"] / "cfg.json", 'w') as fp:
        json.dump(serial, fp, indent=4, sort_keys=True)



if __name__ == "__main__":
    cfg = get_args()

    if cfg["mode"] == "train":
        train(cfg)
    elif cfg["mode"] == "evaluate":
        if cfg["run_dir"] is None:
            print("错误: 评估模式需要指定 --run_dir")
            sys.exit(1)
        cfg["run_dir"] = Path(cfg["run_dir"])
        cfg["camels_root"] = Path(cfg["camels_root"])
        evaluate(cfg)

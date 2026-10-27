
import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



SCALER = {
    'input_means': np.array([3.015, 357.68, 10.864, 10.864, 1055.533]),
    'input_stds':  np.array([7.573, 129.878, 10.932, 10.932, 705.998]),
    'output_mean': np.array([1.49996196]),
    'output_std':  np.array([3.62443672]),
}

INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover',
    'high_prec_timing', 'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio',
    'stream_elas', 'slope_fdc', 'baseflow_index', 'hfd_mean', 'q5', 'q95',
    'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq',
    'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac',
]



def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    if variable == 'inputs':
        return (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        return (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"未知变量类型: {variable}")


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    if variable == 'inputs':
        return feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        return feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"未知变量类型: {variable}")



def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    T, n_features = x.shape
    n_samples = T - seq_length + 1

    x_strides = x.strides
    x_new = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_samples, seq_length, n_features),
        strides=(x_strides[0], x_strides[0], x_strides[1]),
    ).copy()

    y_new = y[seq_length - 1:, :1].copy()

    return x_new, y_new



def load_forcing(camels_root: Path, basin: str) -> Tuple[pd.DataFrame, int]:
    forcing_path = camels_root / 'basin_mean_forcing' / 'nldas_extended'
    if not forcing_path.exists():
        forcing_path = camels_root / 'basin_mean_forcing' / 'nldas'
        if not forcing_path.exists():
            raise RuntimeError(
                f'未找到强迫数据目录。需要 nldas_extended 或 nldas 目录于\n'
                f'  {camels_root / "basin_mean_forcing"}'
            )
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'未找到流域 {basin} 的强迫文件于 {forcing_path}')
    file_path = file_path[0]

    df = pd.read_csv(file_path, sep=r'\s+', header=3)
    dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    if 'Tmax(C)' not in df.columns and 'Tmin(C)' not in df.columns:
        if 'Tmean(C)' in df.columns:
            print(f"  警告: 流域 {basin} 使用 Tmean 替代 Tmax/Tmin（建议用 nldas_extended）")
            df['Tmax(C)'] = df['Tmean(C)']
            df['Tmin(C)'] = df['Tmean(C)']
        else:
            raise RuntimeError(
                f"流域 {basin} 的强迫文件缺少温度列。"
                f"可用列: {list(df.columns)}"
            )

    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(float(content[2].strip()))

    return df, area


def load_discharge(camels_root: Path, basin: str, area: int) -> pd.Series:
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'未找到流域 {basin} 的径流文件')
    file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)
    return df.QObs



def add_camels_attributes(camels_root: Path, db_path: str):
    attributes_path = camels_root / 'camels_attributes_v2.0'
    if not attributes_path.exists():
        raise RuntimeError(f"属性文件夹未找到: {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        df = df_temp.copy() if df is None else pd.concat([df, df_temp], axis=1)

    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    with sqlite3.connect(db_path) as conn:
        df.to_sql('basin_attributes', conn, if_exists='replace')
    print(f"流域属性已存入 {db_path}")


def load_attributes(db_path: str, basins: List[str],
                    drop_lat_lon: bool = True) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1, errors='ignore')

    drop_names = [c for c in df.columns if c in INVALID_ATTR]
    df = df.drop(drop_names, axis=1)

    return df



def get_basin_list(basin_file: str = None) -> List[str]:
    if basin_file is None:
        basin_file = Path(__file__).parent / "data" / "basin_list.txt"
    with open(basin_file, 'r') as f:
        basins = f.read().splitlines()
    return basins



def create_h5_files(camels_root: Path, out_file: Path, basins: List[str],
                    dates: List[pd.Timestamp], with_basin_str: bool = True,
                    seq_length: int = 365):
    if out_file.exists():
        print(f"H5 文件已存在，跳过: {out_file}")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_x, all_y, all_basins, all_q_stds = [], [], [], []

    for basin in tqdm(basins, desc="打包 H5"):
        try:
            df, area = load_forcing(camels_root, basin)
        except RuntimeError:
            print(f"  跳过流域 {basin}（无强迫数据）")
            continue

        df['QObs(mm/d)'] = load_discharge(camels_root, basin, area)

        start_date = dates[0] - pd.DateOffset(days=seq_length - 1)
        end_date = dates[1]
        df = df[start_date:end_date]

        x = np.array([
            df['PRCP(mm/day)'].values, df['SRAD(W/m2)'].values,
            df['Tmax(C)'].values, df['Tmin(C)'].values, df['Vp(Pa)'].values,
        ]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        x = normalize_features(x, variable='inputs')
        y_norm = normalize_features(y, variable='output')

        x, y_norm = reshape_data(x, y_norm, seq_length)
        _, y_raw = reshape_data(
            np.zeros_like(x[:, 0, :].squeeze() if x.ndim > 2 else x),
            y[seq_length - 1:] if y.shape[0] >= seq_length else y,
            1
        )
        y_orig = y[seq_length - 1:, :1]

        valid = (y_norm.flatten() >= -999) & (~np.isnan(y_norm.flatten()))
        valid_orig = y_orig.flatten() >= 0
        valid = valid & valid_orig
        x = x[valid]
        y_norm = y_norm[valid]
        y_orig = y_orig[valid]

        if len(x) == 0:
            continue

        q_std = np.std(y_orig)
        q_stds = np.full(len(x), q_std, dtype=np.float32)

        all_x.append(x.astype(np.float32))
        all_y.append(y_norm.astype(np.float32))
        all_q_stds.append(q_stds)
        if with_basin_str:
            all_basins.extend([basin] * len(x))

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_q_stds = np.concatenate(all_q_stds, axis=0)

    with h5py.File(out_file, 'w') as f:
        f.create_dataset("input_data", data=all_x, compression="gzip")
        f.create_dataset("target_data", data=all_y, compression="gzip")
        f.create_dataset("q_stds", data=all_q_stds, compression="gzip")
        if with_basin_str:
            str_arr = np.array(all_basins, dtype='S8')
            f.create_dataset("sample_2_basin", data=str_arr, compression="gzip")

    print(f"H5 已保存: {out_file} ({all_x.shape[0]} 样本)")



class CamelsH5(Dataset):

    def __init__(self, h5_file: str, basins: List[str], db_path: str,
                 concat_static: bool = False, cache: bool = True,
                 no_static: bool = False):
        self.h5_file = str(h5_file)
        self.basins = basins
        self.db_path = db_path
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static

        self.df = None
        self.attribute_means = None
        self.attribute_stds = None

        if self.cache:
            self.x, self.y, self.sample_2_basin, self.q_stds = self._preload_data()

        self._load_attributes()

        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.cache:
            x = self.x[idx]
            y = self.y[idx]
            basin = self.sample_2_basin[idx]
            q_std = self.q_stds[idx]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx].decode("ascii")
                q_std = f["q_stds"][idx]

        if not self.no_static:
            attributes = self.df.loc[self.df.index == basin].values
            if self.concat_static:
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)

        x = torch.from_numpy(x.astype(np.float32)) if isinstance(x, np.ndarray) else x
        y = torch.from_numpy(y.astype(np.float32)) if isinstance(y, np.ndarray) else y
        q_std = torch.tensor(q_std, dtype=torch.float32) if not isinstance(q_std, torch.Tensor) else q_std

        return x, y, q_std

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = [s.decode("ascii") for s in f["sample_2_basin"][:]]
            q_stds = f["q_stds"][:]
        return x, y, str_arr, q_stds

    def _load_attributes(self):
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()
        df = (df - self.attribute_means) / self.attribute_stds
        self.df = df

    def get_attribute_means(self):
        return self.attribute_means

    def get_attribute_stds(self):
        return self.attribute_stds



class CamelsTXT(Dataset):

    def __init__(self, camels_root: Path, basin: str, dates: List[pd.Timestamp],
                 is_train: bool, seq_length: int = 365,
                 with_attributes: bool = True,
                 attribute_means: pd.Series = None,
                 attribute_stds: pd.Series = None,
                 concat_static: bool = True,
                 db_path: str = None):
        self.camels_root = camels_root
        self.basin = basin
        self.seq_length = seq_length
        self.is_train = is_train
        self.concat_static = concat_static

        self.x, self.y, self.q_std = self._load_data(dates)

        if with_attributes and concat_static:
            self.attributes = self._load_attributes(db_path, attribute_means, attribute_stds)

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if hasattr(self, 'attributes') and self.concat_static:
            x = torch.cat([x, self.attributes.repeat(self.seq_length, 1)], dim=-1)
        return x, self.y[idx], self.q_std

    def _load_data(self, dates) -> Tuple[torch.Tensor, torch.Tensor, float]:
        df, area = load_forcing(self.camels_root, self.basin)
        df['QObs(mm/d)'] = load_discharge(self.camels_root, self.basin, area)

        start_date = dates[0] - pd.DateOffset(days=self.seq_length - 1)
        df = df[start_date:dates[1]]

        x = np.array([
            df['PRCP(mm/day)'].values, df['SRAD(W/m2)'].values,
            df['Tmax(C)'].values, df['Tmin(C)'].values, df['Vp(Pa)'].values,
        ]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        x = normalize_features(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)

        valid = (y.flatten() >= 0) & (~np.isnan(y.flatten()))
        x, y = x[valid], y[valid]

        q_std = np.std(y)

        if self.is_train:
            y = normalize_features(y, variable='output')

        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
                q_std)

    def _load_attributes(self, db_path, means, stds) -> torch.Tensor:
        df = load_attributes(db_path, [self.basin], drop_lat_lon=True)
        df = (df - means) / stds
        values = df.loc[df.index == self.basin].values
        return torch.from_numpy(values.astype(np.float32))
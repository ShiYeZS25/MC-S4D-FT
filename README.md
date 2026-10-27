# MC-S4D-FT

**Mass-Conserving S4D-FT for Daily Rainfall-Runoff Simulation**

## Configurations

| `model` | Configuration |
| --- | --- |
| `siso` | S4D-FT with a pooled linear readout. |
| `softplus` | S4D-FT with Softplus applied to discharge in physical units before re-standardization. |
| `mc` | MC-S4D-FT with a mass-conserving readout and storage state. |

## Run

Run from the repository root. Set `--camels_root` to the CAMELS directory
containing `basin_mean_forcing/nldas_extended` (or `nldas`),
`usgs_streamflow`, and `camels_attributes_v2.0`. The basin list is
`data/basin_list.txt`. Settings are passed through command-line arguments;
`configs/` contains the seed300 reference configurations.

```console
python -m pip install -r requirements.txt
python train.py train --model siso --seed 300 --gpu 0 --camels_root PATH_TO_CAMELS --auto_eval
python train.py train --model softplus --seed 300 --gpu 0 --camels_root PATH_TO_CAMELS --auto_eval
python train.py train --model mc --seed 300 --gpu 0 --camels_root PATH_TO_CAMELS --auto_eval
python train.py evaluate --run_dir runs/RUN_NAME --epoch_num 49 --gpu 0 --camels_root PATH_TO_CAMELS
python eval_comprehensive.py --run_dir runs/RUN_NAME --epoch 49 --gpu 0 --camels_root PATH_TO_CAMELS
```

Use `--gpu -1` for CPU. 

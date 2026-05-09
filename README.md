# DiffGAD: A Diffusion-based Unsupervised Graph Anomaly Detector

```shell
uv venv -p 3.12
uv pip install pygod torch torch_geometric pyyaml tqdm scikit-learn --torch-backend=cu128

```

This is the implementation of our ICLR 2025 paper:

> Jinghan Li, Yuan Gao, Jinda Lu, Junfeng Fang, Congcong Wen, Hui Lin, Xiang Wang. DiffGAD: A Diffusion-based Unsupervised Graph Anomaly Detector. In ICLR 2025. [Arxiv](https://arxiv.org/abs/2410.06549)

![](method.png)

checkpoints/<dataset>/best.pt
```

Disable checkpoint saving with `--no_save`.

## Data

Put `.mat` files under:

```bash
~/datasets/GAD/mat
```

The loader searches `.mat` files case-insensitively and supports common keys from Awesome-Deep-Graph-Anomaly-Detection datasets:

- adjacency: `Network`, `A`, `adj`, `adjacency`
- features: `Attributes`, `attrb`, `features`, `X`, `x`
- labels: `Label`, `gnd`, `label`, `labels`, `y`

## Usage

General format:

```bash
python main.py --dataset <dataset> --lr <lr> --ae_lr <ae_lr> --ae_alpha <ae_alpha> --ae_dropout <dropout> --proto_alpha <proto_alpha> --weight <weight>
```

Examples migrated from the original `configs/*.yaml`:

```bash
python main.py --dataset book   --ae_dropout 0.1 --ae_lr 0.01 --ae_alpha 0.5 --proto_alpha 0.1   --weight 2.0 --lr 0.004
python main.py --dataset Disney --ae_dropout 0.3 --ae_lr 0.01 --ae_alpha 1.0 --proto_alpha 0.001 --weight 2.0 --lr 0.004
python main.py --dataset Enron  --ae_dropout 0.1 --ae_lr 0.01 --ae_alpha 0.0 --proto_alpha 1.0   --weight 2.0 --lr 0.004
python main.py --dataset Reddit --ae_dropout 0.3 --ae_lr 0.05 --ae_alpha 0.8 --hid_dim 32 --proto_alpha 0.1 --weight 0.8 --lr 0.004
python main.py --dataset weibo  --ae_dropout 0.3 --ae_lr 0.01 --ae_alpha 0.8 --proto_alpha 0.01  --weight 1.0 --lr 0.004
```

For other datasets in `~/datasets/GAD/mat`, use the same format, for example:

```bash
python main.py --dataset YelpChi --lr 0.004 --ae_lr 0.01 --ae_alpha 0.8 --ae_dropout 0.3 --proto_alpha 0.1 --weight 1.0
python main.py --dataset Amazon --lr 0.004 --ae_lr 0.01 --ae_alpha 0.8 --ae_dropout 0.3 --proto_alpha 0.1 --weight 1.0
```

Useful runtime flags:

```bash
--data_dir ~/datasets/GAD/mat
--device 0          # use CUDA:0
--device -1         # use CPU
--num_trials 20
--ae_epochs 300
--diff_epochs 800
--patience 100
--sample_steps 50
--save_dir checkpoints
--no_save
```
import argparse
import os
from pathlib import Path

from DiffGAD import DiffGAD


DEFAULTS = {
    # configs/*.yaml from the original repo, moved here for CLI usage
    "book":      dict(ae_dropout=0.1, ae_lr=0.01, ae_alpha=0.5, hid_dim=None, proto_alpha=0.1,   weight=2.0),
    "books":     dict(ae_dropout=0.1, ae_lr=0.01, ae_alpha=0.5, hid_dim=None, proto_alpha=0.1,   weight=2.0),
    "Disney":    dict(ae_dropout=0.3, ae_lr=0.01, ae_alpha=1.0, hid_dim=None, proto_alpha=0.001, weight=2.0),
    "disney":    dict(ae_dropout=0.3, ae_lr=0.01, ae_alpha=1.0, hid_dim=None, proto_alpha=0.001, weight=2.0),
    "Enron":     dict(ae_dropout=0.1, ae_lr=0.01, ae_alpha=0.0, hid_dim=None, proto_alpha=1.0,   weight=2.0),
    "enron":     dict(ae_dropout=0.1, ae_lr=0.01, ae_alpha=0.0, hid_dim=None, proto_alpha=1.0,   weight=2.0),
    "Reddit":    dict(ae_dropout=0.3, ae_lr=0.05, ae_alpha=0.8, hid_dim=32,   proto_alpha=0.1,   weight=0.8),
    "reddit":    dict(ae_dropout=0.3, ae_lr=0.05, ae_alpha=0.8, hid_dim=32,   proto_alpha=0.1,   weight=0.8),
    "weibo":     dict(ae_dropout=0.3, ae_lr=0.01, ae_alpha=0.8, hid_dim=None, proto_alpha=0.01,  weight=1.0),
}


def get_arguments():
    parser = argparse.ArgumentParser(description="DiffGAD reproduction with CLI hyperparameters.")

    # data / runtime
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or .mat filename, e.g. book, Disney, Enron, Reddit, weibo, YelpChi.")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat",
                        help="Directory containing .mat datasets.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Only one final best checkpoint will be saved here.")
    parser.add_argument("--no_save", action="store_true", help="Do not save checkpoints to disk.")

    # AE
    parser.add_argument("--ae_epochs", type=int, default=300)
    parser.add_argument("--ae_lr", type=float, default=None)
    parser.add_argument("--ae_dropout", type=float, default=None)
    parser.add_argument("--ae_alpha", type=float, default=None)
    parser.add_argument("--hid_dim", type=int, default=None)

    # Diffusion
    parser.add_argument("--diff_epochs", type=int, default=800)
    parser.add_argument("--diff_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--proto_alpha", type=float, default=None)
    parser.add_argument("--weight", type=float, default=None)

    args = parser.parse_args()

    # Fill omitted hyperparameters from the old configs, but keep CLI values as highest priority.
    cfg = DEFAULTS.get(args.dataset, DEFAULTS.get(Path(args.dataset).stem, {}))
    for key, value in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # General fallbacks when dataset has no old config.
    if args.ae_lr is None:
        args.ae_lr = 0.01
    if args.ae_dropout is None:
        args.ae_dropout = 0.3
    if args.ae_alpha is None:
        args.ae_alpha = 0.8
    if args.proto_alpha is None:
        args.proto_alpha = 0.1
    if args.weight is None:
        args.weight = 1.0

    return args


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) if args.device >= 0 else ""

    model = DiffGAD(
        hid_dim=args.hid_dim,
        diff_dim=args.diff_dim,
        ae_epochs=args.ae_epochs,
        diff_epochs=args.diff_epochs,
        patience=args.patience,
        lr=args.lr,
        wd=args.wd,
        weight=args.weight,
        sample_steps=args.sample_steps,
        ae_dropout=args.ae_dropout,
        ae_lr=args.ae_lr,
        ae_alpha=args.ae_alpha,
        proto_alpha=args.proto_alpha,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        save_ckpt=not args.no_save,
        num_trials=args.num_trials,
        device=args.device,
    )
    model(args.dataset)


if __name__ == "__main__":
    main()

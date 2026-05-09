import argparse
import os

from DiffGAD import DiffGAD


def get_arguments():
    parser = argparse.ArgumentParser(
        description="DiffGAD reproduction with CLI hyperparameters."
    )

    # data / runtime
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or .mat filename, e.g. book, Disney, Enron, Reddit, weibo, YelpChi.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/datasets/GAD/mat",
        help="Directory containing .mat datasets.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device id. Use -1 for CPU."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="best checkpoint(cache) will be saved here.",
    )

    # AE
    parser.add_argument("--ae_epochs", type=int, default=300)
    parser.add_argument("--ae_lr", type=float, default=0.01)
    parser.add_argument("--ae_dropout", type=float, default=0.3)
    parser.add_argument("--ae_alpha", type=float, default=0.8)
    parser.add_argument("--hid_dim", type=int, default=None)

    # Diffusion
    parser.add_argument("--diff_epochs", type=int, default=800)
    parser.add_argument("--diff_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--proto_alpha", type=float, default=0.1)
    parser.add_argument("--weight", type=float, default=1.0)

    args = parser.parse_args()

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
        num_trials=args.num_trials,
        device=args.device,
    )
    model(args.dataset)


if __name__ == "__main__":
    main()

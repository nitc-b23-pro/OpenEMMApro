"""
plot_losses.py
Standalone re-plot of training-vs-validation loss per epoch, from the two
JSON files train_openemma_tinyvla.py already writes
(lp_epoch_loss.json / val_epoch_losses.json). train_openemma_tinyvla.py
already produces this exact plot itself at the end of a run -- this script
exists so you can regenerate it WITHOUT re-running training, e.g. if the
Kaggle session ended before reaching the final plotting cell, or you just
want the image again for the report/slides without the training script's
side effects.

Usage
-----
    python plot_losses.py
    python plot_losses.py --train-losses lp_epoch_loss.json --val-losses val_epoch_losses.json --output ep_loss_graph.jpg
"""
import argparse
import json

import matplotlib.pyplot as plt


def load_losses(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_losses(train_losses, val_losses, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(list(train_losses.keys()), list(train_losses.values()), marker="o", linestyle="-", label="train")
    plt.plot(list(val_losses.keys()), list(val_losses.values()), marker="s", linestyle="--", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Loss per Epoch")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-losses", type=str, default="lp_epoch_loss.json")
    parser.add_argument("--val-losses", type=str, default="val_epoch_losses.json")
    parser.add_argument("--output", type=str, default="ep_loss_graph.jpg")
    args = parser.parse_args()

    train_losses = load_losses(args.train_losses)
    val_losses = load_losses(args.val_losses)
    plot_losses(train_losses, val_losses, args.output)


if __name__ == "__main__":
    main()

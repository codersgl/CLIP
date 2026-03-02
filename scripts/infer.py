import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from PIL import Image

from clip import InferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP inference for retrieval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text2image", "image2text"],
        required=True,
        help="Retrieval direction",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image paths for retrieval candidates",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help="Text inputs for retrieval candidates",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k results")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Hydra config directory path",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config file name",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show tqdm progress bars during encoding",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Render retrieval results with matplotlib",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="inference_figures",
        help="Directory for rendered figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display figures in window (if environment supports GUI)",
    )

    return parser.parse_args()


def _truncate_text(text: str, max_len: int = 42) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def render_text2image_figure(
    results: list[dict],
    top_k: int,
    figure_path: str,
    mode_title: str,
    show: bool,
):
    if len(results) == 0:
        return

    rows = len(results)
    cols = top_k
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(mode_title, fontsize=12)

    for row, item in enumerate(results):
        query_text = _truncate_text(item["query_text"], max_len=60)
        matches = item["matches"]
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(matches):
                continue

            match = matches[col]
            image_path = match["image_path"]
            score = match["score"]
            try:
                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))
                if col == 0:
                    ax.set_title(f"Q: {query_text}\nTop-{col + 1} | {score:.4f}")
                else:
                    ax.set_title(f"Top-{col + 1} | {score:.4f}")
            except Exception as err:
                ax.text(0.5, 0.5, f"load failed\n{err}", ha="center", va="center")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def render_image2text_figure(
    results: list[dict],
    top_k: int,
    figure_path: str,
    mode_title: str,
    show: bool,
):
    if len(results) == 0:
        return

    rows = len(results)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows), squeeze=False)
    fig.suptitle(mode_title, fontsize=12)

    for row, item in enumerate(results):
        ax = axes[row][0]
        ax.axis("off")
        image_path = item["query_image"]
        matches = item["matches"]
        lines = []
        for rank, match in enumerate(matches[:top_k], start=1):
            lines.append(
                f"Top-{rank}: {_truncate_text(match['text'], 80)} ({match['score']:.4f})"
            )
        text_block = "\n".join(lines) if lines else "No match"

        try:
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(text_block)
        except Exception as err:
            ax.text(
                0.5,
                0.5,
                f"Image load failed: {err}\n{text_block}",
                ha="center",
                va="center",
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def load_config(config_path: str, config_name: str):
    abs_config_path = str(Path(config_path).resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=abs_config_path):
        cfg = compose(config_name=config_name)
    return cfg


def main():
    args = parse_args()
    start_time = time.perf_counter()
    cfg = load_config(args.config_path, args.config_name)

    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    checkpoint_path = str(Path(args.checkpoint).resolve())
    image_paths = [str(Path(path).resolve()) for path in args.images]
    texts = args.texts

    engine = InferenceEngine.from_config(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )

    if args.mode == "text2image":
        payload = engine.retrieve_text_to_image(
            texts=texts,
            image_paths=image_paths,
            top_k=args.topk,
            batch_size=args.batch_size,
            include_errors=True,
            show_progress=args.show_progress,
        )
    else:
        payload = engine.retrieve_image_to_text(
            image_paths=image_paths,
            texts=texts,
            top_k=args.topk,
            batch_size=args.batch_size,
            include_errors=True,
            show_progress=args.show_progress,
        )

    payload_dict = cast(dict[str, Any], payload)
    results = payload_dict["results"]
    errors = payload_dict["errors"]
    stats = payload_dict["stats"]

    duration = time.perf_counter() - start_time
    print(
        (
            f"[inference] mode={args.mode} device={engine.device} topk={args.topk} "
            f"texts(valid/total)={stats['valid_text_count']}/{stats['input_text_count']} "
            f"images(valid/total)={stats['valid_image_count']}/{stats['input_image_count']} "
            f"errors={len(errors)} duration={duration:.2f}s"
        ),
        file=sys.stderr,
    )

    if errors:
        print("[inference] skipped inputs:", file=sys.stderr)
        for error in errors:
            print(
                (
                    f"  - type={error['type']} index={error['index']} "
                    f"input={error['input']} msg={error['message']}"
                ),
                file=sys.stderr,
            )

    if args.visualize:
        os.makedirs(args.figure_dir, exist_ok=True)
        figure_name = f"{args.mode}_top{args.topk}.png"
        figure_path = str(Path(args.figure_dir) / figure_name)
        title = (
            f"mode={args.mode} | topk={args.topk} | "
            f"valid_text={stats['valid_text_count']} valid_image={stats['valid_image_count']}"
        )
        if args.mode == "text2image":
            render_text2image_figure(
                results=results,
                top_k=args.topk,
                figure_path=figure_path,
                mode_title=title,
                show=args.show,
            )
        else:
            render_image2text_figure(
                results=results,
                top_k=args.topk,
                figure_path=figure_path,
                mode_title=title,
                show=args.show,
            )
        print(f"[inference] figure saved: {figure_path}", file=sys.stderr)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

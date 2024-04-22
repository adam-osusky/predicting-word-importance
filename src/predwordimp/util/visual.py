import os
from typing import Any

import matplotlib.pyplot as plt
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from highlight_text import HighlightText
from IPython.display import HTML, display
from matplotlib import cm

from predwordimp.eval.metrics import RankingEvaluator
from predwordimp.util.logger import get_logger

logger = get_logger(__name__)


def load_ds(ds_path: str, filename: str) -> Dataset:
    ds = load_dataset(
        "json",
        data_files=os.path.join(ds_path, filename),
        split="all",
    )
    logger.info(ds)
    return ds  # type: ignore


def view_html(ds: Dataset, idx: int) -> None:
    colored = []
    for word, target in zip(ds[idx]["words"], ds[idx]["target"]):
        color = "red" if target == 1 else "white"
        colored.append(f'<span style="color: {color};">{word}</span>')
    display(HTML(" ".join(colored)))


def view(ds, idx, name="test", line_width=80, figsize=(6, 4)) -> plt.Figure:
    inserted_hl = {
        "bbox": {
            "edgecolor": "orange",
            "facecolor": "yellow",
            "linewidth": 1.5,
            "pad": 1,
        }
    }
    highlight_textprops = []

    text = ""
    num_lines = 1
    for w, t in zip(ds[idx]["words"], ds[idx]["target"]):
        if t == 1:
            highlight_textprops.append(inserted_hl)
            w = "<" + w + ">"

        if len(text) + len(w) - 2 + 1 >= num_lines * line_width:
            text += "\n"
            num_lines += 1

        text += w + " "

    fig, ax = plt.subplots(figsize=figsize)
    # fig, ax = plt.subplots()
    ax.axis("off")

    HighlightText(
        x=0.5,
        y=0.5,
        fontsize=10,
        ha="center",
        va="center",
        s=text,
        highlight_textprops=highlight_textprops,
        ax=ax,
    )

    fig.savefig(f"figs/{name}.pdf", format="pdf")
    plt.show()

    return fig


def highlight_words(
    words: list[str], order: list[int], line_width=80
) -> tuple[str, list[Any]]:
    delims = ("{", "}")
    index_dict = {num: index for index, num in enumerate(order)}
    highlight_textprops = []

    text = ""
    num_lines = 1
    for i, w in enumerate(words):
        if i in index_dict:
            inserted_hl = {
                "bbox": {
                    "edgecolor": "black",
                    "facecolor": "yellow",
                    "linewidth": 1.5,
                    "pad": 1,
                }
            }

            color_value = 1 - index_dict[i] / len(order)
            color = cm.Blues(color_value)
            facecolor = tuple(color[:3])

            # inserted_hl = inserted_hl_base.copy()
            inserted_hl["bbox"]["facecolor"] = facecolor

            highlight_textprops.append(inserted_hl)
            w = delims[0] + w + delims[1]

        if len(text) + len(w) - 2 + 1 >= num_lines * line_width:
            text += "\n"
            num_lines += 1

        text += w + " "
    logger.info(len(order), len(highlight_textprops))
    return text, highlight_textprops


def view_predictions(pred: list[int], label: list[int], words: list[str]):
    delims = ("{", "}")
    pred_order = RankingEvaluator.get_selected_order(pred)
    label_order = RankingEvaluator.get_selected_order(label)

    pred_txt, pred_highlights = highlight_words(words, pred_order)
    label_txt, label_highlights = highlight_words(words, label_order)

    # fig, ax = plt.subplots(nrows=2, figsize=(6, 8))
    fig, ax = plt.subplots(ncols=2, figsize=(14, 2))

    ax[0].axis("off")
    ax[0].set_title("Prediction")
    HighlightText(
        x=0.5,
        y=0.5,
        fontsize=10,
        ha="center",
        va="center",
        s=pred_txt,
        highlight_textprops=pred_highlights,
        ax=ax[0],
        delim=delims,
    )

    ax[1].axis("off")
    ax[1].set_title("Label")
    HighlightText(
        x=0.5,
        y=0.5,
        fontsize=10,
        ha="center",
        va="center",
        s=label_txt,
        highlight_textprops=label_highlights,
        ax=ax[1],
        delim=delims,
    )

    plt.show()

    logger.info(pred_order)
    logger.info(label_order)
    logger.info(words[:10])

    # fig.savefig(f"figs/{name}.pdf", format="pdf")
    # plt.show()

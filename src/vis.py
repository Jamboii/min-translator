from typing import Sequence
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(
    sentence: Sequence[str],
    translation: Sequence[str],
    attention: torch.tensor,
    title: str = "",
    cmap: str = "bone",
) -> None:
    """
    :param sentence: List of source string tokens
    :param translation: List of the model's translation tokens
    :param attention: (1, tgt_len, src_len) torch tensor of attention weights
    """
    # Throw an error if the attention weights shape does not match
    # the target length and source length
    if attention.size(-2) != len(translation) or attention.size(-1) != len(sentence):
        raise ValueError(
            "The dimensions of the attention tensor are not compatible with "
            "the sentence and translation. "
            f"Got {attention.size(-2)} != {len(translation)} "
            f"or {attention.size(-1)} != {len(sentence)}"
        )

    # Determine the number of rows and columns to use for the plot
    while attention.dim() < 4:
        attention = attention.unsqueeze(0)  # (1, 1, tgt_len, src_len)
    nrows, ncols = attention.size(0), attention.size(1)

    # Set up the figure and subplots
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5, nrows * 5),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    if title:
        fig.suptitle(title, size=20)

    # Plot the attention weights
    for i, (row_ax, att_row) in enumerate(zip(axes, attention)):
        for j, (ax, att) in enumerate(zip(row_ax, att_row)):
            pcm = ax.matshow(att.cpu().detach().numpy(), cmap=cmap)
            ax.set_xticks(
                ticks=np.arange(len(sentence)),
                labels=sentence,
                rotation=90,
                size=15,
            )
            # translation = translation[1:]
            ax.set_yticks(
                ticks=np.arange(len(translation)), labels=translation, size=15
            )
            ax.set_title(f"Block {i+1}, Head {j+1}", size=20)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    # Show the plot
    plt.show()
    plt.close()

"""Utility to pretty-print a Q-table."""

import torch


def print_q_table(q_table: torch.Tensor) -> None:
    """
    Pretty-print a Q-table.

    Args:
        q_table (torch.Tensor or np.ndarray): Q-table of shape [num_states, num_actions].
    """
    print("\nQ-table:")
    print("State |   Left   Down   Right   Up")
    print("-------------------------------------")
    for i, row in enumerate(q_table):
        formatted_row = " ".join(f"{val:.2f}" for val in row.tolist())
        print(f"{i:>5} | {formatted_row}")

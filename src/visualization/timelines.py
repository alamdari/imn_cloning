import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ACTIVITY_COLORS = {
    'home': 'skyblue', 'work': 'orange', 'eat': 'green', 'utility': 'purple', 'transit': 'red', 'unknown': 'gray',
    'school': 'yellow', 'shop': 'pink', 'leisure': 'lightgreen', 'health': 'lightcoral', 'admin': 'lightblue', 'finance': 'gold'
}


def _plot_stays(stays: List[Tuple[str, int, int]], y_offset: float, ax, anchor: Tuple = None):
    for act, st, et in stays:
        is_anchor = anchor and (act, st, et) == anchor
        ax.barh(
            y_offset,
            et - st,
            left=st,
            height=0.25,
            color=ACTIVITY_COLORS.get(act, 'black'),
            edgecolor='black' if is_anchor else None,
            linewidth=2 if is_anchor else 0.5,
            alpha=0.9
        )


def _visualize_day_data(day_data: Dict, user_id: int = 0) -> plt.Figure:
    fig, axes = plt.subplots(len(day_data), 1, figsize=(16, 3 * len(day_data)), sharex=True)
    if len(day_data) == 1:
        axes = [axes]

    for ax, (day_date, content) in zip(axes, sorted(day_data.items())):
        original = content["original"]
        synthetics = content["synthetic"]
        anchor = content.get("anchor")

        y_offset = 0.5
        _plot_stays(original, y_offset, ax, anchor=anchor)
        labels = ["Original"]

        for r, stays in sorted(synthetics.items()):
            y_offset += 0.4
            _plot_stays(stays, y_offset, ax)
            labels.append(f"Rand={r}")

        ax.set_yticks([0.5 + i * 0.4 for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 24 * 3600)
        ax.set_xticks([i * 3600 for i in range(0, 25, 2)])
        ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 2)], rotation=0)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)
        ax.set_title(f"User {user_id} - {day_date}", fontsize=12)

    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_handles, title="Activity Types", loc="upper right", fontsize=9, title_fontsize=10)
    fig.suptitle(f"User {user_id} - Original and Synthetic Timelines", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.85, 0.98])
    return fig


def save_user_timelines(day_data: Dict, user_id: int, paths) -> None:
    fig = _visualize_day_data(day_data, user_id=user_id)
    timeline_path = os.path.join(paths.vis_dir(), f"user_{user_id}_timelines.png")
    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    fig.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Timeline visualization saved: {os.path.basename(timeline_path)}")



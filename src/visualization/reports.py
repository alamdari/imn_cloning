import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def user_probs_report(duration_probs, transition_probs, trip_duration_probs, user_id: int, out_folder: str) -> None:
    os.makedirs(out_folder, exist_ok=True)

    expanded = {}
    for act, (hist, bins) in duration_probs.items():
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        expanded[act] = samples

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if expanded:
        sns.boxplot(data=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in expanded.items()])), ax=axes[0])
        axes[0].set_title("Stay Durations per Activity")
        axes[0].set_ylabel("Duration (s)")
        axes[0].tick_params(axis='x', rotation=45)

    df = pd.DataFrame(transition_probs).fillna(0)
    sns.heatmap(df, annot=True, cmap="Blues", cbar_kws={'label': 'Probability'}, ax=axes[1])
    axes[1].set_title("Transition Probability Matrix")
    axes[1].set_xlabel("From Activity")
    axes[1].set_ylabel("To Activity")

    if trip_duration_probs is not None:
        hist, bins = trip_duration_probs
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        if len(samples) > 1:
            sns.kdeplot(samples, fill=True, ax=axes[2])
        else:
            axes[2].hist(samples, bins=10, edgecolor="k")
        axes[2].set_title("Trip Duration Distribution")
        axes[2].set_xlabel("Duration (s)")
        axes[2].set_ylabel("Density")

    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"user_probs_report_{user_id}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    json_report = {
        "duration_probs": {k: {"hist": hist.tolist(), "bins": bins.tolist()} for k, (hist, bins) in duration_probs.items()},
        "transition_probs": {k: {kk: float(vv) for kk, vv in d.items()} for k, d in transition_probs.items()},
        "trip_duration_probs": {"hist": trip_duration_probs[0].tolist(), "bins": trip_duration_probs[1].tolist()} if trip_duration_probs is not None else None
    }
    json_path = os.path.join(out_folder, f"user_probs_report_{user_id}.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"  ✓ Probability report saved: {os.path.basename(fig_path)}, {os.path.basename(json_path)}")



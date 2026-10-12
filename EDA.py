#%%
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import json
#%%
# ---------- Plot style ----------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E3E3E3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 12,
    "axes.labelcolor": "#333333",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "font.size": 11,
    "legend.frameon": False,
})

PALETTE = "#2E7D5B"          # main bar color
ACCENT = "#E4572E"           # color for duplicates

thousands = FuncFormatter(lambda x, _: f"{int(x):,}")


def add_bar_labels(ax, fontsize=9):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h):,}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom",
                fontsize=fontsize, color="#333333",
                xytext=(0, 3), textcoords="offset points",
            )
#%%
OUT_DIR = Path('EDA')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("./Dataset")
FORMATS = {".jpg", ".jpeg", ".png"}
#%%
files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in FORMATS]
assert files, f"No images found under {DATA_DIR.resolve()}"
#%%
def inspect(path):
    try:
        with Image.open(path) as img:
            md5 = hashlib.md5(img.tobytes()).hexdigest()
            size = img.size
            return size, md5
    except Exception as e:
        return None , None
#%%
df = pd.DataFrame({'path': files})
df['label'] = df['path'].apply(lambda x: Path(x).parent.name)
df['crop'] = df['path'].map(lambda x: Path(x).parent.name.split("_")[0])
df['disease'] = df['path'].map(lambda x: Path(x).parent.name.rsplit("__")[-1])
info = df['path'].map(inspect)
df['size'] = info.map(lambda x: x[0])
df['md5'] = info.map(lambda x: x[1])
df['path'] = df['path'].map(str)
df.to_csv(OUT_DIR / "dataset_index.csv", index=False)



#%%
DATA_DIR = Path("./preProcess")
data_type = 'test'
OUT_DIR = Path(f"EDA_preProcessed/{data_type}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(f"preProcess/{data_type}.csv")
#%%
# ============================================================
# dataset_info.json
# ============================================================

def counts_to_dict(series):
    return {str(k): int(v) for k, v in series.items()}

corrupted = df[df["md5"].isna()]

dup_mask = df["md5"].notna() & df.duplicated("md5", keep=False)
dup_groups = df[dup_mask].groupby("md5")["path"].apply(list)

valid_sizes = df["size"].dropna()
widths = valid_sizes.map(lambda s: s[0])
heights = valid_sizes.map(lambda s: s[1])
size_counts = valid_sizes.map(lambda s: f"{s[0]}x{s[1]}").value_counts()

labels_sorted = sorted(df["label"].unique())

dataset_info = {
    "dataset_dir": str(DATA_DIR.resolve()),
    "formats": counts_to_dict(
        df["path"].map(lambda p: Path(p).suffix.lower()).value_counts()
    ),
    "num_images": int(len(df)),
    "num_labels": int(df["label"].nunique()),
    "num_crops": int(df["crop"].nunique()),
    "labels": labels_sorted,
    "images_per_label": counts_to_dict(df["label"].value_counts()),
    "images_per_crop": counts_to_dict(df["crop"].value_counts()),
    "diseases_per_crop": {
        crop: sorted(group["disease"].unique().tolist())
        for crop, group in df.groupby("crop")
    },
    "image_size": {
        "num_unique_sizes": int(size_counts.shape[0]),
        "size_counts": counts_to_dict(size_counts),
        "width": {"min": int(widths.min()), "max": int(widths.max()), "mean": round(float(widths.mean()), 2)},
        "height": {"min": int(heights.min()), "max": int(heights.max()), "mean": round(float(heights.mean()), 2)},
    },
    "corrupted_images": {
        "count": int(len(corrupted)),
        "path": corrupted["path"].tolist(),
    },
    "duplicates": {
        "num_groups": int(len(dup_groups)),
        "num_extra_images": int(sum(len(g) - 1 for g in dup_groups)),
        "groups": [paths for paths in dup_groups.tolist()],
    },
}

with open(OUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

#%%
# ============================================================
# 1. Number of images per crop
# ============================================================

crop_counts = df["crop"].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
crop_counts.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75)

ax.set_title("Number of Images per Crop")
ax.set_xlabel("Crop")
ax.set_ylabel("Number of Images")
ax.yaxis.set_major_formatter(thousands)
ax.grid(axis="x", visible=False)
ax.set_ylim(0, crop_counts.max() * 1.12)
add_bar_labels(ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(OUT_DIR / "01_images_per_crop.png", dpi=200, bbox_inches="tight")
plt.close()

#%%
# ============================================================
# 2. Disease distribution for each crop
# ============================================================

crop_disease = pd.crosstab(df["crop"], df["disease"])

n_diseases = crop_disease.shape[1]
colors = plt.cm.tab20(np.linspace(0, 1, 20))
colors = [colors[i % 20] for i in range(n_diseases)]

fig, ax = plt.subplots(figsize=(14, 8))
crop_disease.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=colors,
    edgecolor="white",
    linewidth=0.5,
    width=0.8,
)

ax.set_title("Disease Distribution per Crop")
ax.set_xlabel("Crop")
ax.set_ylabel("Number of Images")
ax.yaxis.set_major_formatter(thousands)
ax.grid(axis="x", visible=False)
plt.xticks(rotation=45, ha="right")
ax.legend(
    title="Disease",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=11,
    ncol=1 if n_diseases <= 15 else 2,
)
plt.tight_layout()

plt.savefig(OUT_DIR / "02_disease_distribution_per_crop.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# 3. Number of images per label
# ============================================================

disease_counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(14, 7))
disease_counts.plot(
    kind="bar",
    ax=ax,
    color=plt.cm.viridis(np.linspace(0.15, 0.85, len(disease_counts))),
    edgecolor="white",
    width=0.8,
)

ax.set_title("Number of Images per Label")
ax.set_xlabel("Label")
ax.set_ylabel("Number of Images")
ax.yaxis.set_major_formatter(thousands)
ax.grid(axis="x", visible=False)
ax.set_ylim(0, disease_counts.max() * 1.12)
add_bar_labels(ax, fontsize=8)
plt.xticks(rotation=60, ha="right")
plt.tight_layout()

plt.savefig(OUT_DIR / "03_images_per_disease.png", dpi=200, bbox_inches="tight")
plt.close()
#%%
# ============================================================
# 4. Duplicate images
# ============================================================

md5_counts = df["md5"].value_counts()

duplicate_groups = md5_counts[md5_counts > 1]

fig, ax = plt.subplots(figsize=(10, 6))

if len(duplicate_groups) > 0:
    ax.bar(
        range(len(duplicate_groups)),
        duplicate_groups.values,
        color=ACCENT,
        edgecolor="white",
    )

    ax.set_title("Duplicate Image Groups")
    ax.set_xlabel("Duplicate Group")
    ax.set_ylabel("Number of Images")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, duplicate_groups.max() * 1.12)
    add_bar_labels(ax)
else:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(
        0.5,
        0.5,
        "✔  No duplicate images found",
        ha="center",
        va="center",
        fontsize=16,
        color=PALETTE,
        fontweight="bold",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.8", fc="#EAF5EF", ec=PALETTE),
    )
    ax.set_title("Duplicate Image Groups")

plt.tight_layout()

plt.savefig(OUT_DIR / "04_duplicate_images.png", dpi=200, bbox_inches="tight")
plt.close()

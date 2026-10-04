from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "data" / "analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Spotify audio-feature trends, 2010–2019

This notebook rebuilds the portfolio analysis from the original 1,159,764-row dataset. It answers a narrow question: **how do the audio features of tracks in this dataset vary by release year from 2010 through 2019?**

The analysis is descriptive. The dataset is not a random sample of all music or Spotify listening, so confidence intervals quantify precision inside this dataset and do not turn the results into population-level or causal claims. In particular, Spotify's `valence` is an audio-feature score for perceived musical positiveness; it is **not** a sentiment score for lyrics and does not measure listeners' happiness.

Data source: [Spotify 1 Million Tracks on Kaggle](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks).
"""
    ),
    code(
        """
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import display

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")

START_YEAR = 2010
END_YEAR = 2019
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/amitanshjoshi/spotify-1million-tracks"


def find_project_root() -> Path:
    # Find the repository root whether the notebook starts in / or /data.
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

configured_path = os.getenv("SPOTIFY_DATA_PATH")
legacy_data_path = PROJECT_ROOT / "data" / "spotify_data.csv"
if configured_path:
    DATA_PATH = Path(configured_path).expanduser()
elif legacy_data_path.exists():
    DATA_PATH = legacy_data_path
else:
    DATA_PATH = RAW_DIR / "spotify_data.csv"


def download_dataset(destination: Path) -> None:
    # Download and extract the public Kaggle snapshot when the CSV is absent.
    archive_path = destination.with_suffix(".zip")
    print(f"Downloading the dataset to {archive_path} ...")
    urllib.request.urlretrieve(DATASET_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("spotify_data.csv"))
        with archive.open(member) as source, destination.open("wb") as target:
            target.write(source.read())
    archive_path.unlink()


if not DATA_PATH.exists():
    download_dataset(DATA_PATH)

try:
    display_path = DATA_PATH.resolve().relative_to(PROJECT_ROOT.resolve())
except ValueError:
    display_path = DATA_PATH
print(f"Dataset: {display_path}")
"""
    ),
    md(
        """
## 1. Load and validate the source

The expected schema is declared before analysis. This makes a changed or incorrect input fail clearly instead of silently producing different results.
"""
    ),
    code(
        """
EXPECTED_COLUMNS = {
    "artist_name", "track_name", "track_id", "popularity", "year", "genre",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature",
}

raw = pd.read_csv(DATA_PATH)
missing_columns = EXPECTED_COLUMNS.difference(raw.columns)
if missing_columns:
    raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

print(f"Rows: {len(raw):,}")
print(f"Columns: {raw.shape[1]}")
print(f"Years: {raw['year'].min()}–{raw['year'].max()}")
display(raw.head())
"""
    ),
    md(
        """
## 2. Data-quality audit

Three checks are kept separate:

1. missing identifiers;
2. duplicate records and duplicate track keys;
3. domain-range violations and statistically unusual tails.

IQR flags are diagnostic, not automatic deletion rules. Several Spotify features are bounded or intentionally skewed, and unusual values can be valid recordings. The primary trend analysis does not use duration, tempo, or popularity, so extreme values in those fields cannot drive its result.
"""
    ),
    code(
        """
missing_summary = (
    raw.isna().sum()
    .rename("missing_rows")
    .to_frame()
    .assign(missing_pct=lambda x: 100 * x["missing_rows"] / len(raw))
)

duplicate_summary = pd.DataFrame(
    {
        "check": ["exact row", "track_id", "artist_name + track_name"],
        "duplicate_rows": [
            raw.duplicated().sum(),
            raw.duplicated(subset=["track_id"]).sum(),
            raw.duplicated(subset=["artist_name", "track_name"]).sum(),
        ],
    }
)

display(missing_summary.query("missing_rows > 0"))
display(duplicate_summary)
"""
    ),
    code(
        """
range_rules = {
    "popularity": (0, 100),
    "danceability": (0, 1),
    "energy": (0, 1),
    "key": (0, 11),
    "mode": (0, 1),
    "speechiness": (0, 1),
    "acousticness": (0, 1),
    "instrumentalness": (0, 1),
    "liveness": (0, 1),
    "valence": (0, 1),
    "tempo": (0, 250),
    "duration_ms": (1, np.inf),
    "time_signature": (0, 5),
}

range_rows = []
for column, (lower, upper) in range_rules.items():
    invalid = raw[column].isna() | ~raw[column].between(lower, upper, inclusive="both")
    range_rows.append(
        {
            "column": column,
            "expected_range": f"[{lower}, {upper}]",
            "invalid_rows": int(invalid.sum()),
        }
    )
range_summary = pd.DataFrame(range_rows)
display(range_summary)
"""
    ),
    code(
        """
numeric_columns = [
    "popularity", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

outlier_rows = []
for column in numeric_columns:
    q1, q3 = raw[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = ~raw[column].between(lower, upper, inclusive="both")
    outlier_rows.append(
        {
            "column": column,
            "q1": q1,
            "q3": q3,
            "iqr_lower": lower,
            "iqr_upper": upper,
            "iqr_flagged_rows": int(flagged.sum()),
            "iqr_flagged_pct": 100 * flagged.mean(),
        }
    )
outlier_summary = pd.DataFrame(outlier_rows)
display(outlier_summary)
"""
    ),
    md(
        """
**Cleaning decision.** There are no duplicate track keys and no invalid values under the documented feature ranges, so no rows are removed as duplicates or numeric “outliers.” Missing artist/title labels remain visible in the quality audit. Very short (<30 seconds) and very long (>40 minutes) recordings are flagged for inspection rather than silently discarded.
"""
    ),
    code(
        """
analysis = raw.loc[raw["year"].between(START_YEAR, END_YEAR)].copy()
analysis["duration_min"] = analysis["duration_ms"] / 60_000
analysis["duration_flag"] = np.select(
    [analysis["duration_ms"] < 30_000, analysis["duration_ms"] > 2_400_000],
    ["under_30_seconds", "over_40_minutes"],
    default="typical_range",
)

period_quality = pd.Series(
    {
        "rows_2010_2019": len(analysis),
        "missing_artist_name": analysis["artist_name"].isna().sum(),
        "missing_track_name": analysis["track_name"].isna().sum(),
        "under_30_seconds": (analysis["duration_flag"] == "under_30_seconds").sum(),
        "over_40_minutes": (analysis["duration_flag"] == "over_40_minutes").sum(),
        "tempo_zero": (analysis["tempo"] == 0).sum(),
        "popularity_zero": (analysis["popularity"] == 0).sum(),
    },
    name="rows",
).to_frame()
display(period_quality)
"""
    ),
    md(
        """
## 3. Yearly trends and uncertainty

For each year, the notebook reports the mean and a 95% normal-approximation interval. With tens of thousands of rows per year these intervals are narrow, but the dataset's selection process remains the main limitation.

Endpoint differences compare 2019 with 2010. The regression slope summarizes the direction across all ten years; `R²` is included to keep statistical significance in perspective. A tiny p-value with a tiny `R²` is evidence of a precisely estimated but weak year-to-year association, not a large effect.
"""
    ),
    code(
        """
TREND_FEATURES = ["valence", "energy", "danceability", "popularity"]


def yearly_summary(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frames = []
    for feature in features:
        part = data.groupby("year")[feature].agg(["count", "mean", "std"]).reset_index()
        part["feature"] = feature
        part["se"] = part["std"] / np.sqrt(part["count"])
        part["ci_low"] = part["mean"] - 1.96 * part["se"]
        part["ci_high"] = part["mean"] + 1.96 * part["se"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)[
        ["year", "feature", "count", "mean", "std", "se", "ci_low", "ci_high"]
    ]


yearly_trends = yearly_summary(analysis, TREND_FEATURES)
display(yearly_trends.query("feature != 'popularity'").pivot(index="year", columns="feature", values="mean"))
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
labels = {
    "valence": "Valence (musical positiveness)",
    "energy": "Energy",
    "danceability": "Danceability",
}
colors = {"valence": "#8c6bb1", "energy": "#ef8a62", "danceability": "#4daf4a"}

for ax, feature in zip(axes, labels):
    plot_data = yearly_trends.query("feature == @feature")
    x = plot_data["year"].to_numpy()
    mean = plot_data["mean"].to_numpy()
    low = plot_data["ci_low"].to_numpy()
    high = plot_data["ci_high"].to_numpy()
    ax.plot(x, mean, marker="o", linewidth=2.2, color=colors[feature])
    ax.fill_between(x, low, high, alpha=0.18, color=colors[feature], label="95% CI")
    ax.set_title(labels[feature])
    ax.set_xlabel("Release year")
    ax.set_ylabel("Mean score")
    ax.set_xticks(range(START_YEAR, END_YEAR + 1, 2))
    ax.legend(loc="best")

fig.suptitle("Audio-feature means by release year (all 82 genres)", y=1.03, fontsize=15)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
def endpoint_and_trend(data: pd.DataFrame, feature: str) -> dict:
    start = data.loc[data["year"] == START_YEAR, feature]
    end = data.loc[data["year"] == END_YEAR, feature]
    difference = end.mean() - start.mean()
    difference_se = np.sqrt(start.var(ddof=1) / len(start) + end.var(ddof=1) / len(end))
    regression = stats.linregress(data["year"], data[feature])
    return {
        "feature": feature,
        "mean_2010": start.mean(),
        "mean_2019": end.mean(),
        "absolute_change": difference,
        "relative_change_pct": 100 * difference / start.mean(),
        "change_ci_low": difference - 1.96 * difference_se,
        "change_ci_high": difference + 1.96 * difference_se,
        "slope_per_year": regression.slope,
        "slope_ci_low": regression.slope - 1.96 * regression.stderr,
        "slope_ci_high": regression.slope + 1.96 * regression.stderr,
        "p_value": regression.pvalue,
        "r_squared": regression.rvalue**2,
    }


endpoint_changes = pd.DataFrame(endpoint_and_trend(analysis, feature) for feature in TREND_FEATURES)
display(endpoint_changes)
"""
    ),
    md(
        """
## 4. Sensitivity to changing genre composition

Raw yearly means can change because the mix of genres changes. As a sensitivity check, every genre represented in all ten years receives equal weight within each year. This balanced panel is not the one “correct” weighting; it asks whether the direction persists when annual genre counts cannot dominate the result.
"""
    ),
    code(
        """
equal_weight_rows = []
for feature in ["valence", "energy", "danceability"]:
    genre_year = analysis.groupby(["year", "genre"])[feature].mean().unstack("genre")
    balanced_genres = genre_year.columns[genre_year.notna().all(axis=0)]
    equal_weighted = genre_year[balanced_genres].mean(axis=1)
    regression = stats.linregress(equal_weighted.index, equal_weighted.values)
    equal_weight_rows.append(
        {
            "feature": feature,
            "balanced_genres": len(balanced_genres),
            "equal_weight_mean_2010": equal_weighted.loc[START_YEAR],
            "equal_weight_mean_2019": equal_weighted.loc[END_YEAR],
            "absolute_change": equal_weighted.loc[END_YEAR] - equal_weighted.loc[START_YEAR],
            "slope_per_year": regression.slope,
            "r_squared_across_10_year_means": regression.rvalue**2,
        }
    )

genre_mix_sensitivity = pd.DataFrame(equal_weight_rows)
display(genre_mix_sensitivity)
"""
    ),
    md(
        """
## 5. Genre comparison

The main comparison uses the 15 most represented genres in 2010–2019. Ranking is by row count, not by popularity. Means summarize the dataset and should not be read as immutable properties of a genre.
"""
    ),
    code(
        """
genre_summary = (
    analysis.groupby("genre")
    .agg(
        tracks=("track_id", "size"),
        mean_valence=("valence", "mean"),
        mean_energy=("energy", "mean"),
        mean_danceability=("danceability", "mean"),
        mean_popularity=("popularity", "mean"),
    )
    .sort_values("tracks", ascending=False)
    .reset_index()
)
top_genres = genre_summary.head(15)["genre"].tolist()
display(genre_summary.head(15))
"""
    ),
    code(
        """
genre_plot = genre_summary.head(15).set_index("genre")[[
    "mean_valence", "mean_energy", "mean_danceability"
]]

plt.figure(figsize=(9, 7))
sns.heatmap(genre_plot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
plt.title("Mean audio features for the 15 most represented genres")
plt.xlabel("Feature")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
genre_year_trends = (
    analysis.groupby(["year", "genre"])
    .agg(
        tracks=("track_id", "size"),
        mean_valence=("valence", "mean"),
        mean_energy=("energy", "mean"),
        mean_danceability=("danceability", "mean"),
        mean_popularity=("popularity", "mean"),
    )
    .reset_index()
)

genre_valence_changes = []
for genre in top_genres:
    annual = genre_year_trends.query("genre == @genre").sort_values("year")
    regression = stats.linregress(annual["year"], annual["mean_valence"])
    genre_valence_changes.append(
        {
            "genre": genre,
            "tracks": int(genre_summary.set_index("genre").loc[genre, "tracks"]),
            "mean_valence_2010": annual.loc[annual["year"] == START_YEAR, "mean_valence"].iloc[0],
            "mean_valence_2019": annual.loc[annual["year"] == END_YEAR, "mean_valence"].iloc[0],
            "absolute_change": (
                annual.loc[annual["year"] == END_YEAR, "mean_valence"].iloc[0]
                - annual.loc[annual["year"] == START_YEAR, "mean_valence"].iloc[0]
            ),
            "slope_per_year": regression.slope,
            "p_value_across_10_annual_means": regression.pvalue,
        }
    )

genre_valence_changes = pd.DataFrame(genre_valence_changes).sort_values("absolute_change")
display(genre_valence_changes)
"""
    ),
    md(
        """
## 6. Relationships with popularity

Pearson correlations are reported as descriptive associations only. Popularity is a time-sensitive Spotify metric, and this file is a snapshot rather than a historical record of plays. Correlation does not identify a cause of popularity.
"""
    ),
    code(
        """
correlation_columns = ["valence", "energy", "danceability", "popularity"]
correlations = analysis[correlation_columns].corr(method="pearson")
display(correlations)

plt.figure(figsize=(6.5, 5.5))
sns.heatmap(correlations, annot=True, fmt=".3f", cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("Pearson correlations (descriptive, not causal)")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 7. Export compact reproducible results

The repository keeps compact aggregate tables so the reported figures can be checked without committing a second track-level dataset. The raw CSV and any row-level export are intentionally excluded from Git because the notebook can recreate them.
"""
    ),
    code(
        """
quality_records = []
for column, row in missing_summary.iterrows():
    quality_records.append(
        {"check_type": "missing_values", "metric": column, "value": row["missing_rows"],
         "detail": f"{row['missing_pct']:.6f}% of source rows"}
    )
for row in duplicate_summary.itertuples(index=False):
    quality_records.append(
        {"check_type": "duplicates", "metric": row.check, "value": row.duplicate_rows,
         "detail": "duplicate rows"}
    )
for row in range_summary.itertuples(index=False):
    quality_records.append(
        {"check_type": "range_validation", "metric": row.column, "value": row.invalid_rows,
         "detail": f"outside {row.expected_range}"}
    )
for row in outlier_summary.itertuples(index=False):
    quality_records.append(
        {"check_type": "iqr_flag", "metric": row.column, "value": row.iqr_flagged_rows,
         "detail": f"{row.iqr_flagged_pct:.6f}% flagged; not automatically removed"}
    )
for metric, value in period_quality["rows"].items():
    quality_records.append(
        {"check_type": "period_2010_2019", "metric": metric, "value": value,
         "detail": "row count"}
    )
quality_export = pd.DataFrame(quality_records)

yearly_trends.to_csv(PROCESSED_DIR / "yearly_trends.csv", index=False)
endpoint_changes.to_csv(PROCESSED_DIR / "endpoint_changes.csv", index=False)
genre_summary.to_csv(PROCESSED_DIR / "genre_summary.csv", index=False)
genre_year_trends.to_csv(PROCESSED_DIR / "genre_year_trends.csv", index=False)
genre_mix_sensitivity.to_csv(PROCESSED_DIR / "genre_mix_sensitivity.csv", index=False)
genre_valence_changes.to_csv(PROCESSED_DIR / "genre_valence_changes_top15.csv", index=False)
quality_export.to_csv(PROCESSED_DIR / "data_quality_summary.csv", index=False)

print("Exported 7 compact summary files to data/processed")
"""
    ),
    md(
        """
## 8. Conclusions

- Mean valence moves from **0.4587 in 2010 to 0.4267 in 2019**, an absolute change of **−0.0320** or **−7.0%**. The 95% interval for the endpoint difference is **[−0.0353, −0.0287]**. Calendar year alone explains only **0.27%** of track-level valence variance (`R² = 0.0027`), so this is a modest aggregate shift, not a rule about individual songs.
- The direction remains negative in a balanced panel that gives equal weight to each of the 49 genres represented in all ten years: **−0.0451** from 2010 to 2019. Changing genre proportions therefore do not explain away the observed direction in this sensitivity check.
- Mean danceability increases from **0.5315 to 0.5517** (**+3.8%**). Mean energy changes from **0.6547 to 0.6478** (**−1.1%**), which is small in practical terms.
- Genre patterns vary substantially. Among the 15 largest genres, several decline in annual mean valence while others are flat or slightly higher. The overall average should not be presented as universal across genres.
- The analysis supports the statement that tracks in this dataset show a modest decline in **audio valence** over the decade. It does **not** establish that music lyrics became sadder, that listeners became less happy, or that release year caused the change.

### Limitations

- The dataset is a convenience snapshot, not a documented probability sample of all released music or all Spotify listening.
- `year` is release year; the file does not contain historical streaming counts by year.
- Spotify popularity is time-sensitive and should not be interpreted as popularity measured at release.
- Genre labels and audio features are supplied by the source dataset; label accuracy is not independently audited here.
- Confidence intervals describe sampling-style precision under independence assumptions, but unknown dataset construction and clustered artists can add uncertainty not captured by those intervals.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")

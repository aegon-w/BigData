"""Explore the cleaned complaints data with pandas + matplotlib + seaborn.

Run:  .venv/bin/python analysis.py        (run clean.py first)
Plots are saved in the plots/ folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files; remove this line to use plt.show()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

# ---------- style ----------
BLUE = "#2a78d6"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": TEXT, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})
millions = mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")


def barh(series, title, xlabel, fname, fmt=millions, label_fmt="{:,.0f}"):
    """Horizontal bar chart, biggest on top, value label at the end of each bar."""
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(s) + 1.2))
    ax.barh(s.index.astype(str), s.values, color=BLUE, height=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        ax.text(v, y, "  " + label_fmt.format(v), va="center", fontsize=9, color=TEXT2)
    ax.set_xlim(0, s.max() * 1.15)
    fig.savefig(OUT / fname)
    plt.close(fig)
    print("saved", OUT / fname)


# ---------- load ----------
CATS = ["product", "sub_product", "issue", "sub_issue", "company", "state",
        "tags", "channel", "response"]
df = pd.read_csv(
    "complaints_clean.csv",
    parse_dates=["date_received", "date_sent"],
    dtype={c: "category" for c in CATS}
    | {"days_to_company": "Int16", "year": "int16", "complaint_id": "Int64", "timely": bool},
)
print(df.shape)
print(df.dtypes, "\n")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
print(df.describe(include="all").T[["count", "unique", "top"]], "\n")

# Drop the current (incomplete) month so the trend line doesn't fall off at the end
last_full_month = df["date_received"].max().to_period("M") - 1
df = df[df["date_received"].dt.to_period("M") <= last_full_month]

# ---------- 1. complaints per month ----------
monthly = df.set_index("date_received").resample("MS").size()
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(monthly.index, monthly.values, color=BLUE, linewidth=2)
ax.set_title("Complaints received per month")
ax.set_ylabel("Complaints")
ax.yaxis.set_major_formatter(millions)
ax.grid(axis="x", visible=False)
peak = monthly.idxmax()
ax.annotate(f"Peak: {monthly.max():,}\n{peak:%b %Y}", (peak, monthly.max()),
            xytext=(-110, -10), textcoords="offset points", fontsize=9, color=TEXT2,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
fig.savefig(OUT / "1_complaints_per_month.png")
plt.close(fig)
print("saved", OUT / "1_complaints_per_month.png")

# ---------- 2. products ----------
barh(df["product"].value_counts(), "Complaints by product",
     "Complaints", "2_products.png")

# ---------- 3. top companies ----------
barh(df["company"].value_counts().head(10), "Top 10 companies by complaints",
     "Complaints", "3_top_companies.png")

# Without the 3 credit bureaus (they dominate everything else)
bureaus = ["EQUIFAX, INC.", "EXPERIAN INFORMATION SOLUTIONS INC.",
           "TRANSUNION INTERMEDIATE HOLDINGS, INC."]
barh(df.loc[~df["company"].isin(bureaus), "company"].value_counts().head(10),
     "Top 10 companies, excluding the 3 credit bureaus",
     "Complaints", "3b_top_companies_no_bureaus.png")

# ---------- 4. states ----------
barh(df["state"].value_counts().head(15), "Top 15 states by complaints",
     "Complaints", "4_states.png")

# ---------- 5. how companies responded ----------
barh(df["response"].value_counts(), "How companies responded",
     "Complaints", "5_company_response.png")

# ---------- 6. heatmap: response mix per product ----------
mix = pd.crosstab(df["product"], df["response"], normalize="index") * 100
main_responses = ["Closed with explanation", "Closed with non-monetary relief",
                  "Closed with monetary relief", "In progress", "Untimely response"]
mix = mix[main_responses].sort_values("Closed with monetary relief", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(mix, annot=True, fmt=".1f", cmap=BLUES, linewidths=2, linecolor="#fcfcfb",
            cbar_kws={"label": "% of the product's complaints"}, ax=ax)
ax.set_title("Response mix per product (%)")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
fig.savefig(OUT / "6_response_heatmap.png")
plt.close(fig)
print("saved", OUT / "6_response_heatmap.png")

# ---------- 7. timely response rate per product ----------
timely = df.groupby("product", observed=True)["timely"].mean() * 100
barh(timely, "Timely response rate by product", "% answered on time",
     "7_timely_rate.png", fmt=mtick.PercentFormatter(decimals=0), label_fmt="{:.1f}%")

# ---------- 8. submission channel ----------
barh(df["channel"].value_counts(), "How complaints were submitted",
     "Complaints", "8_channels.png")

# ---------- 9. complaints per year by product (seaborn, top 4 products) ----------
top4 = df["product"].value_counts().index[:4]
yearly = (df[df["product"].isin(top4)]
          .groupby(["year", "product"], observed=True).size()
          .reset_index(name="complaints"))
yearly["product"] = yearly["product"].astype(str)
palette = dict(zip(top4, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
g = sns.relplot(data=yearly, x="year", y="complaints", col="product", col_wrap=2,
                kind="line", marker="o", linewidth=2, markersize=6,
                hue="product", palette=palette, legend=False,
                facet_kws={"sharey": False}, height=3.2, aspect=1.6)
g.set_titles("{col_name}")
g.set_axis_labels("", "Complaints")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(millions)
g.figure.suptitle("Complaints per year, top 4 products", x=0.02, ha="left",
                  fontweight="bold", y=1.03)
g.savefig(OUT / "9_yearly_by_product.png")
plt.close(g.figure)
print("saved", OUT / "9_yearly_by_product.png")

# ---------- some quick numbers ----------
print("\nShare of complaints about credit reporting:",
      f"{(df['product'] == 'Credit reporting').mean():.1%}")
print("Share going to the 3 bureaus:", f"{df['company'].isin(bureaus).mean():.1%}")
print("Overall timely response rate:", f"{df['timely'].mean():.1%}")
print("Median days to forward to company:", df["days_to_company"].median())

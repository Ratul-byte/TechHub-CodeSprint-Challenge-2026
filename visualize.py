import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DB_PATH = Path("data/arxiv.db")
PLOTS_DIR = Path("data/plots")
CLEAN_SQL = Path("clean.sql")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate plots from the cleaned arXiv database.")
	parser.add_argument("--db", default=str(DB_PATH), help="Path to the SQLite database.")
	parser.add_argument(
		"--rebuild-clean",
		action="store_true",
		help="Run clean.sql before plotting if the cleaned tables are missing.",
	)
	return parser


def _ensure_clean_tables(conn: sqlite3.Connection, rebuild_clean: bool) -> None:
	cur = conn.cursor()
	tables = {
		row[0]
		for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
	}
	needed = {"papers", "category_stats", "yearly_trends", "publication_status", "author_stats"}
	if needed.issubset(tables):
		return
	if not rebuild_clean:
		raise RuntimeError("Cleaned tables are missing. Run clean.sql first or use --rebuild-clean.")
	if not CLEAN_SQL.exists():
		raise FileNotFoundError("clean.sql was not found in the workspace.")
	conn.executescript(CLEAN_SQL.read_text(encoding="utf-8"))
	conn.commit()


def _setup_style() -> None:
	plt.rcParams.update(
		{
			"figure.figsize": (12, 7),
			"figure.dpi": 120,
			"savefig.dpi": 120,
			"font.size": 11,
			"axes.titlesize": 16,
			"axes.labelsize": 12,
			"legend.fontsize": 10,
			"axes.grid": True,
			"grid.alpha": 0.25,
			"axes.spines.top": False,
			"axes.spines.right": False,
			"axes.facecolor": "#fbfbfb",
			"figure.facecolor": "white",
		}
	)


def _save(fig: plt.Figure, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(path, format="png", bbox_inches="tight")
	plt.close(fig)


def _load_papers(conn: sqlite3.Connection) -> pd.DataFrame:
	return pd.read_sql_query("SELECT * FROM papers", conn)


def _prepare_papers(papers: pd.DataFrame) -> pd.DataFrame:
	plot_papers = papers.copy()
	plot_papers = plot_papers.dropna(subset=["primary_category", "pub_status", "submitted_year", "abstract_word_count"])
	plot_papers["submitted_year"] = pd.to_numeric(plot_papers["submitted_year"], errors="coerce")
	plot_papers["abstract_word_count"] = pd.to_numeric(plot_papers["abstract_word_count"], errors="coerce")
	plot_papers = plot_papers.dropna(subset=["submitted_year", "abstract_word_count"])
	plot_papers["submitted_year"] = plot_papers["submitted_year"].astype(int)
	plot_papers["abstract_word_count"] = plot_papers["abstract_word_count"].astype(int)
	return plot_papers


def _plot_category_summary(papers: pd.DataFrame) -> None:
	counts = (
		papers.groupby(["primary_category", "pub_status"]).size().unstack(fill_value=0).sort_index()
	)
	counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
	fig, ax = plt.subplots()
	colors = {"published": "#2f6f8f", "unpublished": "#d98c3b"}
	bottom = pd.Series([0] * len(counts), index=counts.index)
	for status in ["published", "unpublished"]:
		if status in counts.columns:
			ax.bar(counts.index, counts[status], bottom=bottom, label=status.title(), color=colors[status], width=0.8)
			bottom = bottom + counts[status]
	ax.set_title("Papers per Category by Publication Status")
	ax.set_xlabel("Category")
	ax.set_ylabel("Number of Papers")
	ax.legend(title="Publication Status")
	ax.tick_params(axis="x", rotation=45)
	_save(fig, PLOTS_DIR / "01_papers_per_category.png")


def _plot_submission_trend(papers: pd.DataFrame) -> None:
	by_year = papers.groupby(["submitted_year", "primary_category"]).size().reset_index(name="paper_count")
	top_categories = papers.groupby("primary_category").size().sort_values(ascending=False).head(5).index.tolist()
	by_year["category_group"] = by_year["primary_category"].where(by_year["primary_category"].isin(top_categories), "Other")
	plot_df = by_year.groupby(["submitted_year", "category_group"], as_index=False)["paper_count"].sum()
	years = sorted(plot_df["submitted_year"].unique().tolist())
	fig, ax = plt.subplots()
	palette = {
		"Other": "#7a7a7a",
		**{cat: color for cat, color in zip(top_categories, ["#215e92", "#2a9d8f", "#e76f51", "#8a5fbf", "#d4a017"])},
	}
	for group in top_categories + ["Other"]:
		series = plot_df[plot_df["category_group"] == group].set_index("submitted_year").reindex(years)["paper_count"].fillna(0)
		ax.plot(years, series.values, marker="o", linewidth=2.2, label=group, color=palette.get(group))
	ax.set_title("Submission Trend Over Time")
	ax.set_xlabel("Submission Year")
	ax.set_ylabel("Number of Papers")
	ax.legend(title="Category")
	if len(years) > 12:
		step = max(1, len(years) // 12)
		ax.set_xticks(years[::step])
	else:
		ax.set_xticks(years)
	_save(fig, PLOTS_DIR / "02_submission_trend_over_time.png")


def _plot_publication_breakdown(papers: pd.DataFrame) -> None:
	counts = papers["pub_status"].value_counts().reindex(["published", "unpublished"]).fillna(0)
	fig, ax = plt.subplots()
	ax.pie(
		counts.values,
		labels=[status.title() for status in counts.index],
		autopct="%1.1f%%",
		startangle=90,
		colors=["#2f6f8f", "#d98c3b"],
		textprops={"color": "#222222"},
	)
	ax.set_title("Publication Status Breakdown")
	_save(fig, PLOTS_DIR / "03_publication_status_breakdown.png")


def _plot_abstract_distribution(papers: pd.DataFrame) -> None:
	fig, ax = plt.subplots()
	category_counts = papers["primary_category"].value_counts()
	top_categories = category_counts.head(5).index.tolist()
	plot_data = papers.copy()
	plot_data["category_group"] = plot_data["primary_category"].where(
		plot_data["primary_category"].isin(top_categories),
		"Other",
	)
	group_order = top_categories + ["Other"]
	group_colors = {
		"Other": "#7a7a7a",
		**{
			category: color
			for category, color in zip(
				top_categories,
				["#215e92", "#2a9d8f", "#e76f51", "#8a5fbf", "#d4a017"],
			)
		},
	}
	for group in group_order:
		group_values = plot_data.loc[plot_data["category_group"] == group, "abstract_word_count"]
		if not group_values.empty:
			ax.hist(
				group_values,
				bins=30,
				alpha=0.38,
				label=group,
				color=group_colors.get(group),
				density=True,
			)
	ax.set_title("Abstract Length Distribution by Category")
	ax.set_xlabel("Abstract Word Count")
	ax.set_ylabel("Density")
	ax.legend(title="Category", ncol=2, fontsize=9)
	_save(fig, PLOTS_DIR / "04_abstract_length_distribution.png")


def main() -> None:
	args = _build_parser().parse_args()
	_setup_style()
	db_path = Path(args.db)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")
	with sqlite3.connect(db_path) as conn:
		_ensure_clean_tables(conn, args.rebuild_clean)
		papers = _load_papers(conn)

	plot_papers = _prepare_papers(papers)
	_plot_category_summary(plot_papers)
	_plot_submission_trend(plot_papers)
	_plot_publication_breakdown(plot_papers)
	_plot_abstract_distribution(plot_papers)
	print(f"Plots written to {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
	main()
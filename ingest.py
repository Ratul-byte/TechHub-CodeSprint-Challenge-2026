import argparse
from pathlib import Path
import sqlite3

import pandas as pd


DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "stat.ML", "cs.CV"]


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Ingest arXiv CSV into JSON and SQLite outputs."
	)
	parser.add_argument(
		"--input",
		default="sampled-arxiv-metadata-oai-snapshot.csv",
		help="Path to input CSV file.",
	)
	parser.add_argument(
		"--output-dir",
		default="data",
		help="Directory where outputs will be written.",
	)
	parser.add_argument(
		"--categories",
		nargs="*",
		default=None,
		help=(
			"Optional list of categories to keep. "
			"Example: --categories cs.AI cs.LG"
		),
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=None,
		help="Optional fixed number of rows to sample after filtering.",
	)
	parser.add_argument(
		"--sample-frac",
		type=float,
		default=None,
		help="Optional fraction of rows to sample after filtering (0, 1].",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed used for reproducible sampling.",
	)
	return parser


def _validate_columns(df: pd.DataFrame) -> None:
	required_input = [
		"id",
		"title",
		"abstract",
		"authors",
		"categories",
		"submitted",
		"update_date",
		"journal-ref",
		"doi",
		"comments",
	]
	missing = [col for col in required_input if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required input columns: {missing}")


def _filter_categories(df: pd.DataFrame, categories: list[str] | None) -> pd.DataFrame:
	if not categories:
		return df
	category_set = set(categories)

	def has_any_category(raw: str) -> bool:
		if pd.isna(raw):
			return False
		tokens = str(raw).split()
		return any(token in category_set for token in tokens)

	return df[df["categories"].apply(has_any_category)]


def _sample_subset(
	df: pd.DataFrame,
	sample_size: int | None,
	sample_frac: float | None,
	random_state: int,
) -> pd.DataFrame:
	if sample_size is not None and sample_frac is not None:
		raise ValueError("Use only one of --sample-size or --sample-frac.")
	if sample_frac is not None:
		if sample_frac <= 0 or sample_frac > 1:
			raise ValueError("--sample-frac must be in the range (0, 1].")
		return df.sample(frac=sample_frac, random_state=random_state)
	if sample_size is not None:
		if sample_size <= 0:
			raise ValueError("--sample-size must be a positive integer.")
		n = min(sample_size, len(df))
		return df.sample(n=n, random_state=random_state)
	return df


def main() -> None:
	args = _build_parser().parse_args()

	input_path = Path(args.input)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path)
	_validate_columns(df)

	selected_categories = args.categories if args.categories else None
	if selected_categories is None:
		selected_categories = DEFAULT_CATEGORIES

	df = _filter_categories(df, selected_categories)
	df = _sample_subset(df, args.sample_size, args.sample_frac, args.random_state)

	# Build required output schema.
	out_df = pd.DataFrame(
		{
			"arxiv_id": df["id"].astype(str),
			"title": df["title"],
			"abstract": df["abstract"],
			"authors": df["authors"],
			"categories": df["categories"],
			"primary_category": df["categories"].fillna("").apply(
				lambda x: str(x).split()[0] if str(x).strip() else None
			),
			"submitted": df["submitted"],
			"updated": df["update_date"],
			"journal_ref": df["journal-ref"],
			"doi": df["doi"],
			"comment": df["comments"],
		}
	)

	json_path = output_dir / "papers_raw.json"
	db_path = output_dir / "arxiv.db"

	out_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
	with sqlite3.connect(db_path) as conn:
		out_df.to_sql("raw_papers", conn, if_exists="replace", index=False)

	print(f"Rows written: {len(out_df)}")
	print(f"JSON output: {json_path}")
	print(f"SQLite output: {db_path} (table: raw_papers)")


if __name__ == "__main__":
	main()

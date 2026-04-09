PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS papers;
DROP TABLE IF EXISTS category_stats;
DROP TABLE IF EXISTS yearly_trends;
DROP TABLE IF EXISTS publication_status;
DROP TABLE IF EXISTS author_stats;

CREATE TABLE papers AS
WITH RECURSIVE
base AS (
	SELECT
		trim(CAST(arxiv_id AS TEXT)) AS arxiv_id,
		trim(COALESCE(title, '')) AS title,
		trim(COALESCE(abstract, '')) AS abstract,
		trim(COALESCE(authors, '')) AS authors,
		trim(COALESCE(primary_category, '')) AS primary_category,
		trim(COALESCE(submitted, '')) AS submitted,
		trim(COALESCE(updated, '')) AS updated,
		trim(COALESCE(journal_ref, '')) AS journal_ref,
		trim(COALESCE(doi, '')) AS doi,
		trim(COALESCE(comment, '')) AS comment
	FROM raw_papers
	WHERE trim(CAST(arxiv_id AS TEXT)) <> ''
),
deduped AS (
	SELECT
		*,
		ROW_NUMBER() OVER (
			PARTITION BY arxiv_id
			ORDER BY submitted DESC, updated DESC, title
		) AS rn
	FROM base
	WHERE title <> ''
	  AND abstract <> ''
	  AND authors <> ''
	  AND primary_category <> ''
	  AND submitted <> ''
),
submitted_tokens AS (
	SELECT arxiv_id, trim(submitted) || ' ' AS rest, 0 AS token_idx, '' AS token
	FROM deduped
	WHERE rn = 1
	UNION ALL
	SELECT
		arxiv_id,
		ltrim(substr(rest, instr(rest, ' ') + 1)),
		token_idx + 1,
		trim(substr(rest, 1, instr(rest, ' ') - 1))
	FROM submitted_tokens
	WHERE rest <> ''
),
submitted_years AS (
	SELECT arxiv_id, CAST(token AS INTEGER) AS submitted_year
	FROM submitted_tokens
	WHERE token_idx = 4
),
abstract_tokens AS (
	SELECT
		arxiv_id,
		trim(replace(replace(replace(abstract, char(13), ' '), char(10), ' '), char(9), ' ')) || ' ' AS rest,
		0 AS token_count
	FROM deduped
	WHERE rn = 1
	UNION ALL
	SELECT
		arxiv_id,
		ltrim(substr(rest, instr(rest, ' ') + 1)),
		token_count + CASE WHEN trim(substr(rest, 1, instr(rest, ' ') - 1)) <> '' THEN 1 ELSE 0 END
	FROM abstract_tokens
	WHERE rest <> ''
),
abstract_counts AS (
	SELECT arxiv_id, MAX(token_count) AS abstract_word_count
	FROM abstract_tokens
	GROUP BY arxiv_id
),
author_tokens AS (
	SELECT arxiv_id, trim(authors) || ',' AS rest, 0 AS token_count, '' AS author
	FROM deduped
	WHERE rn = 1
	UNION ALL
	SELECT
		arxiv_id,
		substr(rest, instr(rest, ',') + 1),
		token_count + 1,
		trim(substr(rest, 1, instr(rest, ',') - 1))
	FROM author_tokens
	WHERE rest <> ''
),
author_counts AS (
	SELECT arxiv_id, COUNT(*) AS author_count
	FROM author_tokens
	WHERE author <> ''
	GROUP BY arxiv_id
)
SELECT
	d.arxiv_id,
	d.title,
	d.abstract,
	d.authors,
	d.primary_category,
	d.submitted,
	COALESCE(a.abstract_word_count, 0) AS abstract_word_count,
	COALESCE(ac.author_count, 0) AS author_count,
	CASE
		WHEN instr(d.authors, ',') > 0 THEN trim(substr(d.authors, 1, instr(d.authors, ',') - 1))
		ELSE trim(d.authors)
	END AS first_author,
	sy.submitted_year,
	CASE
		WHEN instr(d.primary_category, '.') > 0 THEN substr(d.primary_category, 1, instr(d.primary_category, '.') - 1)
		ELSE d.primary_category
	END AS subject_area,
	CASE
		WHEN d.journal_ref <> '' THEN 'published'
		ELSE 'unpublished'
	END AS pub_status
FROM deduped d
JOIN submitted_years sy USING (arxiv_id)
JOIN abstract_counts a USING (arxiv_id)
JOIN author_counts ac USING (arxiv_id)
WHERE d.rn = 1;

CREATE TABLE category_stats AS
SELECT
	primary_category AS category,
	COUNT(*) AS total_papers,
	SUM(CASE WHEN pub_status = 'published' THEN 1 ELSE 0 END) AS published_count,
	ROUND(100.0 * SUM(CASE WHEN pub_status = 'published' THEN 1 ELSE 0 END) / COUNT(*), 2) AS publish_rate_pct
FROM papers
GROUP BY primary_category
ORDER BY category;

CREATE TABLE yearly_trends AS
SELECT
	submitted_year AS year,
	primary_category AS category,
	COUNT(*) AS paper_count
FROM papers
GROUP BY submitted_year, primary_category
ORDER BY year, category;

CREATE TABLE publication_status AS
SELECT
	pub_status,
	primary_category AS category,
	COUNT(*) AS paper_count
FROM papers
GROUP BY pub_status, primary_category
ORDER BY pub_status, category;

CREATE TABLE author_stats AS
WITH RECURSIVE
author_tokens AS (
	SELECT arxiv_id, primary_category, submitted_year, trim(authors) || ',' AS rest, '' AS author
	FROM papers
	UNION ALL
	SELECT
		arxiv_id,
		primary_category,
		submitted_year,
		substr(rest, instr(rest, ',') + 1),
		trim(substr(rest, 1, instr(rest, ',') - 1))
	FROM author_tokens
	WHERE rest <> ''
),
clean_authors AS (
	SELECT author, primary_category, submitted_year
	FROM author_tokens
	WHERE author <> ''
),
author_category_counts AS (
	SELECT
		author,
		primary_category AS category,
		COUNT(*) AS category_count
	FROM clean_authors
	GROUP BY author, primary_category
),
author_totals AS (
	SELECT
		author,
		COUNT(*) AS paper_count,
		MIN(submitted_year) AS first_year,
		MAX(submitted_year) AS last_year
	FROM clean_authors
	GROUP BY author
),
top_categories AS (
	SELECT author, category AS top_category
	FROM (
		SELECT
			author,
			category,
			category_count,
			ROW_NUMBER() OVER (
				PARTITION BY author
				ORDER BY category_count DESC, category ASC
			) AS rn
		FROM author_category_counts
	)
	WHERE rn = 1
)
SELECT
	at.author,
	at.paper_count,
	at.first_year,
	at.last_year,
	tc.top_category
FROM author_totals at
JOIN top_categories tc USING (author)
ORDER BY at.author;

-- Quality expectations:
-- 1. All five tables above are created by this script.
-- 2. papers excludes rows with null or empty mandatory fields.
-- 3. papers is deduplicated on arxiv_id.
-- 4. abstract_word_count, author_count, first_author, submitted_year, subject_area, pub_status are computed in papers.
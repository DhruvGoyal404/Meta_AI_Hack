# Data Cleaning & ETL OpenEnv Environment

This environment simulates a real-world data cleaning and ETL pipeline task where an AI agent must clean messy CSV/Excel data through a series of operations. It is designed for training and evaluating reinforcement learning agents on data preparation tasks.

## Motivation

Data cleaning is a common, time-consuming task in data science and analytics. Automating it with AI can save significant time and reduce errors. This environment provides a structured way to train agents to perform typical data cleaning operations: fixing nulls, correcting data types, removing duplicates, normalizing values, merging tables, and more.

## Action and Observation Spaces

### Actions
Each action is a JSON object with an `operation` field and other fields as needed.

- `fill_nulls`: Fill missing values in a column.
  - Required: `column`, `strategy` (`mean`, `median`, `mode`, `constant`, `forward_fill`, `backward_fill`)
  - Optional: `value` (for constant fill)
- `cast_column`: Change data type.
  - Required: `column`, `dtype` (`int`, `float`, `str`, `datetime`)
- `remove_duplicates`: Remove duplicate rows.
  - Optional: `subset` (list of columns), `keep` (`first`, `last`, `false`)
- `normalize_values`: Normalize string values.
  - Required: `column`, `method` (`lower`, `upper`, `regex`)
  - For regex: `pattern`, `replacement`
- `filter_outliers`: Remove outliers based on z-score.
  - Required: `column`, `method` (`zscore`)
  - Optional: `threshold` (default 3.0)
- `merge_tables`: Join two tables.
  - Required: `left_table`, `right_table`, `on`
  - Optional: `how` (`inner`, `left`, `right`, `outer`), `output_table`
- `add_derived_column`: Create a new column from an existing one.
  - Required: `column_name`, `source_column`, `transform` (`year_from_date`, `log1p`, `abs`, `len`, `upper`, `lower`)
- `submit`: End episode.

### Observations
Observations provide:
- Task description and progress (step count, max steps)
- Current state of tables (head, dtypes, null counts, duplicates, row counts)
- Reward and done flag
- Partial score (current grader score)

## Tasks

1. **Easy: Null Fixer**
   - Fix nulls and data types in a 50-row customer dataset (age as string, salary nulls).
   - Expected: age integer, salary integer, no nulls.
   - Max steps: 10

2. **Medium: Schema Normalizer**
   - Clean a 200-row orders dataset: remove duplicates, normalize country names (USA/usa/U.S.A → US), fix date format (YYYY/MM/DD → YYYY-MM-DD), fill null amounts.
   - Max steps: 20

3. **Hard: ETL Pipeline**
   - Join orders and customers tables (400 orders, 100 customers), remove outliers in amount (z-score >3), add a derived column `year` from date.
   - Max steps: 30

## Setup and Usage

### Local Development
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn server.app:app --reload --port 7860`
4. Interact with the environment using HTTP requests or the OpenEnv client.

### Docker
Build and run:
```bash
docker build -t dataclean-env ./server
docker run -p 7860:7860 dataclean-env
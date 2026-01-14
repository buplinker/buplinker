# BUPLinker

BUPLinker is a tool designed to link User Reviews (UR) and Pull Requests (PR) to analyze the relationship between user feedback and software development activities.

---

## 💻 Environment

- **OS**: macOS or Linux
- **Python**: 3.11
- **Git LFS**: Required for managing large files.
  ```bash
  brew install git-lfs && git lfs install
  ```
---
## ⚙️ Setup

### 1. Install Dependencies & Data

Run the following commands to download the necessary models, install libraries, and extract compressed results:

```shell
# Download FastText language identification model
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Install required Python packages
pip install -r requirements.txt
```

### 2. Configuration

* **GitHub**: Set your GitHub Auth Token in `project_config.py`.
* **OpenAI API**: Create a `.env` file in the root directory and add your key:
```text
OPENAI_API_KEY=your_api_key_here
```



---

## 🚀 How to Run BUPLinker

### Option A: Quick Start (Using Pre-prepared Data)

If you do not have a MySQL server, follow these steps:

#### 1. Extract Pre-computed Results

Extract the compressed output files:

```bash
tar xJf buplinker/code/output.tar.xz
tar xJf analysis/timeline/time_processed_data/all_years.tar.xz
tar xJf analysis/timeline/time_processed_data/limited_years.tar.xz
```

#### 2. Download Input Pairs Dataset

Download the `input_pairs` dataset from [Google Drive](https://drive.google.com/drive/folders/1eqfou_mbbqI0TqtCD8mk0l4BYTjavuxa?usp=drive_link).

#### 3. Place and Extract Dataset Files

Place the downloaded folder according to the [Project Structure](#-project-structure) and extract the compressed files:

```bash
tar xJf buplinker/dataset/input_pairs/ur_pr/all_years.tar.xz
tar xJf buplinker/dataset/input_pairs/ur_pr/limited_years.tar.xz
tar xJf buplinker/dataset/input_pairs/pr_ur/all_years.tar.xz
tar xJf buplinker/dataset/input_pairs/pr_ur/limited_years.tar.xz
```

#### 4. Proceed to Execution

Proceed directly to [Run BUPLinker Execution](#1-run-buplinker-execution).

### Option B: Full Pipeline (From Scratch)

If you want to fetch raw data and create your own tables, follow these steps:

> **Note**: It may take 2 or 3 days to fetch all data

#### 1. Set Your MySQL Server

Configure your MySQL credentials (URL, user name, password, and database name) in `project_config.py` if you have a MySQL server set up.

#### 2. Database & Data Fetching

```bash
# Create database tables
python3 data_fetch/database/tables.py

# Insert data from various sources
python3 data_fetch/repositories.py
python3 data_fetch/google_play_data.py
python3 data_fetch/github_data.py

```

#### 3. Preprocessing

Extract PR titles and templates for candidate selection:

```bash
python3 buplinker/dataset/preprocess/template_extractor.py

```

* Output: `buplinker/dataset/preprocess/template_title_repositories/*`

#### 4. Create Input Pairs

Filter candidate UR-PR pairs.

```bash
python3 buplinker/dataset/create_buplinker_input_pairs.py --limited

```

* `--limited`: Uses the first four years of data since the app's release.
* (Omit the flag to use all available data).

---

## 📊 Execution & Analysis

### 1. Run BUPLinker Execution

Apply the linking algorithm to the prepared input pairs:

```bash
bash ./buplinker/code/buplinker.sh

```

> **Note**: To switch between "limited" (4 years) and "all years", modify the `LIMITED` variable inside the `.sh` script.
> **Note**: Running BUPLinker for all repositories will cost around $150

### 2. Data Formatting for Analysis

If you have a MySQL server set up, format the output into timeline-friendly data.
Otherwise, skip this step:

```bash
python3 analysis/timeline/time_processed_data/create_timeline_data.py --limited

```

### 3. Run Analysis Metrics

Generate the final statistics for linked ratio and time:

| Task | Command | Output Directory |
| --- | --- | --- |
| **Analyze Linked Ratio** | `python3 analysis/timeline/linked_ratio.py --limited` | `analysis/timeline/results/linked_ratio/` |
| **Analyze Linked Time** | `python3 analysis/timeline/linked_time.py --limited` | `analysis/timeline/results/linked_time/` |

---

## 📘 How to Evaluate BUPLinker

Execute BUPLinker to perform UR→PR and PR→UR link prediction and evaluation based on the generated CSV files.

The predicted link results are saved as CSV/JSON files.  
Evaluation metrics (precision, recall, F1, etc.) are output to stdout and log files.

**Before running**, edit `buplinker/code/buplinker.sh` and set the following variables:

* `EVALUATION=true`: Enable evaluation mode (outputs evaluation metrics).
* `LIMITED=true`: Uses the first four years of data (limited_random).

Then run:

```bash
bash ./buplinker/code/buplinker.sh
```

* Output: `buplinker/code/output/{group_type}/limited_random/` or `buplinker/code/output/{group_type}/all_random/`

---

## 📂 Project Structure

```
buplinker/
├── buplinker/
│   ├── code/
│   │   ├── buplinker.py          # Main BUPLinker execution script
│   │   ├── buplinker.sh          # Batch processing script
│   │   ├── util.py               # Utility functions
│   │   ├── prompts/              # LLM prompts for UR-PR and PR-UR linking
│   │   └── output/               # BUPLinker execution results
│   └── dataset/
│       ├── create_buplinker_input_pairs.py  # Generate candidate pairs
│       ├── preprocess/
│       │   ├── template_extractor.py            # Extract PR/Issue templates
│       │   ├── preprocess_pr.py                 # Preprocess PR descriptions
│       │   ├── label_user_review.py             # Label user reviews with ARdoc
│       │   ├── label_repository.py              # Label repositories with categories
│       │   ├── groundtruthbots.csv              # List of bot-generated PRs used during preprocessing
│       │   ├── template_titles.csv              # Template title extraction results
│       │   ├── template_titles_repositories/    # Repository template titles (per repository)
│       │   └── prompts/                         # LLM prompts for template extraction
│       └── input_pairs/                              # Input pair datasets
│           ├── pr_ur/                                # PR → UR candidate pairs
│           │   ├── limited_random_input_pairs.csv    # Evaluation results of randomly sampled data
│           │   ├── limited_years/                    # First 4 years of data
│           │   └── all_years/                        # All available years
│           └── ur_pr/                                # UR → PR candidate pairs
│               ├── limited_random_input_pairs.csv    # Evaluation results of randomly sampled data
│               ├── limited_years/                    # First 4 years of data
│               └── all_years/                        # All available years
├── data_fetch/
│   ├── database/
│   │   ├── tables.py             # Database schema definitions
│   │   ├── get.py                # Database read operations
│   │   └── set.py                # Database write operations
│   ├── github_data.py            # Fetch GitHub data (PRs, Issues, Releases)
│   ├── google_play_data.py       # Fetch Google Play Store reviews
│   ├── repositories.py           # Load repository data from CSV and add to database
│   ├── template_fetcher.py       # Fetch PR/Issue templates from GitHub
│   ├── query_templates/          # GraphQL query templates for GitHub API
│   │   ├── issues.graphql        # GraphQL query for fetching GitHub issues
│   │   ├── pullRequests.graphql  # GraphQL query for fetching GitHub pull requests
│   │   └── releases.graphql      # GraphQL query for fetching GitHub releases
│   └── tables/ 
│       └── repositories.csv      # List of analyzed repositories                   
├── analysis/
│   └── timeline/
│       ├── linked_ratio.py              # Analyze linking ratio metrics
│       ├── linked_time.py               # Analyze linking time metrics
│       ├── base_plotter.py              # Base plotting utilities
│       ├── statistics_analyzer.py       # Statistical analysis utilities
│       ├── statistics_types.py          # Type definitions for statistics
│       ├── time_processed_data/         # Formatted data for timeline analysis
│       │   ├── create_timeline_data.py  # Convert BUPLinker results to timeline format
│       │   ├── limited_years/           # Processed data for first 4 years
│       │   └── all_years/               # Processed data for all years
│       └── results/                     # Analysis results and visualizations
│           ├── linked_ratio/            # Linking ratio analysis results
│           └── linked_time/             # Linking time analysis results
├── project_config.py             # Configuration file (GitHub token, MySQL settings)
├── root_util.py                  # Root-level utility functions
└── requirements.txt              # Python dependencies
```

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

# Extract pre-computed results
tar -xJf buplinker/code/output.tar.xz

```

### 2. Configuration

* **GitHub & MySQL**: Set your GitHub Auth Token and MySQL credentials (user, password, host, database) in `project_config.py`.
* **OpenAI API**: Create a `.env` file in the root directory and add your key:
```text
OPENAI_API_KEY=your_api_key_here
```



---

## 🚀 How to Run BUPLinker

### Option A: Quick Start (Using Pre-prepared Data)

If you do not have a MySQL server, download the `input_pairs` dataset:

1. Download from [Google Drive](https://drive.google.com/drive/folders/1eqfou_mbbqI0TqtCD8mk0l4BYTjavuxa?usp=drive_link).
2. Place the folder according to the [Project Structure](#-project-structure).
3. Proceed directly to [Run BUPLinker Execution](#1-run-buplinker-execution).

### Option B: Full Pipeline (From Scratch)

If you want to fetch raw data and create your own tables, follow these steps:

#### 1. Database & Data Fetching

```bash
# Create database tables
python3 data_fetch/database/tables.py

# Insert data from various sources
python3 data_fetch/repositories.py
python3 data_fetch/google_play_data.py
python3 data_fetch/github_data.py

```

#### 2. Preprocessing

Extract PR titles and templates for candidate selection:

```bash
python3 buplinker/dataset/preprocess/template_extractor.py

```

* Output: `buplinker/dataset/preprocess/template_title_repositories/*`

#### 3. Create Input Pairs

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
cd buplinker/code && bash buplinker.sh

```

> **Note**: To switch between "limited" (4 years) and "all years", modify the `LIMITED` variable inside the `.sh` script.

### 2. Data Formatting for Analysis

Format the output into timeline-friendly data:

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
│       └── input_pairs/                  # Input pair datasets
│           ├── pr_ur/                    # PR → UR candidate pairs
│           │   ├── limited_random.csv    # Evaluation results of randomly sampled data
│           │   ├── limited_years/        # First 4 years of data
│           │   └── all_years/            # All available years
│           └── ur_pr/                    # UR → PR candidate pairs
│               ├── limited_random.csv    # Evaluation results of randomly sampled data
│               ├── limited_years/        # First 4 years of data
│               └── all_years/            # All available years
├── data_fetch/
│   ├── database/
│   │   ├── tables.py             # Database schema definitions
│   │   ├── get.py                # Database read operations
│   │   └── set.py                # Database write operations
│   ├── github_data.py            # Fetch GitHub data (PRs, Issues, Releases)
│   ├── google_play_data.py       # Fetch Google Play Store reviews
│   ├── template_fetcher.py       # Fetch PR/Issue templates from GitHub
│   └── query_templates/          # GraphQL query templates for GitHub API
│       ├── issues.graphql
│       ├── pullRequests.graphql
│       └── releases.graphql
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
├── repositories.csv              # List of analyzed repositores
└── requirements.txt              # Python dependencies
```

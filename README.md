# Environment
- Git LFS: `brew install git-lfs && git lfs install`
- Python 3.11
- macOS or Linux

# Setup

```shell
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
pip install -r requirements.txt
tar -xJf analysis/timeline/results.tar.xz
tar -xJf buplinker/code/output.tar.xz
```
Set your github auth token into `project_config.py` and cerate `.env` file and set your oprnai api key into there, like `OPENAI_API_KEY=hoge`.

# Run BUPLinker
## Set input data for BUPLinker
If you don't have mysql server or if you want to run the BUPLinker,
place the following data into `buplinker/dataset`, naming `input_pairs`:
[https://drive.google.com/drive/folders/1eqfou_mbbqI0TqtCD8mk0l4BYTjavuxa?usp=sharing](https://drive.google.com/drive/folders/1DtnC7vLlhqXIuKDXdRs7dIGAEYxKgfNW?usp=sharing)
If you want to creat tables and set all raw data, run the following step, how to create tables and fetch data.

## 🗄️ How to Create Tables and Fetch Data
### 1. Set mysgl database
Set your mysql user, password, host, dabase into `project_config.py`. 

### 2. Create tables
```bash
python3 data_fetch/database/tables.py
```
### 3. Insert repository data
```bash
python3 data_fetch/repositories.py
```
### 4. Insert user reviews
```bash
python3 data_fetch/google_play_data.py
```
### 5. Insert Github data (pull request, issue, release, pull request template, and issue template)
```bash
python3 data_fetch/github_data.py
```
### 6. Extract PR / Template Titles  
Extract and use PR tiles and templates for preprocessing.
#### Command

```bash
python3 buplinker/dataset/preprocess/template_extractor.py
```

#### Output directory

```bash
buplinker/dataset/preprocess/template_title_repositories/
```

### 7. Create Input Pairs

Create candidate UR-PR pairs.
This step is needed because BUPLinker does not compare the all UR–PR, PR-UR but filter candidate pairs.

#### Command
```bash
python3 buplinker/dataset/create_buplinker_input_pairs.py --limited
```
`--limited` means BUPLinker uses first four years data since the apps release.
If not set the argument, BUPLinker uses all data avilable.

#### Output directory
```bash
buplinker/dataset/input_pairs/ur_pr/limited_years
```
and
```bash
buplinker/dataset/input_pairs/pr_ur/limited_years
```
If you not set `--limited` as an argument, the pairs are output into `all_years` directory.

## 📊 How to Run BUPLinker and Analyze the Output

### 1. Run BUPLinker
Apply BUPLinker on the prepared input pairs.

#### Command
```bash
bash ./buplinker/code/buplinker.sh
```
If you change `LIMITED` value into `false`, BUPLinker uses the four years data since the apps release.

#### Output directory
```bash
buplinker/code/output/ur_pr/limited_years/results
```
and
```bash
buplinker/code/output/pr_ur/limited_years/results
```
If you change `LIMITED` value into `false`, the results are output into  `all_years` directory.

### 2. Format input data for analyzing

#### Command

```bash
python3 analysis/timeline/time_processed_data/create_timeline_data.py --limited
```
If you not set `--limited` as an argument, the pairs are output into `all_years` directory.

#### Output directory

```bash
analysis/timeline/time_processed_data/limited_years
```
If you not set `--limited` as an argument, the pairs are output into `all_years` directory.

### 2. Analyze Linked Ratio

#### Command

```bash
python3 analysis/timeline/linked_ratio.py --limited
```

#### Output directory

```bash
analysis/timeline/results/linked_ratio/limited_years
```

### 3. Analyze Linked Time

#### Command

```bash
python3 analysis/timeline/linked_time.py --limited
```

#### Output directory

```bash
analysis/timeline/results/linked_time/limited_years
```

## Structure

<!--
**buplinker/buplinker** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

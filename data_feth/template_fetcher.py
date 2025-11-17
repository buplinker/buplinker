#!python3
import subprocess
import tempfile
import os
import shutil
from datetime import datetime
import hashlib
import sys
import yaml
import glob
from tqdm import tqdm
import data_fetch.database.get as data_getter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import database.set as data_setter
from database.tables import Repository, PullRequestTemplate, IssueTemplate
from root_util import target_repos


def clone_or_update_repository(repo: Repository) -> str:
    """
    リポジトリをcloned_repositoriesディレクトリにクローンまたは更新する
    """
    # github_repositoriesディレクトリを作成
    repos_dir = os.path.join(os.path.dirname(__file__), "cloned_repositories")
    os.makedirs(repos_dir, exist_ok=True)
    
    # リポジトリのクローン先パス
    repo_dir = os.path.join(repos_dir, f"{repo.owner}.{repo.name}")
    repo_url = f"https://github.com/{repo.owner}/{repo.name}.git"
    
    # 既にクローンされている場合は更新、そうでなければクローン
    if os.path.exists(repo_dir):
        #TODO: リポジトリのデフォルトブランチをGitHub APIなどから取得して利用することを検討しても良いかもしれません。(feedbackType: IMPROVEMENT)
        try:
            # git pullで更新
            subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Updated existing repository: {repo_dir}")
            return repo_dir
        except subprocess.CalledProcessError as e:
            # mainブランチがない場合はmasterを試す
            try:
                subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Updated existing repository (master): {repo_dir}")
                return repo_dir
            except subprocess.CalledProcessError:
                # 更新に失敗した場合は削除して再クローン
                print(f"Failed to update repository, removing and re-cloning: {repo_dir}")
                shutil.rmtree(repo_dir, ignore_errors=True)
    
    # クローン
    try:
        subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Cloned new repository: {repo_dir}")
        return repo_dir
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository {repo.owner}.{repo.name}: {e}", file=sys.stderr)
        shutil.rmtree(repo_dir, ignore_errors=True)
        return None


def check_pull_request_template_exists(repo_path: str) -> str or None:
    """
    プルリクエストテンプレートファイルのパスを検出する（最初に見つかったものを返す）
    """
    possible_paths = [
        ".github/PULL_REQUEST_TEMPLATE.md",  
        "PULL_REQUEST_TEMPLATE.md",
        "docs/PULL_REQUEST_TEMPLATE.md",
        ".github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md",
    ]
    
    for path in possible_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            # ディレクトリをスキャンして実際のファイル名を取得
            dir_path = os.path.dirname(full_path)
            filename = os.path.basename(path)
            
            if os.path.exists(dir_path):
                for actual_file in os.listdir(dir_path):
                    if actual_file.lower() == filename.lower():
                        actual_path = os.path.join(os.path.dirname(path), actual_file)
                        print(f"Found PR template file: {actual_path}")
                        return actual_path
            
            # フォールバック: 元のパスを返す
            print(f"Found PR template file: {path}")
            return path
    
    return None


def check_all_pull_request_template_paths(repo_path: str) -> list:
    """
    プルリクエストテンプレートファイルのパスを全て検出する（複数のパスが存在する場合に対応）
    Git履歴から過去に存在していたファイルも含めて検出する
    """
    possible_paths = [
        ".github/PULL_REQUEST_TEMPLATE.md",  
        "PULL_REQUEST_TEMPLATE.md",
        "docs/PULL_REQUEST_TEMPLATE.md",
        ".github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md",
    ]
    
    found_paths = []
    
    # Git履歴から過去に存在していた全てのテンプレートファイルパスを取得
    try:
        # 全ての履歴からテンプレート関連ファイルを検索
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 履歴から見つかったファイルパスを収集
            all_files_in_history = set(result.stdout.strip().split('\n'))
            
            # テンプレートファイルのパターンに一致するものを探す
            for file_path in all_files_in_history:
                if not file_path:
                    continue
                
                # ファイル名がPULL_REQUEST_TEMPLATEに関連するものをチェック
                file_lower = file_path.lower()
                if 'pull_request_template' in file_lower and (file_path.endswith('.md') or file_path.endswith('.txt')):
                    # 相対パスに変換（必要に応じて）
                    if file_path not in found_paths:
                        found_paths.append(file_path)
                        print(f"Found PR template file in history: {file_path}")
    except Exception as e:
        print(f"Error searching Git history for template files: {e}")
    
    # 現在存在するファイルもチェック
    for path in possible_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            # ディレクトリをスキャンして実際のファイル名を取得
            dir_path = os.path.dirname(full_path)
            filename = os.path.basename(path)
            
            if os.path.exists(dir_path):
                for actual_file in os.listdir(dir_path):
                    if actual_file.lower() == filename.lower():
                        actual_path = os.path.join(os.path.dirname(path), actual_file)
                        if actual_path not in found_paths:
                            found_paths.append(actual_path)
                            print(f"Found PR template file (current): {actual_path}")
                        break
            else:
                # フォールバック: 元のパスを返す
                if path not in found_paths:
                    found_paths.append(path)
                    print(f"Found PR template file (current): {path}")
    
    # .github/PULL_REQUEST_TEMPLATE/ ディレクトリ内のファイルもGit履歴から検索
    try:
        # ディレクトリが過去に存在したかチェック
        dir_result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--", ".github/PULL_REQUEST_TEMPLATE/"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if dir_result.returncode == 0:
            dir_files = set(dir_result.stdout.strip().split('\n'))
            for file_path in dir_files:
                if not file_path:
                    continue
                # .mdまたは.txtファイルで、PULL_REQUEST_TEMPLATEディレクトリ内のもの
                if (file_path.startswith('.github/PULL_REQUEST_TEMPLATE/') and 
                    (file_path.endswith('.md') or file_path.endswith('.txt')) and
                    file_path not in found_paths):
                    found_paths.append(file_path)
                    print(f"Found PR template file in directory history: {file_path}")
    except Exception as e:
        print(f"Error searching directory history: {e}")
    
    return found_paths


def check_issue_template_exists(repo_path: str) -> list:
    """
    イシューテンプレートファイルのパスを検出する（MarkdownとYAML両方対応）
    Git履歴から過去に存在していたファイルも含めて検出する
    """
    found_templates = []
    found_paths_set = set()  # 重複チェック用
    
    # Git履歴から過去に存在していた全てのissueテンプレートファイルパスを取得
    try:
        # 全ての履歴からテンプレート関連ファイルを検索
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 履歴から見つかったファイルパスを収集
            all_files_in_history = set(result.stdout.strip().split('\n'))
            
            # テンプレートファイルのパターンに一致するものを探す
            for file_path in all_files_in_history:
                if not file_path:
                    continue
                
                file_lower = file_path.lower()
                
                # Issue template関連のファイルをチェック
                if 'issue_template' in file_lower:
                    # YAMLファイル
                    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        if file_path not in found_paths_set:
                            found_paths_set.add(file_path)
                            found_templates.append({"path": file_path, "type": "yaml"})
                            print(f"Found YAML issue template in history: {file_path}")
                    # Markdownファイル
                    elif file_path.endswith('.md') or file_path.endswith('.txt'):
                        if file_path not in found_paths_set:
                            found_paths_set.add(file_path)
                            found_templates.append({"path": file_path, "type": "markdown"})
                            print(f"Found markdown issue template in history: {file_path}")
    except Exception as e:
        print(f"Error searching Git history for issue template files: {e}")
    
    # .github/ISSUE_TEMPLATE/ ディレクトリ内のファイルもGit履歴から検索
    try:
        dir_result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--", ".github/ISSUE_TEMPLATE/"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if dir_result.returncode == 0:
            dir_files = set(dir_result.stdout.strip().split('\n'))
            for file_path in dir_files:
                if not file_path:
                    continue
                # .github/ISSUE_TEMPLATE/ ディレクトリ内のファイル
                if (file_path.startswith('.github/ISSUE_TEMPLATE/') and file_path not in found_paths_set):
                    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        found_paths_set.add(file_path)
                        found_templates.append({"path": file_path, "type": "yaml"})
                        print(f"Found YAML issue template in directory history: {file_path}")
                    elif file_path.endswith('.md') or file_path.endswith('.txt'):
                        found_paths_set.add(file_path)
                        found_templates.append({"path": file_path, "type": "markdown"})
                        print(f"Found markdown issue template in directory history: {file_path}")
    except Exception as e:
        print(f"Error searching directory history: {e}")
    
    # 現在存在するMarkdownテンプレートの検索
    markdown_paths = [
        ".github/ISSUE_TEMPLATE.md",
        "ISSUE_TEMPLATE.md", 
        "docs/ISSUE_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/ISSUE_TEMPLATE.md",
    ]
    
    for path in markdown_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            # ディレクトリをスキャンして実際のファイル名を取得
            dir_path = os.path.dirname(full_path)
            filename = os.path.basename(path)
            
            if os.path.exists(dir_path):
                for actual_file in os.listdir(dir_path):
                    if actual_file.lower() == filename.lower():
                        actual_path = os.path.join(os.path.dirname(path), actual_file)
                        if actual_path not in found_paths_set:
                            found_paths_set.add(actual_path)
                            found_templates.append({"path": actual_path, "type": "markdown"})
                            print(f"Found markdown issue template (current): {actual_path}")
                        break
    
    # 現在存在するYAMLテンプレートの検索
    yaml_template_dir = os.path.join(repo_path, ".github/ISSUE_TEMPLATE")
    if os.path.exists(yaml_template_dir):
        yaml_files = glob.glob(os.path.join(yaml_template_dir, "*.yaml")) + glob.glob(os.path.join(yaml_template_dir, "*.yml"))
        for yaml_file in yaml_files:
            # 相対パスに変換（Gitコマンド用）
            relative_path = os.path.relpath(yaml_file, repo_path)
            if relative_path not in found_paths_set:
                found_paths_set.add(relative_path)
                found_templates.append({"path": relative_path, "type": "yaml"})
                print(f"Found YAML issue template (current): {relative_path}")
        
        # Markdownテンプレートの検索（.github/ISSUE_TEMPLATE/ ディレクトリ内）
        md_files = glob.glob(os.path.join(yaml_template_dir, "*.md"))
        for md_file in md_files:
            # 相対パスに変換（Gitコマンド用）
            relative_path = os.path.relpath(md_file, repo_path)
            if relative_path not in found_paths_set:
                found_paths_set.add(relative_path)
                found_templates.append({"path": relative_path, "type": "markdown"})
                print(f"Found Markdown issue template (current): {relative_path}")
    
    return found_templates


def parse_yaml_issue_template(yaml_path: str) -> str:
    """
    YAML形式のissueテンプレートから重要な項目のみを抽出してMarkdown形式に変換
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)
        
        if not template_data:
            return ""
        
        # 基本情報を抽出
        name = template_data.get('name', '')
        description = template_data.get('description', '')
        title = template_data.get('title', '')
        
        # bodyセクションから重要な項目を抽出
        body_content = ""
        if 'body' in template_data and isinstance(template_data['body'], list):
            for item in template_data['body']:
                if item.get('type') == 'markdown':
                    # markdownコンテンツをそのまま追加
                    markdown_content = item.get('attributes', {}).get('value', '')
                    if markdown_content:
                        body_content += f"{markdown_content}\n\n"
                
                elif item.get('type') == 'textarea':
                    # textareaフィールドの情報を抽出
                    attributes = item.get('attributes', {})
                    label = attributes.get('label', '')
                    description_text = attributes.get('description', '')
                    placeholder = attributes.get('placeholder', '')
                    required = item.get('validations', {}).get('required', False)
                    
                    if label:
                        required_mark = " (required)" if required else ""
                        body_content += f"## {label}{required_mark}\n"
                        if description_text:
                            body_content += f"{description_text}\n"
                        if placeholder:
                            body_content += f"*{placeholder}*\n"
                        body_content += "\n"
                
                elif item.get('type') == 'input':
                    # inputフィールドの情報を抽出
                    attributes = item.get('attributes', {})
                    label = attributes.get('label', '')
                    description_text = attributes.get('description', '')
                    placeholder = attributes.get('placeholder', '')
                    required = item.get('validations', {}).get('required', False)
                    
                    if label:
                        required_mark = " (required)" if required else ""
                        body_content += f"## {label}{required_mark}\n"
                        if description_text:
                            body_content += f"{description_text}\n"
                        if placeholder:
                            body_content += f"*{placeholder}*\n"
                        body_content += "\n"
                
                elif item.get('type') == 'dropdown':
                    # dropdownフィールドの情報を抽出
                    attributes = item.get('attributes', {})
                    label = attributes.get('label', '')
                    description_text = attributes.get('description', '')
                    options = attributes.get('options', [])
                    required = item.get('validations', {}).get('required', False)
                    
                    if label:
                        required_mark = " (required)" if required else ""
                        body_content += f"## {label}{required_mark}\n"
                        if description_text:
                            body_content += f"{description_text}\n"
                        if options:
                            body_content += "Options:\n"
                            for option in options:
                                body_content += f"- {option}\n"
                        body_content += "\n"
                
                elif item.get('type') == 'checkboxes':
                    # checkboxesフィールドの情報を抽出
                    attributes = item.get('attributes', {})
                    label = attributes.get('label', '')
                    description_text = attributes.get('description', '')
                    options = attributes.get('options', [])
                    
                    if label:
                        body_content += f"## {label}\n"
                        if description_text:
                            body_content += f"{description_text}\n"
                        if options:
                            for option in options:
                                option_label = option.get('label', '')
                                option_required = option.get('required', False)
                                required_mark = " (required)" if option_required else ""
                                body_content += f"- {option_label}{required_mark}\n"
                        body_content += "\n"
        
        # 最終的なMarkdownテンプレートを構築
        template_markdown = ""
        if name:
            template_markdown += f"# {name}\n\n"
        if description:
            template_markdown += f"{description}\n\n"
        if title:
            template_markdown += f"**Title format:** {title}\n\n"
        
        if body_content:
            template_markdown += body_content
        
        return template_markdown.strip()
        
    except Exception as e:
        print(f"Error parsing YAML template {yaml_path}: {e}")
        return ""


def get_existing_templates_content(repository_id: int) -> dict:
    """
    既存のテンプレート内容を取得してハッシュマップとして返す
    """
    existing_templates = {}
    try:
        if data_getter.has_issue_templates(repository_id):
            templates = data_getter.issue_templates(repository_id)
            for template in templates:
                # テンプレート内容のハッシュをキーとして使用
                content_hash = hashlib.md5(template.template.encode('utf-8')).hexdigest()
                existing_templates[content_hash] = template
    except Exception as e:
        print(f"Error getting existing templates: {e}")
    
    return existing_templates


def template_content_changed(new_content: str, existing_templates: dict) -> bool:
    """
    新しいテンプレート内容が既存のものと異なるかチェック
    """
    if not new_content:
        return False
    
    new_content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
    return new_content_hash not in existing_templates


def get_yaml_template_content_at_commit(repo_path: str, commit_hash: str, template_path: str) -> str:
    """
    特定のコミット時点でのYAMLテンプレート内容を取得し、Markdown形式に変換
    """
    try:
        # まず、そのコミット時点でファイルが存在するかチェック
        check_result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_hash}:{template_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if check_result.returncode != 0:
            # ファイルが存在しない場合は空文字を返す（エラーではない）
            return ""
        
        # 特定のコミット時点でのファイル内容を取得
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{template_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error getting template content for commit {commit_hash[:8]}: {result.stderr}")
            return ""
        
        yaml_content = result.stdout
        
        # 一時ファイルにYAML内容を保存してパース
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name
        
        try:
            # YAMLテンプレートをMarkdown形式に変換
            markdown_content = parse_yaml_issue_template(temp_file_path)
            return markdown_content
        finally:
            # 一時ファイルを削除
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Error getting YAML template content for commit {commit_hash[:8]}: {e}")
        return ""


def get_template_commits(repo_path: str, template_path: str) -> list:
    """
    プルリクエストテンプレートファイルの変更履歴を取得する
    """
    # git log --follow でテンプレートファイルの履歴を取得
    result = subprocess.run(
        ["git", "log", "--follow", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso", "--", template_path],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        # ファイルが存在しない場合やエラーの場合
        if result.returncode == 128:  # git log のエラー（ファイルが存在しないなど）
            print(f"Template file {template_path} not found in {repo_path}", file=sys.stderr)
        else:
            print(f"Git log failed with return code {result.returncode}: {result.stderr}", file=sys.stderr)
        return []
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if line:
            parts = line.split('|', 4)
            if len(parts) >= 5:
                commit_hash, author_name, author_email, date_str, message = parts
                commits.append({
                    'hash': commit_hash,
                    'author_name': author_name,
                    'author_email': author_email,
                    'date': datetime.fromisoformat(date_str.replace(' +', '+')),
                    'message': message
                })
    return commits


def get_template_content_at_commit(repo_path: str, commit_hash: str, template_path: str) -> str:
    """
    特定のコミット時点でのテンプレート内容を取得する
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{template_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to get template content for commit {commit_hash}: {e}", file=sys.stderr)
        return ""


def fetch_pull_request_templates(repo: Repository): 
    # リポジトリをクローンまたは更新
    repo_path = clone_or_update_repository(repo)
    if not repo_path:
        return
    
    try:
        # テンプレートファイルのパスを全て検出（複数のパスが存在する場合に対応）
        template_paths = check_all_pull_request_template_paths(repo_path)
        if not template_paths:
            print(f"No PR template file found for {repo.owner}.{repo.name}")
            return
        
        # 各テンプレートパスを処理
        for template_path in template_paths:
            print(f"Processing PR template: {template_path}")
            
            # テンプレートの変更履歴を取得
            commits = get_template_commits(repo_path, template_path)
            
            if not commits:
                print(f"No PR template history found for {template_path}")
                continue
                    
            # 各コミットのテンプレート内容を取得して保存
            for i, commit in tqdm(enumerate(commits), desc=f"Processing PR templates: {template_path}"):
                
                # テンプレート内容を取得
                template_content = get_template_content_at_commit(repo_path, commit['hash'], template_path)
                
                if template_content:                
                    # データベースに保存
                    template = PullRequestTemplate(
                        id=commit['hash'],
                        file_path=template_path,
                        repository_id=repo.id,
                        template=template_content,
                        created_at=commit['date'],
                        author=commit['author_name']
                    )
                    
                    data_setter.add_pull_request_template(repo.id, template)
                else:
                    print(f"No PR template content found for commit {commit['hash'][:8]}")
    
    except Exception as e:
        print(f"Error fetching pull request templates for {repo.owner}.{repo.name}: {e}", file=sys.stderr)


def fetch_issue_templates(repo: Repository):  
    # リポジトリをクローンまたは更新
    repo_path = clone_or_update_repository(repo)
    if not repo_path:
        return
    
    try:
        # テンプレートファイルのパスを検出（MarkdownとYAML両方対応）
        template_files = check_issue_template_exists(repo_path)
        if not template_files:
            print(f"No issue template files found for {repo.owner}.{repo.name}")
            return
        
        # 各テンプレートファイルを処理
        for template_info in template_files:
            template_path = template_info['path']
            template_type = template_info['type']
            
            print(f"Processing {template_type} template: {template_path}")
            
            # テンプレートの変更履歴を取得
            commits = get_template_commits(repo_path, template_path)
            
            if not commits:
                print(f"No issue template history found for {template_path}")
                continue
                    
            # 各コミットのテンプレート内容を取得して保存
            for i, commit in tqdm(enumerate(commits), desc=f"Processing {template_type} issue templates"):
                
                # テンプレート内容を取得
                if template_type == 'yaml':
                    # YAMLテンプレートの場合は特別な処理
                    template_content = get_yaml_template_content_at_commit(repo_path, commit['hash'], template_path)
                else:
                    # Markdownテンプレートの場合は従来の処理
                    template_content = get_template_content_at_commit(repo_path, commit['hash'], template_path)
                
                if template_content:                
                    # データベースに保存（差分チェックを無効化）
                    template = IssueTemplate(
                        id=commit['hash'],
                        repository_id=repo.id,
                        file_path=template_path,
                        template=template_content,
                        created_at=commit['date'],
                        author=commit['author_name']
                    )
                    
                    data_setter.add_issue_template(repo.id, template)
                    print(f"Added template from commit {commit['hash'][:8]}")
                else:
                    print(f"No issue template content found for commit {commit['hash'][:8]}")
    
    except Exception as e:
        print(f"Error fetching issue templates for {repo.owner}.{repo.name}: {e}", file=sys.stderr)
    

if __name__ == "__main__":
    for repo in target_repos():
        print(f"{repo.id}: {repo.owner}.{repo.name}")
        
        try:
            # プルリクエストテンプレートを取得
            print(f"Fetching PR templates for {repo.owner}.{repo.name}...")
            fetch_pull_request_templates(repo)
            
            # イシューテンプレートを取得
            print(f"Fetching issue templates for {repo.owner}.{repo.name}...")
            fetch_issue_templates(repo)
            
        except Exception as e:
            print(f"Error processing repository {repo.owner}.{repo.name}: {e}", file=sys.stderr)
            continue
    
    print("Done!")
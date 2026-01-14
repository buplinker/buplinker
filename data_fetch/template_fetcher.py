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
    Clone or update repository to cloned_repositories directory
    """
    # Create github_repositories directory
    repos_dir = os.path.join(os.path.dirname(__file__), "cloned_repositories")
    os.makedirs(repos_dir, exist_ok=True)
    
    # Repository clone destination path
    repo_dir = os.path.join(repos_dir, f"{repo.owner}.{repo.name}")
    repo_url = f"https://github.com/{repo.owner}/{repo.name}.git"
    
    # Update if already cloned, otherwise clone
    if os.path.exists(repo_dir):
        #TODO: Consider fetching the repository's default branch from GitHub API and using it. (feedbackType: IMPROVEMENT)
        try:
            # Update with git pull
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
            # Try master branch if main branch doesn't exist
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
                # Remove and re-clone if update fails
                print(f"Failed to update repository, removing and re-cloning: {repo_dir}")
                shutil.rmtree(repo_dir, ignore_errors=True)
    
    # Clone
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
    Detect pull request template file path (returns the first one found)
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
            # Scan directory to get actual filename
            dir_path = os.path.dirname(full_path)
            filename = os.path.basename(path)
            
            if os.path.exists(dir_path):
                for actual_file in os.listdir(dir_path):
                    if actual_file.lower() == filename.lower():
                        actual_path = os.path.join(os.path.dirname(path), actual_file)
                        print(f"Found PR template file: {actual_path}")
                        return actual_path
            
            # Fallback: return original path
            print(f"Found PR template file: {path}")
            return path
    
    return None


def check_all_pull_request_template_paths(repo_path: str) -> list:
    """
    Detect all pull request template file paths (handles cases where multiple paths exist)
    Also detects files that existed in the past from Git history
    """
    possible_paths = [
        ".github/PULL_REQUEST_TEMPLATE.md",  
        "PULL_REQUEST_TEMPLATE.md",
        "docs/PULL_REQUEST_TEMPLATE.md",
        ".github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md",
    ]
    
    found_paths = []
    
    # Get all template file paths that existed in the past from Git history
    try:
        # Search for template-related files from all history
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Collect file paths found in history
            all_files_in_history = set(result.stdout.strip().split('\n'))
            
            # Search for files matching template file patterns
            for file_path in all_files_in_history:
                if not file_path:
                    continue
                
                # Check files related to PULL_REQUEST_TEMPLATE
                file_lower = file_path.lower()
                if 'pull_request_template' in file_lower and (file_path.endswith('.md') or file_path.endswith('.txt')):
                    # Convert to relative path (if needed)
                    if file_path not in found_paths:
                        found_paths.append(file_path)
                        print(f"Found PR template file in history: {file_path}")
    except Exception as e:
        print(f"Error searching Git history for template files: {e}")
    
    # Also check currently existing files
    for path in possible_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            # Scan directory to get actual filename
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
                # Fallback: return original path
                if path not in found_paths:
                    found_paths.append(path)
                    print(f"Found PR template file (current): {path}")
    
    # Also search for files in .github/PULL_REQUEST_TEMPLATE/ directory from Git history
    try:
        # Check if directory existed in the past
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
                # .md or .txt files in PULL_REQUEST_TEMPLATE directory
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
    Detect issue template file paths (supports both Markdown and YAML)
    Also detects files that existed in the past from Git history
    """
    found_templates = []
    found_paths_set = set()  # For duplicate checking
    
    # Get all issue template file paths that existed in the past from Git history
    try:
        # Search for template-related files from all history
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--name-only", "--pretty=format:", "--"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Collect file paths found in history
            all_files_in_history = set(result.stdout.strip().split('\n'))
            
            # Search for files matching template file patterns
            for file_path in all_files_in_history:
                if not file_path:
                    continue
                
                file_lower = file_path.lower()
                
                # Check files related to Issue template
                if 'issue_template' in file_lower:
                    # YAML files
                    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        if file_path not in found_paths_set:
                            found_paths_set.add(file_path)
                            found_templates.append({"path": file_path, "type": "yaml"})
                            print(f"Found YAML issue template in history: {file_path}")
                    # Markdown files
                    elif file_path.endswith('.md') or file_path.endswith('.txt'):
                        if file_path not in found_paths_set:
                            found_paths_set.add(file_path)
                            found_templates.append({"path": file_path, "type": "markdown"})
                            print(f"Found markdown issue template in history: {file_path}")
    except Exception as e:
        print(f"Error searching Git history for issue template files: {e}")
    
    # Also search for files in .github/ISSUE_TEMPLATE/ directory from Git history
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
                # Files in .github/ISSUE_TEMPLATE/ directory
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
    
    # Search for currently existing Markdown templates
    markdown_paths = [
        ".github/ISSUE_TEMPLATE.md",
        "ISSUE_TEMPLATE.md", 
        "docs/ISSUE_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/ISSUE_TEMPLATE.md",
    ]
    
    for path in markdown_paths:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            # Scan directory to get actual filename
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
    
    # Search for currently existing YAML templates
    yaml_template_dir = os.path.join(repo_path, ".github/ISSUE_TEMPLATE")
    if os.path.exists(yaml_template_dir):
        yaml_files = glob.glob(os.path.join(yaml_template_dir, "*.yaml")) + glob.glob(os.path.join(yaml_template_dir, "*.yml"))
        for yaml_file in yaml_files:
            # Convert to relative path (for Git commands)
            relative_path = os.path.relpath(yaml_file, repo_path)
            if relative_path not in found_paths_set:
                found_paths_set.add(relative_path)
                found_templates.append({"path": relative_path, "type": "yaml"})
                print(f"Found YAML issue template (current): {relative_path}")
        
        # Search for Markdown templates (in .github/ISSUE_TEMPLATE/ directory)
        md_files = glob.glob(os.path.join(yaml_template_dir, "*.md"))
        for md_file in md_files:
            # Convert to relative path (for Git commands)
            relative_path = os.path.relpath(md_file, repo_path)
            if relative_path not in found_paths_set:
                found_paths_set.add(relative_path)
                found_templates.append({"path": relative_path, "type": "markdown"})
                print(f"Found Markdown issue template (current): {relative_path}")
    
    return found_templates


def parse_yaml_issue_template(yaml_path: str) -> str:
    """
    Extract only important items from YAML format issue template and convert to Markdown format
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)
        
        if not template_data:
            return ""
        
        # Extract basic information
        name = template_data.get('name', '')
        description = template_data.get('description', '')
        title = template_data.get('title', '')
        
        # Extract important items from body section
        body_content = ""
        if 'body' in template_data and isinstance(template_data['body'], list):
            for item in template_data['body']:
                if item.get('type') == 'markdown':
                    # Add markdown content as-is
                    markdown_content = item.get('attributes', {}).get('value', '')
                    if markdown_content:
                        body_content += f"{markdown_content}\n\n"
                
                elif item.get('type') == 'textarea':
                    # Extract textarea field information
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
                    # Extract input field information
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
                    # Extract dropdown field information
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
                    # Extract checkboxes field information
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
        
        # Build final Markdown template
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
    Get existing template content and return as hash map
    """
    existing_templates = {}
    try:
        if data_getter.has_issue_templates(repository_id):
            templates = data_getter.issue_templates(repository_id)
            for template in templates:
                # Use template content hash as key
                content_hash = hashlib.md5(template.template.encode('utf-8')).hexdigest()
                existing_templates[content_hash] = template
    except Exception as e:
        print(f"Error getting existing templates: {e}")
    
    return existing_templates


def template_content_changed(new_content: str, existing_templates: dict) -> bool:
    """
    Check if new template content differs from existing ones
    """
    if not new_content:
        return False
    
    new_content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
    return new_content_hash not in existing_templates


def get_yaml_template_content_at_commit(repo_path: str, commit_hash: str, template_path: str) -> str:
    """
    Get YAML template content at a specific commit and convert to Markdown format
    """
    try:
        # First, check if file exists at that commit point
        check_result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_hash}:{template_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if check_result.returncode != 0:
            # Return empty string if file doesn't exist (not an error)
            return ""
        
        # Get file content at specific commit point
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
        
        # Save YAML content to temporary file and parse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name
        
        try:
            # Convert YAML template to Markdown format
            markdown_content = parse_yaml_issue_template(temp_file_path)
            return markdown_content
        finally:
            # Delete temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Error getting YAML template content for commit {commit_hash[:8]}: {e}")
        return ""


def get_template_commits(repo_path: str, template_path: str) -> list:
    """
    Get change history of pull request template file
    """
    # Get template file history with git log --follow
    result = subprocess.run(
        ["git", "log", "--follow", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso", "--", template_path],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        # If file doesn't exist or error occurred
        if result.returncode == 128:  # git log error (file doesn't exist, etc.)
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
    Get template content at a specific commit point
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
    # Clone or update repository
    repo_path = clone_or_update_repository(repo)
    if not repo_path:
        return
    
    try:
        # Detect all template file paths (handles cases where multiple paths exist)
        template_paths = check_all_pull_request_template_paths(repo_path)
        if not template_paths:
            print(f"No PR template file found for {repo.owner}.{repo.name}")
            return
        
        # Process each template path
        for template_path in template_paths:
            print(f"Processing PR template: {template_path}")
            
            # Get template change history
            commits = get_template_commits(repo_path, template_path)
            
            if not commits:
                print(f"No PR template history found for {template_path}")
                continue
                    
            # Get and save template content for each commit
            for i, commit in tqdm(enumerate(commits), desc=f"Processing PR templates: {template_path}"):
                
                # Get template content
                template_content = get_template_content_at_commit(repo_path, commit['hash'], template_path)
                
                if template_content:                
                    # Save to database
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
    # Clone or update repository
    repo_path = clone_or_update_repository(repo)
    if not repo_path:
        return
    
    try:
        # Detect template file paths (supports both Markdown and YAML)
        template_files = check_issue_template_exists(repo_path)
        if not template_files:
            print(f"No issue template files found for {repo.owner}.{repo.name}")
            return
        
        # Process each template file
        for template_info in template_files:
            template_path = template_info['path']
            template_type = template_info['type']
            
            print(f"Processing {template_type} template: {template_path}")
            
            # Get template change history
            commits = get_template_commits(repo_path, template_path)
            
            if not commits:
                print(f"No issue template history found for {template_path}")
                continue
                    
            # Get and save template content for each commit
            for i, commit in tqdm(enumerate(commits), desc=f"Processing {template_type} issue templates"):
                
                # Get template content
                if template_type == 'yaml':
                    # Special processing for YAML templates
                    template_content = get_yaml_template_content_at_commit(repo_path, commit['hash'], template_path)
                else:
                    # Standard processing for Markdown templates
                    template_content = get_template_content_at_commit(repo_path, commit['hash'], template_path)
                
                if template_content:                
                    # Save to database (diff check disabled)
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
            # Fetch pull request templates
            print(f"Fetching PR templates for {repo.owner}.{repo.name}...")
            fetch_pull_request_templates(repo)
            
            # Fetch issue templates
            print(f"Fetching issue templates for {repo.owner}.{repo.name}...")
            fetch_issue_templates(repo)
            
        except Exception as e:
            print(f"Error processing repository {repo.owner}.{repo.name}: {e}", file=sys.stderr)
            continue
    
    print("Done!")
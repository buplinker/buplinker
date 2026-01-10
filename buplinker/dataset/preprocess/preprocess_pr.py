import os
import sys
import re
import pickle
import time

from sqlalchemy.types import DateTime
from tqdm import tqdm

# Optional torch import for CUDA memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd

import data_fetch.database.get as data_getter
from buplinker.dataset.preprocess.template_extractor import TemplateExtractor
from data_fetch.database.get import repository_by_id
from data_fetch.database.tables import Repository
from data_fetch.database.set import reset_database_connection
from root_util import ContentType

# Global cache: repository_id -> {pr_id -> preprocessed description}
_preprocessing_cache = {}

# Template/extractor cache (managed per repository_id)
_template_context_cache = {}

# TemplateExtractor cache (managed per repository_id for memory efficiency)
_template_extractor_cache = {}

# Last save time of persistent cache (for rate limiting)
_last_persist_time = {}

# Cache directory configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_log_func(logger=None, error=False):
    if logger:
        return logger.error if error else logger.info
    else:
        return (lambda msg: print(msg, file=sys.stderr)) if error else print

def _get_cache_path(repo: Repository):
    """Generate cache file path"""
    return os.path.join(CACHE_DIR, f"{repo.id}_{repo.owner}.{repo.name}.pkl")

def _load_description_without_template_cache(repo: Repository, logger=None):
    """Load description without template cache per repository"""
    cache_path = _get_cache_path(repo)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            get_log_func(logger=logger, error=True)(f"Error loading description without template cache for repo {repo.owner}.{repo.name}: {e}")
    return None

def _save_description_without_template_cache(repository_id: int, ctx: dict, logger=None, force=False):
    """Save description without template cache per repository"""
    global _last_persist_time
    
    # Rate limiting: skip if less than 600 seconds since last save (except when force=True)
    if not force:
        last_time = _last_persist_time.get(repository_id, 0)
        current_time = time.time()
        if current_time - last_time < 600:
            return
    
    get_log_func(logger=logger)(f"Saving description without template cache for repository {repository_id}")

    try:
        repo = repository_by_id(repository_id)
        if repo:
            cache_path = _get_cache_path(repo)
            with open(cache_path, 'wb') as f:
                pickle.dump(ctx, f)
            _last_persist_time[repository_id] = time.time()
    except Exception as e:
        get_log_func(logger=logger, error=True)(f"Error saving description without template cache for repository {repository_id}: {e}")

def clear_description_without_template_cache(repo: Repository, logger=None):
    """Clear description without template cache (only for specified repository if repo is provided)"""
    global _template_context_cache
    if repo is not None:
        # Clear cache for specific repository
        cache_path = _get_cache_path(repo)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            get_log_func(logger=logger)(f"Cleared description without template cache for repo {repo.owner}.{repo.name}")
        if repo.id in _template_context_cache:
            del _template_context_cache[repo.id]

def clear_preprocessing_cache(logger=None):
    """Manually clear memory cache"""
    global _preprocessing_cache
    _preprocessing_cache.clear()
    get_log_func(logger=logger)(f"Preprocessing cache cleared")

def clear_preprocessing_cache_for_repo(repository_id: int, logger=None):
    """Clear memory cache for specific repository"""
    global _preprocessing_cache
    if repository_id in _preprocessing_cache:
        del _preprocessing_cache[repository_id]
        get_log_func(logger=logger)(f"Preprocessing cache cleared for repository {repository_id}")

def preprocess_pr_data(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    preprocessed_df = df.copy()

    preprocessed_df = filter_bot_prs(preprocessed_df)
    if preprocessed_df.empty:
        return preprocessed_df
        
    repository_id = int(preprocessed_df.iloc[0]['repository_id'])
    
    # Load persistent cache
    persistent_cache = _get_template_context(repository_id).get("pr_description_without_template", {})
    
    # Get memory cache per repository_id
    if repository_id not in _preprocessing_cache:
        _preprocessing_cache[repository_id] = {}
    memory_cache = _preprocessing_cache[repository_id]
    
    # Create list of unique PR IDs (preserving order)
    unique_pr_ids = preprocessed_df['pr_id'].astype(str).unique()
    
    # Map data per unique PR
    pr_id_to_row_idx = {}
    pr_id_to_data = {}
    for idx, row in preprocessed_df.iterrows():
        pr_id = str(row['pr_id'])
        if pr_id not in pr_id_to_row_idx:
            pr_id_to_row_idx[pr_id] = idx
            pr_id_to_data[pr_id] = {
                'description': str(row['description']) if pd.notna(row['description']) else '',
                'description_html': str(row['description_html']) if pd.notna(row['description_html']) else '',
                'pull_request_date': row['pull_request_date'],
                'repository_id': str(row['repository_id']),
            }
    
    # Perform cache check
    cache_results = {}
    cache_miss_pr_ids = []
    cache_updated = False
    
    for pr_id in unique_pr_ids:
        if pr_id in memory_cache:
            # Get from memory cache and also save to persistent cache
            enhanced_desc = memory_cache[pr_id]
            if pr_id not in persistent_cache:
                persistent_cache[pr_id] = enhanced_desc
            cache_results[pr_id] = enhanced_desc
        elif pr_id in persistent_cache:
            # Get from persistent cache and also save to memory cache
            enhanced_desc = persistent_cache[pr_id]
            memory_cache[pr_id] = enhanced_desc
            cache_results[pr_id] = enhanced_desc
        else:
            cache_results[pr_id] = None
            cache_miss_pr_ids.append(pr_id)
    
    # Display cache usage status
    get_log_func(logger)(f"  Cache status: {len(cache_miss_pr_ids)}/{len(unique_pr_ids)} unique PRs need processing")
    
    # Process only cache misses
    if cache_miss_pr_ids:
        get_log_func(logger)(f"  Processing {len(cache_miss_pr_ids)} PRs not in cache...")
        
        # Extract templates in batch
        batch_data = [pr_id_to_data[pr_id] for pr_id in cache_miss_pr_ids]
        enhanced_descriptions_batch = _process_descriptions_batch(
            [d['description'] for d in batch_data],
            [d['description_html'] for d in batch_data],
            [d['pull_request_date'] for d in batch_data],
            [d['repository_id'] for d in batch_data],
            cache_miss_pr_ids,
            logger=logger,
        )
        
        # Save results to both caches
        for pr_id, enhanced_desc in zip(cache_miss_pr_ids, enhanced_descriptions_batch):
            memory_cache[pr_id] = enhanced_desc
            persistent_cache[pr_id] = enhanced_desc
            cache_results[pr_id] = enhanced_desc
            cache_updated = True
        
        # Save persistent cache (save as context maintaining structure)
        if cache_updated:
            full_context = _get_template_context(repository_id)
            _save_description_without_template_cache(repository_id, full_context, logger=logger, force=True)
            get_log_func(logger)(f"  Saved {len(cache_miss_pr_ids)} new descriptions to persistent cache")
    
    # Assign processed description to all PRs (same PR ID gets same description even if duplicated)
    preprocessed_df['description'] = preprocessed_df['pr_id'].astype(str).map(cache_results)

    # Sync from memory cache to persistent cache (for reliability)
    if cache_updated:
        full_context = _get_template_context(repository_id)
        _save_description_without_template_cache(repository_id, full_context, logger=logger, force=True)
        get_log_func(logger)(f"  Final sync: Updated persistent cache with {len(persistent_cache)} descriptions")

    # Keep memory cache (for reuse within same process)
    # Use clear_preprocessing_cache() if manual clearing is needed
    return preprocessed_df


def _process_descriptions_batch(descriptions, description_htmls, pull_request_dates, repository_ids, pr_ids, logger=None):
    """Preprocess descriptions in batch"""
    enhanced_descriptions = []
    
    for i in tqdm(range(len(descriptions)), desc="Processing descriptions batch"):
        try:
            # Ensure type conversion
            pr_id = pr_ids[i]
            repository_id = repository_ids[i]
            
            description_without_template = extract_template_content_for_single_text(
                str(descriptions[i]),
                str(description_htmls[i]),
                pull_request_dates[i],
                ContentType.PR,
                repository_id,
                pr_id,
                logger=logger,
            )
            
            description_without_template = description_without_template or ""
            
            enhanced_desc = resolve_description_links(
                repository_id,
                description_without_template,
                str(description_htmls[i]),
                processed_refs=set([f"pull_request_#{pr_id}"]),
                logger=logger,
            )
            
            # Final cleanup (apply again as a precaution)
            enhanced_desc = _clean_description_text(enhanced_desc)
            
            enhanced_descriptions.append(enhanced_desc)
            
        except Exception as e:
            error_str = str(e)
            # Safely log exception message (don't pass file argument to logger method)
            try:
                log_func = get_log_func(logger=logger, error=True)
                log_func(f"Error processing PR {pr_id}: {error_str}")
            except Exception as log_error:
                # Use print directly if logger call also fails
                print(f"Error processing PR {pr_id}: {error_str}", file=sys.stderr)
                print(f"Logger error: {log_error}", file=sys.stderr)
            
            # Clear memory and exit if CUDA out of memory error
            if "CUDA" in error_str and "out of memory" in error_str:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        try:
                            log_func = get_log_func(logger=logger, error=True)
                            log_func("Cleared CUDA cache before exit")
                        except:
                            print("Cleared CUDA cache before exit", file=sys.stderr)
                    except Exception as cache_error:
                        try:
                            log_func = get_log_func(logger=logger, error=True)
                            log_func(f"Failed to clear CUDA cache: {cache_error}")
                        except:
                            print(f"Failed to clear CUDA cache: {cache_error}", file=sys.stderr)
                        sys.exit(1)
            
            sys.exit(1)
    
    return enhanced_descriptions

def filter_bot_prs(df_pull_request: pd.DataFrame, logger=None) -> pd.DataFrame:
    if df_pull_request.empty:
        return df_pull_request
    
    try:
        base_path = os.path.dirname(__file__)
        bot_csv_path = os.path.join(base_path, 'groundtruthbots.csv')
        bot_df = pd.read_csv(bot_csv_path)
        bot_df = bot_df[bot_df['type'] == 'Bot']
        
        # Create set of bot accounts (for fast lookup)
        bot_accounts = set(bot_df['account'].tolist())
        
        # Vectorized bot detection
        def is_bot_author_vectorized(author):
            if pd.isna(author) or author is None:
                return False
            
            author_str = str(author).lower()
            
            # 1. If author ends with 'bot'
            if author_str.endswith('bot') or author_str.endswith('[bot]'):
                return True
            
            # 2. If account matches in groundtruthbots.csv
            return author in bot_accounts
        
        # Exclude bot PRs (vectorized)
        original_count = len(df_pull_request)
        df_filtered = df_pull_request[~df_pull_request['author'].apply(is_bot_author_vectorized)]
        filtered_count = len(df_filtered)
        get_log_func(logger=logger)(f"  Filtered out {original_count - filtered_count} bot PRs from {original_count} total PRs")
        return df_filtered

    except Exception as e:
        get_log_func(logger=logger, error=True)(f"  Error filtering bot PRs: {e}")
        return df_pull_request

def _get_template_context(repository_id: int):
    """Lazily initialize and return template/extractor context (managed per repository_id)."""
    global _template_context_cache
    
    # Check memory cache (managed per repository_id)
    if repository_id in _template_context_cache:
        return _template_context_cache[repository_id]
    
    # Check persistent cache
    repo = repository_by_id(repository_id)
    persistent_template_cache = _load_description_without_template_cache(repo)
    if persistent_template_cache is not None:
        _template_context_cache[repository_id] = {
            "pr_description_without_template": persistent_template_cache.get("pr_description_without_template", {}),
            "issue_description_without_template": persistent_template_cache.get("issue_description_without_template", {}),
        }
        return _template_context_cache[repository_id]

    # Return empty cache if no cache exists
    _template_context_cache[repository_id] = {
        "pr_description_without_template": {},
        "issue_description_without_template": {},
    }
    return _template_context_cache[repository_id]

def extract_template_content_for_single_text(description: str, description_html: str, created_at: DateTime, content_type: ContentType, repository_id: int, pr_or_issue_id: str, logger=None, max_retries: int = 2) -> str:
    """
    Execute template extraction (with retry functionality)
    
    Args:
        max_retries: Maximum number of retries (default: 2 = initial attempt + 1 retry)
    """
    # Helper function to safely log
    def safe_log(msg):
        try:
            log_func = get_log_func(logger=logger, error=True)
            log_func(msg)
        except Exception as log_error:
            print(msg, file=sys.stderr)
            print(f"Logger error: {log_error}", file=sys.stderr)
    
    last_exception = None
    use_cpu_mode = False  # GPU priority, switch to CPU only on OOM
    
    for attempt in range(max_retries + 1):  # 0, 1, 2... (max_retries + 1 attempts)
        try:
            created_at = pd.to_datetime(created_at)
            ctx = _get_template_context(repository_id)
            
            cache_key = f"{content_type.value}_description_without_template"
            
            extraction_cache = ctx.get(cache_key, {})
            if pr_or_issue_id in extraction_cache:
                return extraction_cache[pr_or_issue_id]
            
            # Execute template removal using TemplateExtractor
            # GPU priority, switch to CPU only on OOM
            # Cache per repository_id to improve memory efficiency
            global _template_extractor_cache
            device = 'cpu' if use_cpu_mode else None  # None for auto-select (GPU if CUDA available, otherwise CPU)
            cache_key_extractor = f"{repository_id}_{device}"
            if cache_key_extractor not in _template_extractor_cache:
                _template_extractor_cache[cache_key_extractor] = TemplateExtractor(device=device)
            extractor = _template_extractor_cache[cache_key_extractor]
            
            # Get Repository object
            repo = repository_by_id(repository_id)
            
            # Execute template removal (use HTML content if provided)
            extracted_result = extractor.extract_unique_content(
                repo,
                description,
                description_html,
                created_at,
                content_type,
            )
            # Get extraction result (use original text if extraction fails)
            extracted_text = extracted_result.get("extracted", description)
            
            # Add result to cache
            extraction_cache[pr_or_issue_id] = extracted_text
            ctx[cache_key] = extraction_cache
            
            # Also save to persistent cache
            _save_description_without_template_cache(repository_id, ctx, logger=logger)

            return extracted_text

        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            # If not final attempt, perform appropriate handling and retry
            if attempt < max_retries:
                # Clear memory if CUDA out of memory error
                if "CUDA" in error_str and "out of memory" in error_str:
                    safe_log(f"CUDA out of memory error (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            safe_log(f"Cleared CUDA cache")
                        except Exception as cache_error:
                            safe_log(f"Failed to clear CUDA cache: {cache_error}")
                    
                    # Switch to CPU mode and retry
                    if not use_cpu_mode:
                        use_cpu_mode = True
                        safe_log("Switching to CPU mode due to CUDA out of memory error...")
                        # Clear existing GPU extractor to free memory
                        old_cache_key = f"{repository_id}_{None}"
                        if old_cache_key in _template_extractor_cache:
                            del _template_extractor_cache[old_cache_key]
                    else:
                        safe_log("Retrying in CPU mode...")
                    
                    # Continue loop for retry
                    continue
                # Reset connection if database connection error
                elif "Can't reconnect until invalid transaction is rolled back" in error_str:
                    safe_log(f"Database connection error (attempt {attempt + 1}/{max_retries + 1}), resetting connection: {error_str}")
                    reset_database_connection()
                    safe_log("Database connection reset, retrying...")
                    # Continue loop for retry
                    continue
                else:
                    safe_log(f"Error extracting template content (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                    safe_log("Retrying...")
                    # Continue loop for retry
                    continue
            else:
                # If failed on final attempt
                safe_log(f"Failed to extract template content after {max_retries + 1} attempts")
                
                # Clear memory and exit if CUDA out of memory error
                if "CUDA" in error_str and "out of memory" in error_str:
                    safe_log(f"CUDA out of memory error (final attempt): {error_str}")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            safe_log("Cleared CUDA cache before exit")
                        except Exception as cache_error:
                            safe_log(f"Failed to clear CUDA cache: {cache_error}")
                # Reset connection and exit if database connection error
                elif "Can't reconnect until invalid transaction is rolled back" in error_str:
                    safe_log(f"Database connection error (final attempt): {error_str}")
                    reset_database_connection()
                    safe_log("Database connection reset before exit")
                else:
                    safe_log(f"Error extracting template content (final attempt): {error_str}")
                
                # Exit if all attempts failed
                sys.exit(1)
    
    # Should not reach here, but just in case
    if last_exception:
        safe_log(f"Unexpected error: {last_exception}")
        sys.exit(1)
    raise RuntimeError("extract_template_content_for_single_text: Unexpected state")


def _clean_description_text(text: str) -> str:
    """
    Remove unnecessary text (URLs) from description
    """
    if not text:
        return ""
    
    # Remove URLs (http://, https://, ftp://, www., etc.)
    # Remove all URLs (including links to GitHub issues/PRs)
    url_patterns = [
        r'https?://[^\s]+',  # http:// or https://
        r'ftp://[^\s]+',     # ftp://
        r'www\.[^\s]+',      # URLs starting with www.
    ]
    
    for pattern in url_patterns:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            text = text[:match.start()] + text[match.end():]
    
    # Normalize consecutive spaces to one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def resolve_description_links(repository_id: int, description: str, description_html: str, processed_refs: set = set(), logger=None) -> str:
    
    enhanced_description = description
    
    # Regex pattern to find href patterns
    data_url_pattern = r'href="([^"]*)"'
    
    # Track processed URLs within one PR or issue
    processed_urls = set()
    
    # Get all hrefs at once (including position information)
    matches = list(re.finditer(data_url_pattern, description_html))
    
    # Process from back to front to prevent string position shifts
    for match in reversed(matches):
        url = match.group(1)
        
        # Skip already processed URLs
        if url in processed_urls:
            continue
            
        processed_urls.add(url)
        
        # Check issue/number pattern
        issue_match = re.search(r'issues/(\d+)', url)
        if issue_match:
            issue_number = int(issue_match.group(1))
            issue_info = _process_issue_reference(
                repository_id, issue_number, processed_refs, logger=logger
            )
            enhanced_description += issue_info
        
        # Check pull/number pattern
        pull_match = re.search(r'pull/(\d+)', url)
        if pull_match:
            pull_number = int(pull_match.group(1))
            pr_info = _process_pull_request_reference(
                repository_id, pull_number, processed_refs, logger=logger
            )
            enhanced_description += pr_info
    
    # Apply cleanup processing
    enhanced_description = _clean_description_text(enhanced_description)
    
    return enhanced_description


def _process_issue_reference(repository_id: int, issue_number: int, processed_refs: set, logger=None) -> str:
    issue_ref = f"issue_#{issue_number}"
    
    # Check for circular reference
    if issue_ref in processed_refs:
        return ""
        
    try:
        issue = data_getter.issue_by_number(repository_id, issue_number)
        if issue:
            # Add to processed references
            processed_refs.add(issue_ref)
            
            # First apply template removal
            description_without_template = extract_template_content_for_single_text(
                issue.bodyText,
                issue.bodyHtml,
                issue.created_at,
                ContentType.ISSUE,
                repository_id,
                issue.id,
                logger=logger,
            )
                        
            # Execute link resolution on bodyHtml after template removal
            # Use bodyText after template removal for link resolution
            processed_issue_body = resolve_description_links(
                repository_id, 
                description_without_template, 
                issue.bodyHtml,  # Use original bodyHtml (contains href)
                processed_refs.copy(),  # Pass copy to prevent circular references
                logger=logger,
            )
                        
            return f" {issue_ref}: {issue.title} {processed_issue_body}"
    except Exception as e:
        get_log_func(logger=logger, error=True)(f"Error fetching issue {issue_number}: {e}")
    
    return ""

def _process_pull_request_reference(repository_id: int, pull_number: int, processed_refs: set, logger=None) -> str:
    pr_ref = f"pull_request_#{pull_number}"
    
    # Check for circular reference
    if pr_ref in processed_refs:
        return ""
        
    try:
        pull_request = data_getter.pull_request_by_number(repository_id, pull_number)
        if pull_request:
            # Add to processed references
            processed_refs.add(pr_ref)
            
            # First apply template removal
            description_without_template = extract_template_content_for_single_text(
                pull_request.bodyText,
                pull_request.bodyHtml,
                pull_request.created_at,
                ContentType.PR,
                repository_id,
                pull_request.id,  # Pass PR ID
                logger=logger,
            )
            
            description_without_template = description_without_template or ""
            
            # Execute link resolution on bodyHtml after template removal
            # Use bodyText after template removal for link resolution
            processed_pr_body = resolve_description_links(
                repository_id, 
                description_without_template, 
                pull_request.bodyHtml,  # Use original bodyHtml (contains href)
                processed_refs.copy(),  # Pass copy to prevent circular references
                logger=logger,
            )
            
            processed_pr_body = processed_pr_body or ""
            
            return f" {pr_ref}: {pull_request.title} {processed_pr_body}"
    except Exception as e:
        get_log_func(logger=logger, error=True)(f"Error fetching pull request {pull_number}: {e}")
    
    return ""

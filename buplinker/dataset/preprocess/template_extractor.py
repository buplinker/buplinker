#!python3
"""
Pull request template extraction module
A staged extraction approach considering cost efficiency
"""

import os
import sys
import json
import re
from typing import Dict, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import project_config as config
from data_fetch.database.get import pull_request_templates, issue_templates
from root_util import target_repos, ContentType
from data_fetch.database.tables import Repository

load_dotenv()


class TemplateExtractor:
    def __init__(self, device=None):
        """
        Args:
            device: Specify 'cpu' or 'cuda'. If None, automatically selects (CUDA if available, otherwise CPU)
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Auto-select if device is not specified
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        self.device = device
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        # Embedding cache per template
        self._template_embedding_cache = {}
    
    def analyze_template_structure(self, template_content: str) -> tuple[dict, int, int]:        
        prompt_tokens = 0
        completion_tokens = 0
        
        # TODO: change the prompt
        system_message = self._load_prompt("template_system_message.txt")
        user_message = self._load_prompt("template_user_message.txt").format(
            template=template_content
        )

        response = self.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=config.OPENAI_TEMPERATURE,
        )

        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)

        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        
        analysis = json.loads(result_text)   
        if "sections" in analysis:
            for section in analysis["sections"]:
                if "name" in section:
                    original_name = section["name"]
                    normalized_name = self._normalize_section_name(original_name)
                    section["name"] = normalized_name
                
                # Normalize header_pattern as well
                if "header_pattern" in section:
                    original_pattern = section["header_pattern"]
                    # Remove parentheses from header_pattern as well
                    normalized_pattern = re.sub(r'\s*\([^)]*\)', '', original_pattern)
                    section["header_pattern"] = normalized_pattern
        return analysis, prompt_tokens, completion_tokens
    
    def calculate_template_similarity(self, pr_description: str, template_content: str) -> float:
        try:
            # Pre-check for NaN and non-string values
            if pd.isna(pr_description) or pd.isna(template_content):
                return 0.0
            
            pr_text = self._preprocess_text(pr_description)
            template_text = self._preprocess_text(template_content)
            
            if not pr_text or not template_text:
                return 0.0
            
            pr_embedding = self.similarity_model.encode([pr_text])
            template_embedding = self.similarity_model.encode([template_text])
            
            similarity = cosine_similarity(pr_embedding, template_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}", file=sys.stderr)
            return 0.0
    
    def extract_unique_content(self, repo: Repository, description: str, description_html: str, created_at: datetime, template_type: ContentType) -> Dict:
        result = {
            "original": description,
            "extracted": description,
        }
        
        try:
            # Attempt HTML-based template removal if HTML is provided
            html_extracted = self.extract_from_html(repo, description, description_html, created_at, template_type)
            result["extracted"] = html_extracted
            return result
            
        except Exception as e:
            print(f"Error extracting unique content: {e}", file=sys.stderr)
            return result
    
    def get_closest_template_analysis(self, repo: Repository, description: str, created_at: datetime, template_type: ContentType):
        """Get the most recent template analysis result before the specified datetime"""
        output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
        done_marker = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_TEMPLATE_TITLES_DONE")

        if not os.path.exists(done_marker):
            print(f"PR template analysis marker not found for {repo.owner}.{repo.name}. "
                  f"Expected marker: {done_marker}. Please run template extraction first.")
            raise SystemExit(1)


        if template_type == ContentType.PR:
            return self._get_pr_template_analysis(repo, created_at)
        else:
            return self._get_issue_template_analysis(repo, description, created_at)
    
    def _get_pr_template_analysis(self, repo: Repository, created_at: datetime):
        """Get PR template analysis result"""
        json_file = os.path.join(os.path.dirname(__file__), "template_title_repositories", 
                                f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.PR.value}_template_titles.json")

        return self._load_template_analysis_from_file(json_file, created_at)
    
    def _get_issue_template_analysis(self, repo: Repository, description: str, created_at: datetime):
        """Get Issue template analysis result (select optimal template based on similarity)"""
        try:
            # Get available Issue template analysis results
            available_templates = self._get_available_issue_templates(repo, created_at)
            if not available_templates:
                return None
            
            # Calculate similarity with each template
            template_similarities = {}
            # Key for template embedding cache
            cache_key_prefix = f"{repo.id}_{repo.owner}_{repo.name}"

            for template_name, template_data in available_templates.items():
                # Get template content
                template_content = self._extract_template_content(template_data)
                if template_content:
                    # Cache key for template embedding
                    template_cache_key = f"{cache_key_prefix}_{template_name}"
                    
                    # Get or compute template embedding from cache
                    if template_cache_key in self._template_embedding_cache:
                        template_embedding = self._template_embedding_cache[template_cache_key]
                    else:
                        template_text = self._preprocess_text(template_content)
                        template_embedding = self.similarity_model.encode([template_text])
                        self._template_embedding_cache[template_cache_key] = template_embedding
                    
                    # Compute Issue embedding
                    issue_text = self._preprocess_text(description)
                    issue_embedding = self.similarity_model.encode([issue_text])
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(issue_embedding, template_embedding)[0][0]
                    template_similarities[template_name] = float(similarity)
            
            # Select template with highest similarity
            if template_similarities:
                best_template = max(template_similarities, key=template_similarities.get)
                return available_templates[best_template]
            
        except Exception as e:
            print(f"Error getting issue template analysis: {e}")
    
    def _get_available_issue_templates(self, repo: Repository, created_at: datetime) -> dict:
        """Get available Issue template analysis results"""
        try:
            output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
            pattern = f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.ISSUE.value}_*_template_titles.json"
            matching_files = glob.glob(os.path.join(output_dir, pattern))
            
            available_templates = {}
            for file_path in matching_files:
                # Extract template name from filename
                filename = os.path.basename(file_path)
                template_name = filename.replace(f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.ISSUE.value}_", "").replace("_template_titles.json", "")
                
                # Load template analysis result
                template_data = self._load_template_analysis_from_file(file_path, created_at)
                if template_data:
                    available_templates[template_name] = template_data
            
            return available_templates
            
        except Exception as e:
            print(f"Error getting available issue templates: {e}")
            return {}
    
    def _extract_template_content(self, template_data: dict) -> str:
        """Extract template content from template analysis result"""
        try:
            sections = template_data.get("analysis", {}).get("sections", [])
            content_parts = []
            
            for section in sections:
                if section.get("name"):
                    content_parts.append(section["name"])
            
            return " ".join(content_parts)
            
        except Exception as e:
            print(f"Error extracting template content: {e}")
            return ""
    
    def _normalize_section_name(self, section_name: str) -> str:
        """Normalize section name (remove annotations and parentheses)"""
        if not section_name:
            return ""
        
        # Remove annotations in parentheses
        # Remove (required), (optional), (recommended), etc.
        section_name = re.sub(r'\s*\([^)]*\)', '', section_name)
        
        # Remove other common annotations
        annotations = [
            'required', 'optional', 'recommended', 'mandatory',
            'needed', 'important', 'critical', 'essential'
        ]
        
        for annotation in annotations:
            # Match at word boundaries
            pattern = r'\b' + re.escape(annotation) + r'\b'
            section_name = re.sub(pattern, '', section_name, flags=re.IGNORECASE)
        
        # Normalize whitespace
        section_name = re.sub(r'\s+', ' ', section_name).strip()
        
        # Remove trailing colons and periods
        section_name = re.sub(r'[:.]$', '', section_name)
        
        return section_name
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        if not text:
            return ""
        # Convert line breaks to spaces and normalize consecutive spaces
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_template_boilerplate(self, text: str) -> str:
        """Remove template boilerplate"""
        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Remove empty checkboxes
        text = re.sub(r'-\s*\[\s*\]\s*[^\n]*', '', text)
        # Remove placeholders
        text = re.sub(r'<.*?>', '', text)
        # Normalize consecutive whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def extract_from_html(self, repo: Repository, description: str, html_content: str, created_at: datetime, template_type: ContentType) -> Optional[str]:
        """Extract only important content from HTML-formatted template"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get template analysis result (used to remove section names)
            template_analysis_data = self.get_closest_template_analysis(repo, description, created_at, template_type)
            
            # Extract only important sections
            important_sections = self._extract_important_sections(repo, description, soup, created_at, template_type)
            
            # Extract content before headings (e.g., first paragraph)
            pre_heading_content = self._extract_pre_heading_content(soup)
            
            # If template analysis result is missing or important sections are empty, extract first paragraph
            if not important_sections:
                # If there are no headings, extract paragraphs excluding unimportant sections
                if template_analysis_data and "analysis" in template_analysis_data:
                    # Exclude unimportant sections from all paragraphs
                    text = self._extract_content_excluding_unimportant_sections(soup, template_analysis_data["analysis"])
                elif pre_heading_content:
                    # Use pre-heading content if available (when template analysis result is missing)
                    text = pre_heading_content
                else:
                    # If template analysis result is missing, extract all paragraphs
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        texts = [p.get_text().strip() for p in paragraphs]
                        text = ' '.join(texts)
                    else:
                        # If no paragraphs found, return original text as-is
                        text = soup.get_text()
                
                # Remove section names if template analysis result is available
                if template_analysis_data and "analysis" in template_analysis_data:
                    text = self._remove_template_section_names_from_text(text, template_analysis_data["analysis"])
                
                # Process line breaks appropriately
                text = re.sub(r'\n+', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            # Combine important section contents
            extracted_text = self._combine_important_content(important_sections)
            
            # Also combine pre-heading content (after removing section names)
            if pre_heading_content:
                # Exclude unimportant sections from pre-heading content before removing section names
                if template_analysis_data and "analysis" in template_analysis_data:
                    # Exclude unimportant sections from HTML before headings
                    # pre_heading_content is text, so get elements before headings directly from original soup
                    first_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if first_heading:
                        # Exclude unimportant sections from elements before first heading
                        pre_heading_elements = []
                        all_elements = soup.find_all(['p', 'ul', 'ol', 'div'])
                        for element in all_elements:
                            if element in first_heading.find_all_previous(['p', 'ul', 'ol', 'div']):
                                if not element.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                                    pre_heading_elements.append(element)
                        
                        # Create temporary soup to exclude unimportant sections
                        temp_soup = BeautifulSoup('<div></div>', 'html.parser')
                        temp_div = temp_soup.div
                        for element in pre_heading_elements:
                            temp_div.append(element.__copy__())
                        
                        pre_heading_content = self._extract_content_excluding_unimportant_sections(temp_div, template_analysis_data["analysis"])
                    else:
                        # No headings (unusual case)
                        pre_heading_soup = BeautifulSoup(pre_heading_content, 'html.parser')
                        pre_heading_content = self._extract_content_excluding_unimportant_sections(pre_heading_soup, template_analysis_data["analysis"])
                    
                    # Remove section names
                    pre_heading_content = self._remove_template_section_names_from_text(pre_heading_content, template_analysis_data["analysis"])
                
                if extracted_text:
                    extracted_text = pre_heading_content + " " + extracted_text
                else:
                    extracted_text = pre_heading_content
            
            # Check if meaningful content remains
            if extracted_text and len(extracted_text.strip()) > 10:
                # Process line breaks appropriately
                extracted_text = re.sub(r'\n+', ' ', extracted_text)
                extracted_text = re.sub(r'\s+', ' ', extracted_text)
                return extracted_text.strip()
            
            # If important sections are empty, extract first paragraph
            print("Important sections are empty. Extracting first paragraph.")
            if pre_heading_content:
                text = pre_heading_content
            else:
                first_p = soup.find('p')
                if first_p:
                    text = first_p.get_text()
                else:
                    text = soup.get_text()
            
            # Remove section names if template analysis result is available
            if template_analysis_data and "analysis" in template_analysis_data:
                text = self._remove_template_section_names_from_text(text, template_analysis_data["analysis"])
            
            # Process line breaks appropriately
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting from HTML: {e}", file=sys.stderr)
            return None
    
    def _remove_template_section_names_from_text(self, text: str, analysis: dict) -> str:
        """Remove template section names from text (keep important section names)"""
        if not text or not analysis:
            return text
        
        # Create set of important section names (these are not removed)
        important_section_names = set()
        for section in analysis.get("sections", []):
            is_meaningful = section.get("is_meaningful_content", False)
            importance = section.get("importance", "low")
            if is_meaningful and importance != "low":
                section_name = section.get("name", "")
                header_pattern = section.get("header_pattern", "")
                if section_name:
                    normalized_name = self._normalize_section_name(section_name).lower()
                    important_section_names.add(normalized_name)
                if header_pattern:
                    pattern_text = re.sub(r'\*\*|###|##|#|\*\s*', '', header_pattern).strip()
                    if pattern_text:
                        normalized_pattern = self._normalize_section_name(pattern_text).lower()
                        important_section_names.add(normalized_pattern)
        
        # Get unimportant section names and header_patterns
        sections_to_remove = []
        for section in analysis.get("sections", []):
            section_name = section.get("name", "")
            header_pattern = section.get("header_pattern", "")
            if section_name:
                sections_to_remove.append(section_name)
            if header_pattern:
                # Remove markdown markers from header_pattern
                pattern_text = re.sub(r'\*\*|###|##|#|\*\s*', '', header_pattern).strip()
                if pattern_text:
                    sections_to_remove.append(pattern_text)
        
        # Remove each section name from text (excluding important section names)
        for section_name in sections_to_remove:
            if not section_name:
                continue
            # Normalize section name (for comparison)
            normalized_section_name = self._normalize_section_name(section_name).lower()
            # Don't remove important section names
            if normalized_section_name in important_section_names:
                continue
            # Pattern starting with normalized section name (may include parentheses)
            # Also matches when section name is followed by parentheses or colon
            pattern = re.escape(normalized_section_name)
            # Remove when section name starts, followed by spaces, colon, parentheses, etc.
            text = re.sub(r'^\s*' + pattern + r'(\s*\([^)]*\))?\s*:?\s*', '', text, flags=re.IGNORECASE)
            # Also remove section names within text (handles parentheses as well)
            text = re.sub(r'\b' + pattern + r'(\s*\([^)]*\))?\s*:?\s*', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_content_excluding_unimportant_sections(self, soup: BeautifulSoup, analysis: dict) -> str:
        """Extract content excluding unimportant sections when there are no headings"""
        # Create set of unimportant section names
        unimportant_section_names = set()
        unimportant_header_patterns = set()
        
        for section in analysis.get("sections", []):
            is_meaningful = section.get("is_meaningful_content", False)
            importance = section.get("importance", "low")
            
            if not is_meaningful or importance == "low":
                section_name = section.get("name", "")
                header_pattern = section.get("header_pattern", "")
                if section_name:
                    # Normalize and add section name
                    normalized_name = self._normalize_section_name(section_name).lower()
                    unimportant_section_names.add(normalized_name)
                if header_pattern:
                    # Remove markdown markers from header_pattern
                    pattern_text = re.sub(r'\*\*|###|##|#|\*\s*', '', header_pattern).strip()
                    if pattern_text:
                        # Normalize header_pattern as well
                        normalized_pattern = self._normalize_section_name(pattern_text).lower()
                        unimportant_header_patterns.add(normalized_pattern)
        
        # Get all paragraphs
        all_paragraphs = soup.find_all('p')
        extracted_texts = []
        
        i = 0
        while i < len(all_paragraphs):
            p = all_paragraphs[i].__copy__()
            p_text = p.get_text().strip()
            
            # Check text within <strong> tags
            strong_tags = p.find_all('strong')
            
            # Check if unimportant section
            is_all_unimportant = len(strong_tags) > 0  # True only when there are headings
            has_important_section = False
            for strong in strong_tags:
                strong_text = strong.get_text().strip()
                normalized_strong_text = self._normalize_section_name(strong_text).lower()
                
                # Check if unimportant section
                is_unimportant = False
                for section_name in unimportant_section_names:
                    if (normalized_strong_text == section_name or 
                        normalized_strong_text.startswith(section_name + " ") or
                        normalized_strong_text.startswith(section_name + "(") or
                        normalized_strong_text.startswith(section_name + ":")):
                        is_unimportant = True
                        break
                
                if not is_unimportant:
                    for pattern in unimportant_header_patterns:
                        if pattern in normalized_strong_text or normalized_strong_text.startswith(pattern):
                            is_unimportant = True
                            break
                
                if not is_unimportant:
                    has_important_section = True
                    is_all_unimportant = False
                    # Add square brackets to <strong> tags of important sections (after removing annotations in parentheses)
                    strong_text = strong.get_text().strip()
                    if strong_text:
                        # Remove annotations in parentheses
                        strong_text = re.sub(r'\s*\([^)]*\)', '', strong_text).strip()
                        strong.clear()
                        strong.append(f'[{strong_text}] ')
                else:
                    # Remove <strong> tags of unimportant sections
                    strong.decompose()
            
            # Skip content of this section if all headings are unimportant
            if is_all_unimportant and strong_tags:
                # Skip until next heading (paragraph with <strong> tag)
                i += 1
                while i < len(all_paragraphs):
                    next_p = all_paragraphs[i]
                    next_strong = next_p.find('strong')
                    # Exit when next section (paragraph with <strong> tag) is found
                    if next_strong:
                        break
                    i += 1
                continue
            
            # Add text if not unimportant section or after removing unimportant sections
            if p_text:
                modified_text = p.get_text().strip()
                if modified_text and len(modified_text) > 0:
                    extracted_texts.append(modified_text)
            
            i += 1
        
        return ' '.join(extracted_texts)
    
    def _extract_important_sections(self, repo: Repository, description: str, soup: BeautifulSoup, created_at: datetime, template_type: ContentType) -> list:
        """Extract only important sections"""
        important_sections = []
        
        # Get template analysis result
        template_analysis_data = self.get_closest_template_analysis(repo, description, created_at, template_type)
        if not template_analysis_data or "analysis" not in template_analysis_data:
            # Return empty list if template analysis result is missing (return original text as-is)
            return important_sections
        
        analysis = template_analysis_data["analysis"]
        
        # Create mapping of important section names and header_patterns
        important_section_info = {}
        for section in analysis.get("sections", []):
            is_meaningful = section.get("is_meaningful_content", False)
            importance = section.get("importance", "low")
            
            if is_meaningful and importance != "low":
                section_name = section.get("name", "")
                header_pattern = section.get("header_pattern", "")
                if section_name:
                    # Normalize section name and use as key
                    normalized_name = self._normalize_section_name(section_name).lower()
                    important_section_info[normalized_name] = {
                        'name': section_name,
                        'header_pattern': header_pattern
                    }
        
        # Dynamically detect top-level heading
        top_level = self._get_top_level_heading(soup)
        
        # Check only top-level heading tags
        for heading in soup.find_all(top_level):
            section_title_raw = heading.get_text().strip()
            # Normalize heading before comparison
            section_title = self._normalize_section_name(section_title_raw).lower()
            
            # Check if important section (compare using normalized section names)
            if section_title in important_section_info:
                # Get entire section (from top-level heading to next top-level heading)
                section_element = self._get_section_element(heading, top_level)
                if section_element:
                    section_info = important_section_info[section_title]
                    important_sections.append({
                        'title': section_title,
                        'section_name': section_info['name'],
                        'header_pattern': section_info['header_pattern'],
                        'element': section_element
                    })
        return important_sections
    
    def _get_top_level_heading(self, soup: BeautifulSoup) -> str:
        """Detect top-level heading in HTML content"""
        # Check h2 through h6 in order
        for level in range(2, 7):
            headings = soup.find_all(f'h{level}')
            if headings:
                return f'h{level}'
        
        # Default to h2 if no headings found
        return 'h2'
    
    def _extract_pre_heading_content(self, soup: BeautifulSoup) -> str:
        """Extract content before first heading (e.g., paragraphs)"""
        try:
            # Get all headings (in document order)
            all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if not all_headings:
                # Extract all paragraphs if no headings found
                paragraphs = soup.find_all('p')
                if paragraphs:
                    texts = [p.get_text().strip() for p in paragraphs]
                    combined = ' '.join(texts)
                    combined = re.sub(r'\n+', ' ', combined)
                    combined = re.sub(r'\s+', ' ', combined)
                    return combined.strip()
                return ""
            
            # Get first heading
            first_heading = all_headings[0]
            
            # Extract all elements before first heading (preserving document order)
            pre_heading_texts = []
            all_elements = soup.find_all(['p', 'ul', 'ol', 'div'])
            
            for element in all_elements:
                # Extract elements before first heading
                if element in first_heading.find_all_previous(['p', 'ul', 'ol', 'div']):
                    # Exclude elements nested within headings
                    if not element.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text = element.get_text().strip()
                        if text:
                            pre_heading_texts.append(text)
            
            if pre_heading_texts:
                combined = ' '.join(pre_heading_texts)
                combined = re.sub(r'\n+', ' ', combined)
                combined = re.sub(r'\s+', ' ', combined)
                return combined.strip()
            
            return ""
            
        except Exception as e:
            print(f"Error extracting pre-heading content: {e}", file=sys.stderr)
            return ""
    
    def _combine_important_content(self, important_sections: list) -> str:
        """Combine important section contents to build text"""
        combined_text = []
        
        for section in important_sections:
            section_element = section['element']
            section_name = section.get('section_name', '')
            header_pattern = section.get('header_pattern', '')
            
            # Extract text from section content (pass section name and header_pattern)
            section_text = self._extract_section_text(section_element, section_name, header_pattern)
            
            if section_text and section_text.strip():
                # Process line breaks appropriately
                section_text = re.sub(r'\n+', ' ', section_text.strip())
                combined_text.append(section_text)
        
        return ' '.join(combined_text)
    
    def _extract_section_text(self, section_element, section_name: str = '', header_pattern: str = '') -> str:
        """Extract text from section element (keep important section names)"""
        # Keep heading elements (h1-h6) to display important section titles
        # Keep <strong> tags of section names (since they are important sections)
        # Keep content as-is
        
        # Add "[" and "]" to section title (after removing annotations in parentheses)
        for heading in section_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text().strip()
            if heading_text:
                # Remove annotations in parentheses
                heading_text = re.sub(r'\s*\([^)]*\)', '', heading_text).strip()
                # Get text and replace after adding square brackets
                heading.clear()
                heading.append(f'[{heading_text}] ')
        
        # Remove checkboxes and empty list items
        for checkbox in section_element.find_all('input', {'type': 'checkbox'}):
            checkbox.parent.decompose()
        
        # Add numbers to li elements to preserve numbered list numbers (before removing empty items)
        for ol in section_element.find_all('ol'):
            for idx, li in enumerate(ol.find_all('li'), start=1):
                li_text = li.get_text().strip()
                if li_text:
                    # Remove text nodes and replace with numbered text
                    li.clear()
                    li.append(f"{idx}. {li_text}")
        
        # Remove empty list items and template boilerplate
        for li in section_element.find_all('li'):
            li_text = li.get_text().strip()
            if not li_text or self._is_template_list_item(li_text):
                li.decompose()
        
        # Remove empty paragraphs
        for p in section_element.find_all('p'):
            if not p.get_text().strip():
                p.decompose()
        
        # Extract text (keep section names as-is since they are important sections)
        text = section_element.get_text()
        
        # Process line breaks and whitespace appropriately
        text = re.sub(r'\n\s*\n', ' ', text)  # Multiple line breaks to single space
        text = re.sub(r'\n+', ' ', text)      # Single line breaks to space
        text = re.sub(r'\s+', ' ', text)      # Consecutive spaces to single space
        
        return text.strip()
    
    
    def _get_section_element(self, heading, top_level: str) -> Optional:
        """Get entire section element from heading element (including subsections)"""
        try:
            from bs4 import BeautifulSoup
            
            # Create new div to store section content
            section_content = BeautifulSoup('<div></div>', 'html.parser').div
            
            # Include heading element itself (to display title)
            section_content.append(heading.__copy__())
            
            # Start from element following heading element
            current = heading.next_sibling
            
            # Collect elements until next top-level heading is found or document ends
            # Lower-level headings are treated as subsections within the same section
            while current:
                if current.name == top_level:
                    # Exit when next top-level heading is reached
                    break
                
                if hasattr(current, 'name') and current.name:
                    # Copy element and add to section content (use copy instead of extract)
                    section_content.append(current.__copy__())
                else:
                    # Text node case
                    current = current.next_sibling
                    continue
                
                current = current.next_sibling
            
            return section_content if section_content.contents else None
            
        except Exception as e:
            print(f"Error getting section element: {e}", file=sys.stderr)
            return None
    
    
    def _is_template_list_item(self, text: str) -> bool:
        """Determine if list item is template boilerplate"""
        # Remove basic empty items or items with only whitespace
        if not text.strip():
            return True
        
        return False
    
    def _save_analysis_to_json(self, repo: Repository, created_at: datetime, file_path: str, analysis: dict, content_type: ContentType, output_dir: str):
        """Save analysis result to JSON file"""
        try:
            # Generate filename
            if content_type == ContentType.PR:
                json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_template_titles.json")
            else:
                # For Issue templates, extract template name
                template_name = self._extract_template_name_from_path(file_path)
                json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_{template_name}_template_titles.json")
            
            # Load existing JSON file (if exists)
            cache = {}
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            
            # Add new analysis result
            date_key = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
            cache[date_key] = {
                "file_path": file_path,
                "analysis": analysis,
                "content_type": content_type.value
            }
            
            # Save to JSON file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving analysis to JSON: {e}", file=sys.stderr)
    
    def _extract_template_name_from_path(self, file_path: str) -> str:
        """Extract template name from file path"""
        if not file_path:
            return "default"
        
        # Get filename without extension
        filename = os.path.basename(file_path)
        template_name = os.path.splitext(filename)[0]
        
        # Remove numeric prefix (e.g., "1-bug.yml" -> "bug")
        if template_name and template_name[0].isdigit() and '-' in template_name:
            template_name = template_name.split('-', 1)[1]
        
        # Normalize common template names
        template_name = template_name.lower()
        
        # Unify underscores to hyphens
        template_name = template_name.replace('_', '-')
        
        return template_name
    
    def _load_template_analysis_from_file(self, json_file: str, created_at: datetime):
        """Load template analysis result from JSON file"""
        try:
            # Silently return None if file does not exist
            if not os.path.exists(json_file):
                return None
            with open(json_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Convert datetime to ISO format string
            target_date = created_at.isoformat()
            
            # Filter valid dates (before target_date)
            valid_dates = [date_key for date_key in cache.keys() 
                          if date_key and date_key <= target_date]
            
            if not valid_dates:
                return None
            
            # Select most recent date
            closest_date = max(valid_dates)
            return cache[closest_date]
            
        except Exception as e:
            # Log only exceptions other than file not found
            if os.path.exists(json_file):
                print(f"Error loading template analysis from {json_file}: {e}")
            return None
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load prompt file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file {prompt_file} not found, using default prompt", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"Error loading prompt file {prompt_file}: {e}", file=sys.stderr)
            return ""
    
    def process_repositories(self, repo: Repository): 
        output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Marker to run only once per repository
            done_marker = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_TEMPLATE_TITLES_DONE")
            if os.path.exists(done_marker):
                print("Template titles already processed for this repository. Skipping.")
                return
            
            # Process Pull Request templates
            pr_json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.PR.value}_template_titles.json")
            if not os.path.exists(pr_json_file):
                self._process_templates(repo, ContentType.PR, pull_request_templates(repo.id))
            
            # Process Issue templates (by type)
            issue_templates_list = issue_templates(repo.id)
            if issue_templates_list:
                self._process_templates(repo, ContentType.ISSUE, issue_templates_list)
            else:
                print("  - No Issue templates found")
                
        except Exception as e:
            print(f"Error processing {repo.owner}.{repo.name}: {e}")
        else:
            # Create marker on successful completion (only check existence for idempotency)
            try:
                with open(done_marker, 'w', encoding='utf-8') as f:
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            except Exception:
                pass
    
    def _process_templates(self, repo: Repository, content_type: ContentType, templates_list):
        start_time = time.time()
        prompt_tokens_sum = 0
        completion_tokens_sum = 0

        """Process template list"""
        if not templates_list:
            print(f"  - No {content_type} templates found")
            return
        
        output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
        os.makedirs(output_dir, exist_ok=True)

        if content_type == ContentType.PR:
            # Process PR templates (as before)
            if not os.path.exists(os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_template_titles.json")):
                print(f"Processing {content_type.value} templates...")
                for template in tqdm(templates_list, total=len(templates_list), desc=f"Analyzing {content_type.value} templates"):
                    analysis, prompt_tokens, completion_tokens = self.analyze_template_structure(template.template)
                    prompt_tokens_sum += prompt_tokens
                    completion_tokens_sum += completion_tokens
                    self._save_analysis_to_json(repo, template.created_at, template.file_path, analysis, content_type, output_dir)
            else:
                print(f"{content_type.value} templates already exists")
        elif content_type == ContentType.ISSUE:
            # Process Issue templates (grouped by type)
            # Group templates by type
            template_groups = {}
            for template in templates_list:
                template_name = self._extract_template_name_from_path(template.file_path)
                template_groups.setdefault(template_name, []).append(template)
            
            # Process each template type
            for template_name, templates in template_groups.items():
                template_json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_{template_name}_template_titles.json")
                if not os.path.exists(template_json_file):
                    print(f"  Processing {template_name} templates...")
                    for template in tqdm(templates, total=len(templates), desc=f"Analyzing {template_name} templates"):
                        analysis, prompt_tokens, completion_tokens = self.analyze_template_structure(template.template)
                        prompt_tokens_sum += prompt_tokens
                        completion_tokens_sum += completion_tokens
                        self._save_analysis_to_json(repo, template.created_at, template.file_path, analysis, content_type, output_dir)
                else:
                    print(f"{content_type.value}_{template_name} template analysis already exists")
        
        summary = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "num_rows": len(templates_list),
            "content_type": content_type.value,
            "processing_time_seconds": round(time.time() - start_time, 3),
            "gpt_prompt_tokens": prompt_tokens_sum,
            "gpt_completion_tokens": completion_tokens_sum,
            "gpt_total_tokens": prompt_tokens_sum + completion_tokens_sum,
        }
        json_path = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_template_extractor_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    for repo in target_repos():
        print(f"{repo.id}: {repo.owner}.{repo.name}")
        extractor = TemplateExtractor()
        extractor.process_repositories(repo)

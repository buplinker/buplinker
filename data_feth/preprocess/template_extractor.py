#!python3
"""
プルリクエストテンプレート抽出モジュール
コスト効率を考慮した段階的な抽出アプローチ
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import project_config as config
from data_fetch.database.get import pull_request_templates, issue_templates
from root_util import target_repos, ContentType
from data_fetch.database.tables import Repository

load_dotenv()


class TemplateExtractor:
    def __init__(self, device=None):
        """
        Args:
            device: 'cpu' または 'cuda' を指定。Noneの場合は自動選択（CUDAが利用可能ならCUDA、そうでなければCPU）
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # deviceが指定されていない場合は自動選択
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        self.device = device
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        # テンプレートごとのEmbeddingキャッシュ
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
                
                # header_patternも正規化
                if "header_pattern" in section:
                    original_pattern = section["header_pattern"]
                    # header_patternからも括弧を削除
                    normalized_pattern = re.sub(r'\s*\([^)]*\)', '', original_pattern)
                    section["header_pattern"] = normalized_pattern
        return analysis, prompt_tokens, completion_tokens
    
    def calculate_template_similarity(self, pr_description: str, template_content: str) -> float:
        try:
            # NaNや非文字列の事前チェック
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
            # HTMLが提供されている場合は、HTMLベースのテンプレート削除を試行
            html_extracted = self.extract_from_html(repo, description, description_html, created_at, template_type)
            result["extracted"] = html_extracted
            return result
            
        except Exception as e:
            print(f"Error extracting unique content: {e}", file=sys.stderr)
            return result
    
    def get_closest_template_analysis(self, repo: Repository, description: str, created_at: datetime, template_type: ContentType):
        """指定された日時より前の最も直近のテンプレート分析結果を取得"""
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
        """PRテンプレート分析結果を取得"""
        json_file = os.path.join(os.path.dirname(__file__), "template_title_repositories", 
                                f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.PR.value}_template_titles.json")

        return self._load_template_analysis_from_file(json_file, created_at)
    
    def _get_issue_template_analysis(self, repo: Repository, description: str, created_at: datetime):
        """Issueテンプレート分析結果を取得（類似度ベースで最適なテンプレートを選択）"""
        try:
            # 利用可能なIssueテンプレート分析結果を取得
            available_templates = self._get_available_issue_templates(repo, created_at)
            if not available_templates:
                return None
            
            # 各テンプレートとの類似度を計算
            template_similarities = {}
            # テンプレートのEmbeddingキャッシュ用のキー
            cache_key_prefix = f"{repo.id}_{repo.owner}_{repo.name}"

            for template_name, template_data in available_templates.items():
                # テンプレートの内容を取得
                template_content = self._extract_template_content(template_data)
                if template_content:
                    # テンプレートEmbeddingのキャッシュキー
                    template_cache_key = f"{cache_key_prefix}_{template_name}"
                    
                    # キャッシュからテンプレートEmbeddingを取得または計算
                    if template_cache_key in self._template_embedding_cache:
                        template_embedding = self._template_embedding_cache[template_cache_key]
                    else:
                        template_text = self._preprocess_text(template_content)
                        template_embedding = self.similarity_model.encode([template_text])
                        self._template_embedding_cache[template_cache_key] = template_embedding
                    
                    # Issue のEmbeddingを計算
                    issue_text = self._preprocess_text(description)
                    issue_embedding = self.similarity_model.encode([issue_text])
                    
                    # コサイン類似度を計算
                    similarity = cosine_similarity(issue_embedding, template_embedding)[0][0]
                    template_similarities[template_name] = float(similarity)
            
            # 最も類似度の高いテンプレートを選択
            if template_similarities:
                best_template = max(template_similarities, key=template_similarities.get)
                return available_templates[best_template]
            
        except Exception as e:
            print(f"Error getting issue template analysis: {e}")
    
    def _get_available_issue_templates(self, repo: Repository, created_at: datetime) -> dict:
        """利用可能なIssueテンプレート分析結果を取得"""
        try:
            output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
            pattern = f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.ISSUE.value}_*_template_titles.json"
            matching_files = glob.glob(os.path.join(output_dir, pattern))
            
            available_templates = {}
            for file_path in matching_files:
                # ファイル名からテンプレート名を抽出
                filename = os.path.basename(file_path)
                template_name = filename.replace(f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.ISSUE.value}_", "").replace("_template_titles.json", "")
                
                # テンプレート分析結果を読み込み
                template_data = self._load_template_analysis_from_file(file_path, created_at)
                if template_data:
                    available_templates[template_name] = template_data
            
            return available_templates
            
        except Exception as e:
            print(f"Error getting available issue templates: {e}")
            return {}
    
    def _extract_template_content(self, template_data: dict) -> str:
        """テンプレート分析結果からテンプレート内容を抽出"""
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
        """セクション名を正規化（注釈や括弧を削除）"""
        if not section_name:
            return ""
        
        # 括弧内の注釈を削除
        # (required), (optional), (recommended) などを削除
        section_name = re.sub(r'\s*\([^)]*\)', '', section_name)
        
        # その他の一般的な注釈を削除
        annotations = [
            'required', 'optional', 'recommended', 'mandatory',
            'needed', 'important', 'critical', 'essential'
        ]
        
        for annotation in annotations:
            # 単語境界でマッチング
            pattern = r'\b' + re.escape(annotation) + r'\b'
            section_name = re.sub(pattern, '', section_name, flags=re.IGNORECASE)
        
        # 余分な空白を整理
        section_name = re.sub(r'\s+', ' ', section_name).strip()
        
        # 末尾のコロンやピリオドを削除
        section_name = re.sub(r'[:.]$', '', section_name)
        
        return section_name
    
    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        if not text:
            return ""
        # 改行を空白に変換してから連続する空白を整理
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_template_boilerplate(self, text: str) -> str:
        """テンプレート定型文を除去"""
        # コメント除去
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # 空のチェックボックス除去
        text = re.sub(r'-\s*\[\s*\]\s*[^\n]*', '', text)
        # プレースホルダー除去
        text = re.sub(r'<.*?>', '', text)
        # 連続する空白を整理
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def extract_from_html(self, repo: Repository, description: str, html_content: str, created_at: datetime, template_type: ContentType) -> Optional[str]:
        """HTML形式のテンプレートから重要なコンテンツのみを抽出"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # テンプレート分析結果を取得（セクション名を削除するために使用）
            template_analysis_data = self.get_closest_template_analysis(repo, description, created_at, template_type)
            
            # 重要なセクションのみを抽出
            important_sections = self._extract_important_sections(repo, description, soup, created_at, template_type)
            
            # 見出しより前の部分（最初の段落など）を抽出
            pre_heading_content = self._extract_pre_heading_content(soup)
            
            # テンプレート分析結果がない場合や重要なセクションが空の場合は、最初の段落を抽出
            if not important_sections:
                # 見出しがない場合、重要でないセクションを除外して段落を抽出
                if template_analysis_data and "analysis" in template_analysis_data:
                    # すべての段落から重要でないセクションを除外
                    text = self._extract_content_excluding_unimportant_sections(soup, template_analysis_data["analysis"])
                elif pre_heading_content:
                    # 見出しより前の部分がある場合はそれを使用（テンプレート分析結果がない場合）
                    text = pre_heading_content
                else:
                    # テンプレート分析結果がない場合は、すべての段落を抽出
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        texts = [p.get_text().strip() for p in paragraphs]
                        text = ' '.join(texts)
                    else:
                        # 段落が見つからない場合は、元のテキストをそのまま返す
                        text = soup.get_text()
                
                # テンプレート分析結果がある場合は、セクション名を削除
                if template_analysis_data and "analysis" in template_analysis_data:
                    text = self._remove_template_section_names_from_text(text, template_analysis_data["analysis"])
                
                # 改行を適切に処理
                text = re.sub(r'\n+', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            # 重要なセクションの内容を結合
            extracted_text = self._combine_important_content(important_sections)
            
            # 見出しより前の部分も結合（セクション名を削除してから）
            if pre_heading_content:
                # 見出しより前の部分からも重要でないセクションを除外してからセクション名を削除
                if template_analysis_data and "analysis" in template_analysis_data:
                    # 見出しより前の部分のHTMLから重要でないセクションを除外
                    # pre_heading_contentはテキストなので、元のsoupから見出しより前の要素を直接取得
                    first_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if first_heading:
                        # 最初の見出しより前の要素から重要でないセクションを除外
                        pre_heading_elements = []
                        all_elements = soup.find_all(['p', 'ul', 'ol', 'div'])
                        for element in all_elements:
                            if element in first_heading.find_all_previous(['p', 'ul', 'ol', 'div']):
                                if not element.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                                    pre_heading_elements.append(element)
                        
                        # 重要でないセクションを除外するために、一時的なsoupを作成
                        temp_soup = BeautifulSoup('<div></div>', 'html.parser')
                        temp_div = temp_soup.div
                        for element in pre_heading_elements:
                            temp_div.append(element.__copy__())
                        
                        pre_heading_content = self._extract_content_excluding_unimportant_sections(temp_div, template_analysis_data["analysis"])
                    else:
                        # 見出しがない場合（通常は来ない）
                        pre_heading_soup = BeautifulSoup(pre_heading_content, 'html.parser')
                        pre_heading_content = self._extract_content_excluding_unimportant_sections(pre_heading_soup, template_analysis_data["analysis"])
                    
                    # セクション名を削除
                    pre_heading_content = self._remove_template_section_names_from_text(pre_heading_content, template_analysis_data["analysis"])
                
                if extracted_text:
                    extracted_text = pre_heading_content + " " + extracted_text
                else:
                    extracted_text = pre_heading_content
            
            # 意味のあるコンテンツが残っているかチェック
            if extracted_text and len(extracted_text.strip()) > 10:
                # 改行を適切に処理
                extracted_text = re.sub(r'\n+', ' ', extracted_text)
                extracted_text = re.sub(r'\s+', ' ', extracted_text)
                return extracted_text.strip()
            
            # 重要なセクションが空の場合は、最初の段落を抽出
            print("Important sections are empty. Extracting first paragraph.")
            if pre_heading_content:
                text = pre_heading_content
            else:
                first_p = soup.find('p')
                if first_p:
                    text = first_p.get_text()
                else:
                    text = soup.get_text()
            
            # テンプレート分析結果がある場合は、セクション名を削除
            if template_analysis_data and "analysis" in template_analysis_data:
                text = self._remove_template_section_names_from_text(text, template_analysis_data["analysis"])
            
            # 改行を適切に処理
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting from HTML: {e}", file=sys.stderr)
            return None
    
    def _remove_template_section_names_from_text(self, text: str, analysis: dict) -> str:
        """テキストからテンプレートのセクション名を削除（重要なセクション名は保持）"""
        if not text or not analysis:
            return text
        
        # 重要なセクション名のセットを作成（これらは削除しない）
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
        
        # 重要でないセクション名とheader_patternを取得
        sections_to_remove = []
        for section in analysis.get("sections", []):
            section_name = section.get("name", "")
            header_pattern = section.get("header_pattern", "")
            if section_name:
                sections_to_remove.append(section_name)
            if header_pattern:
                # header_patternからマークダウンマーカーを除去
                pattern_text = re.sub(r'\*\*|###|##|#|\*\s*', '', header_pattern).strip()
                if pattern_text:
                    sections_to_remove.append(pattern_text)
        
        # 各セクション名をテキストから削除（重要なセクション名は除外）
        for section_name in sections_to_remove:
            if not section_name:
                continue
            # セクション名を正規化（比較用）
            normalized_section_name = self._normalize_section_name(section_name).lower()
            # 重要なセクション名は削除しない
            if normalized_section_name in important_section_names:
                continue
            # 正規化されたセクション名で始まるパターン（括弧などを含む可能性がある）
            # セクション名の後に括弧やコロンが続く場合もマッチ
            pattern = re.escape(normalized_section_name)
            # セクション名で始まり、その後に空白、コロン、括弧などが続く場合を削除
            text = re.sub(r'^\s*' + pattern + r'(\s*\([^)]*\))?\s*:?\s*', '', text, flags=re.IGNORECASE)
            # テキスト内のセクション名も削除（括弧を含む場合も対応）
            text = re.sub(r'\b' + pattern + r'(\s*\([^)]*\))?\s*:?\s*', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_content_excluding_unimportant_sections(self, soup: BeautifulSoup, analysis: dict) -> str:
        """見出しがない場合、重要でないセクションを除外してコンテンツを抽出"""
        # 重要でないセクション名のセットを作成
        unimportant_section_names = set()
        unimportant_header_patterns = set()
        
        for section in analysis.get("sections", []):
            is_meaningful = section.get("is_meaningful_content", False)
            importance = section.get("importance", "low")
            
            if not is_meaningful or importance == "low":
                section_name = section.get("name", "")
                header_pattern = section.get("header_pattern", "")
                if section_name:
                    # セクション名を正規化して追加
                    normalized_name = self._normalize_section_name(section_name).lower()
                    unimportant_section_names.add(normalized_name)
                if header_pattern:
                    # header_patternからマークダウンマーカーを除去
                    pattern_text = re.sub(r'\*\*|###|##|#|\*\s*', '', header_pattern).strip()
                    if pattern_text:
                        # header_patternも正規化
                        normalized_pattern = self._normalize_section_name(pattern_text).lower()
                        unimportant_header_patterns.add(normalized_pattern)
        
        # すべての段落を取得
        all_paragraphs = soup.find_all('p')
        extracted_texts = []
        
        i = 0
        while i < len(all_paragraphs):
            p = all_paragraphs[i].__copy__()
            p_text = p.get_text().strip()
            
            # <strong>タグ内のテキストをチェック
            strong_tags = p.find_all('strong')
            
            # 重要でないセクションかどうかチェック
            is_all_unimportant = len(strong_tags) > 0  # 見出しがある場合のみTrue
            has_important_section = False
            for strong in strong_tags:
                strong_text = strong.get_text().strip()
                normalized_strong_text = self._normalize_section_name(strong_text).lower()
                
                # 重要でないセクションかどうかチェック
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
                    # 重要なセクションの<strong>タグに角括弧を追加（括弧内の注釈を削除してから）
                    strong_text = strong.get_text().strip()
                    if strong_text:
                        # 括弧内の注釈を削除
                        strong_text = re.sub(r'\s*\([^)]*\)', '', strong_text).strip()
                        strong.clear()
                        strong.append(f'[{strong_text}] ')
                else:
                    # 重要でないセクションの<strong>タグを削除
                    strong.decompose()
            
            # すべての見出しが重要でない場合、このセクションの内容をスキップ
            if is_all_unimportant and strong_tags:
                # 次の見出し（<strong>タグがある段落）までスキップ
                i += 1
                while i < len(all_paragraphs):
                    next_p = all_paragraphs[i]
                    next_strong = next_p.find('strong')
                    # 次のセクション（<strong>タグがある段落）が見つかったら終了
                    if next_strong:
                        break
                    i += 1
                continue
            
            # 重要でないセクションでない場合、または重要でないセクションを削除した場合、テキストを追加
            if p_text:
                modified_text = p.get_text().strip()
                if modified_text and len(modified_text) > 0:
                    extracted_texts.append(modified_text)
            
            i += 1
        
        return ' '.join(extracted_texts)
    
    def _extract_important_sections(self, repo: Repository, description: str, soup: BeautifulSoup, created_at: datetime, template_type: ContentType) -> list:
        """重要なセクションのみを抽出"""
        important_sections = []
        
        # テンプレート分析結果を取得
        template_analysis_data = self.get_closest_template_analysis(repo, description, created_at, template_type)
        if not template_analysis_data or "analysis" not in template_analysis_data:
            # テンプレート分析結果がない場合は、空のリストを返す（元のテキストをそのまま返す）
            return important_sections
        
        analysis = template_analysis_data["analysis"]
        
        # 重要なセクション名とheader_patternのマッピングを作成
        important_section_info = {}
        for section in analysis.get("sections", []):
            is_meaningful = section.get("is_meaningful_content", False)
            importance = section.get("importance", "low")
            
            if is_meaningful and importance != "low":
                section_name = section.get("name", "")
                header_pattern = section.get("header_pattern", "")
                if section_name:
                    # セクション名を正規化してキーとして使用
                    normalized_name = self._normalize_section_name(section_name).lower()
                    important_section_info[normalized_name] = {
                        'name': section_name,
                        'header_pattern': header_pattern
                    }
        
        # 最上位の見出しレベルを動的に検出
        top_level = self._get_top_level_heading(soup)
        
        # 最上位の見出しタグのみをチェック
        for heading in soup.find_all(top_level):
            section_title_raw = heading.get_text().strip()
            # 見出しも正規化してから比較
            section_title = self._normalize_section_name(section_title_raw).lower()
            
            # 重要なセクションかどうかをチェック（正規化後のセクション名で比較）
            if section_title in important_section_info:
                # セクション全体（最上位見出しから次の最上位見出しまで）を取得
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
        """HTMLコンテンツ内の最上位の見出しレベルを検出"""
        # h2からh6まで順番にチェック
        for level in range(2, 7):
            headings = soup.find_all(f'h{level}')
            if headings:
                return f'h{level}'
        
        # 見出しが見つからない場合はh2をデフォルトとする
        return 'h2'
    
    def _extract_pre_heading_content(self, soup: BeautifulSoup) -> str:
        """最初の見出しより前のコンテンツ（段落など）を抽出"""
        try:
            # すべての見出しを取得（ドキュメント順序で）
            all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if not all_headings:
                # 見出しが見つからない場合は、すべての段落を抽出
                paragraphs = soup.find_all('p')
                if paragraphs:
                    texts = [p.get_text().strip() for p in paragraphs]
                    combined = ' '.join(texts)
                    combined = re.sub(r'\n+', ' ', combined)
                    combined = re.sub(r'\s+', ' ', combined)
                    return combined.strip()
                return ""
            
            # 最初の見出しを取得
            first_heading = all_headings[0]
            
            # 最初の見出しより前のすべての要素を抽出（ドキュメント順序を保持）
            pre_heading_texts = []
            all_elements = soup.find_all(['p', 'ul', 'ol', 'div'])
            
            for element in all_elements:
                # 最初の見出しより前にある要素を抽出
                if element in first_heading.find_all_previous(['p', 'ul', 'ol', 'div']):
                    # 見出し内にネストされている要素は除外
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
        """重要なセクションの内容を結合してテキストを構築"""
        combined_text = []
        
        for section in important_sections:
            section_element = section['element']
            section_name = section.get('section_name', '')
            header_pattern = section.get('header_pattern', '')
            
            # セクションの内容からテキストを抽出（セクション名とheader_patternを渡す）
            section_text = self._extract_section_text(section_element, section_name, header_pattern)
            
            if section_text and section_text.strip():
                # 改行を適切に処理
                section_text = re.sub(r'\n+', ' ', section_text.strip())
                combined_text.append(section_text)
        
        return ' '.join(combined_text)
    
    def _extract_section_text(self, section_element, section_name: str = '', header_pattern: str = '') -> str:
        """セクション要素からテキストを抽出（重要なセクション名は保持）"""
        # 見出し要素（h1-h6）も保持する（重要セクションのタイトルを表示するため）
        # セクション名の<strong>タグも保持する（重要なセクションなので保持する）
        # 内容はそのまま保持
        
        # セクションタイトルに「[」と「]」を追加（括弧内の注釈を削除してから）
        for heading in section_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text().strip()
            if heading_text:
                # 括弧内の注釈を削除
                heading_text = re.sub(r'\s*\([^)]*\)', '', heading_text).strip()
                # テキストを取得して、角括弧を追加してから置き換え
                heading.clear()
                heading.append(f'[{heading_text}] ')
        
        # チェックボックスや空のリスト項目を削除
        for checkbox in section_element.find_all('input', {'type': 'checkbox'}):
            checkbox.parent.decompose()
        
        # 番号付きリストの番号を保持するため、li要素に番号を追加（空の項目を削除する前に行う）
        for ol in section_element.find_all('ol'):
            for idx, li in enumerate(ol.find_all('li'), start=1):
                li_text = li.get_text().strip()
                if li_text:
                    # テキストノードを削除して番号付きテキストに置き換え
                    li.clear()
                    li.append(f"{idx}. {li_text}")
        
        # 空のリスト項目やテンプレート定型文を削除
        for li in section_element.find_all('li'):
            li_text = li.get_text().strip()
            if not li_text or self._is_template_list_item(li_text):
                li.decompose()
        
        # 空の段落を削除
        for p in section_element.find_all('p'):
            if not p.get_text().strip():
                p.decompose()
        
        # テキストを抽出（重要なセクションなので、セクション名はそのまま保持）
        text = section_element.get_text()
        
        # 改行と空白を適切に処理
        text = re.sub(r'\n\s*\n', ' ', text)  # 複数の改行を単一の空白に
        text = re.sub(r'\n+', ' ', text)      # 単一の改行も空白に
        text = re.sub(r'\s+', ' ', text)      # 連続する空白を単一の空白に
        
        return text.strip()
    
    
    def _get_section_element(self, heading, top_level: str) -> Optional:
        """見出し要素からセクション全体の要素を取得（サブセクションも含む）"""
        try:
            from bs4 import BeautifulSoup
            
            # セクションの内容を格納するための新しいdivを作成
            section_content = BeautifulSoup('<div></div>', 'html.parser').div
            
            # 見出し要素自体も含める（タイトルを表示するため）
            section_content.append(heading.__copy__())
            
            # 見出し要素の次の要素から開始
            current = heading.next_sibling
            
            # 次の最上位見出し要素が見つかるまで、または文書の終わりまで要素を収集
            # より下位の見出しは同じセクション内のサブセクションとして扱う
            while current:
                if current.name == top_level:
                    # 次の最上位見出し要素に到達したら終了
                    break
                
                if hasattr(current, 'name') and current.name:
                    # 要素をコピーしてセクション内容に追加（extractではなくcopyを使用）
                    section_content.append(current.__copy__())
                else:
                    # テキストノードの場合
                    current = current.next_sibling
                    continue
                
                current = current.next_sibling
            
            return section_content if section_content.contents else None
            
        except Exception as e:
            print(f"Error getting section element: {e}", file=sys.stderr)
            return None
    
    
    def _is_template_list_item(self, text: str) -> bool:
        """リスト項目がテンプレート定型文かどうかを判定"""
        # 基本的な空項目や空白のみの項目を削除
        if not text.strip():
            return True
        
        return False
    
    def _save_analysis_to_json(self, repo: Repository, created_at: datetime, file_path: str, analysis: dict, content_type: ContentType, output_dir: str):
        """分析結果をJSONファイルに保存"""
        try:
            # ファイル名を生成
            if content_type == ContentType.PR:
                json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_template_titles.json")
            else:
                # Issueテンプレートの場合、テンプレート名を抽出
                template_name = self._extract_template_name_from_path(file_path)
                json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{content_type.value}_{template_name}_template_titles.json")
            
            # 既存のJSONファイルを読み込み（存在する場合）
            cache = {}
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            
            # 新しい分析結果を追加
            date_key = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
            cache[date_key] = {
                "file_path": file_path,
                "analysis": analysis,
                "content_type": content_type.value
            }
            
            # JSONファイルに保存
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving analysis to JSON: {e}", file=sys.stderr)
    
    def _extract_template_name_from_path(self, file_path: str) -> str:
        """ファイルパスからテンプレート名を抽出"""
        if not file_path:
            return "default"
        
        # ファイル名から拡張子を除いて取得
        filename = os.path.basename(file_path)
        template_name = os.path.splitext(filename)[0]
        
        # 数字プレフィックスを除去（例: "1-bug.yml" -> "bug"）
        if template_name and template_name[0].isdigit() and '-' in template_name:
            template_name = template_name.split('-', 1)[1]
        
        # 一般的なテンプレート名の正規化
        template_name = template_name.lower()
        
        # アンダースコアをハイフンに統一
        template_name = template_name.replace('_', '-')
        
        return template_name
    
    def _load_template_analysis_from_file(self, json_file: str, created_at: datetime):
        """JSONファイルからテンプレート分析結果を読み込み"""
        try:
            # ファイルが存在しない場合は静かにNone
            if not os.path.exists(json_file):
                return None
            with open(json_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # 日時をISO形式の文字列に変換
            target_date = created_at.isoformat()
            
            # 有効な日時（target_date以前）をフィルタリング
            valid_dates = [date_key for date_key in cache.keys() 
                          if date_key and date_key <= target_date]
            
            if not valid_dates:
                return None
            
            # 最も直近の日時を選択
            closest_date = max(valid_dates)
            return cache[closest_date]
            
        except Exception as e:
            # 存在しない以外の例外のみログ
            if os.path.exists(json_file):
                print(f"Error loading template analysis from {json_file}: {e}")
            return None
    
    def _load_prompt(self, prompt_file: str) -> str:
        """プロンプトファイルを読み込む"""
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
            # リポジトリ単位で一度だけ実行するためのマーカー
            done_marker = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_TEMPLATE_TITLES_DONE")
            if os.path.exists(done_marker):
                print("Template titles already processed for this repository. Skipping.")
                return
            
            # Pull Request テンプレートの処理
            pr_json_file = os.path.join(output_dir, f"{repo.id}_{repo.owner}.{repo.name}_{ContentType.PR.value}_template_titles.json")
            if not os.path.exists(pr_json_file):
                self._process_templates(repo, ContentType.PR, pull_request_templates(repo.id))
            
            # Issue テンプレートの処理（種類別）
            issue_templates_list = issue_templates(repo.id)
            if issue_templates_list:
                self._process_templates(repo, ContentType.ISSUE, issue_templates_list)
            else:
                print("  - No Issue templates found")
                
        except Exception as e:
            print(f"Error processing {repo.owner}.{repo.name}: {e}")
        else:
            # 正常終了時にマーカー作成（冪等にするため存在チェックのみ）
            try:
                with open(done_marker, 'w', encoding='utf-8') as f:
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            except Exception:
                pass
    
    def _process_templates(self, repo: Repository, content_type: ContentType, templates_list):
        start_time = time.time()
        prompt_tokens_sum = 0
        completion_tokens_sum = 0

        """テンプレートリストを処理"""
        if not templates_list:
            print(f"  - No {content_type} templates found")
            return
        
        output_dir = os.path.join(os.path.dirname(__file__), "template_title_repositories")
        os.makedirs(output_dir, exist_ok=True)

        if content_type == ContentType.PR:
            # PRテンプレートの処理（従来通り）
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
            # Issueテンプレートの処理（種類別にグループ化）
            # テンプレートを種類別にグループ化
            template_groups = {}
            for template in templates_list:
                template_name = self._extract_template_name_from_path(template.file_path)
                template_groups.setdefault(template_name, []).append(template)
            
            # 各テンプレート種類ごとに処理
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

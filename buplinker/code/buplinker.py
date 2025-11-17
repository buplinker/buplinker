import os
import sys
from util import clean_text, NDCG_at_K, Hit_at_K, precision_at_K, recall_at_K, MRR, results_to_df_llm, recall, precision, F1_score, get_group_key

# プロジェクトルートをパスに追加（他のインポートの前に実行）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from datetime import datetime
import time
import json
from dotenv import load_dotenv
import shutil
from typing import List, Dict, Any, Tuple
from enum import Enum
import re
from tqdm import tqdm
import project_config as config
from root_util import GroupType

load_dotenv()

class PromptType(Enum):
    """プロンプトタイプの定義"""
    ORIGINAL = "original"
    RERANK = "rerank"
    IDENTIFY = "identify"


def normalize_group_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def format_prompt(text, top_k_results, group_type='ur_pr', prompt_type='rerank', prompt_with_relevance=False, use_gpt5=False):
    prompts_dir = os.path.join(os.path.dirname(__file__), f"{group_type}/prompts")
    
    if use_gpt5:
        system_prompt_filename = f"{prompt_type}_system_with_reason_gpt5.txt"
    else:
        system_prompt_filename = f"{prompt_type}_system_with_relevance.txt" if prompt_with_relevance else f"{prompt_type}_system.txt"
    with open(os.path.join(prompts_dir, system_prompt_filename), "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    with open(os.path.join(prompts_dir, "user.txt"), "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()

    if group_type == GroupType.UR_PR.value:
        pull_requests_text = ""
        for ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target in top_k_results:
            pull_requests_text += f"Pull Request ID: {pr_id}. Pull Request Title/Description: {pull_request}\n"

        user_prompt = user_prompt_template.format(
            user_review=text,
            pull_requests=pull_requests_text
        )
    else:
        user_reviews_text = ""
        for ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target in top_k_results:
            user_reviews_text += f"User Review ID: {ur_id}. User Review: {user_review}\n"

        user_prompt = user_prompt_template.format(
            pull_request=text,
            user_reviews=user_reviews_text
        )
    
    if use_gpt5:
        return [{"role": "developer", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    else:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def results_with_gpt(df: pd.DataFrame, query_text, results, top_k, prompt_type='original', group_type='ur_pr', prompt_with_relevance=False, use_gpt5=False) -> Tuple[List[Dict[str, Any]], int, int, List[Dict[str, Any]]]:
    prompt_tokens = 0
    completion_tokens = 0
    warnings: List[Dict[str, Any]] = []
    
    if len(results) <= top_k:
        print(f"Fewer than {top_k} results, reranking all available.")
        top_k_results = results
        remaining_results = []
    else:
        top_k_results = results[:top_k] 
        remaining_results = results[top_k:]  

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        messages = format_prompt(query_text, top_k_results, group_type, prompt_type, prompt_with_relevance, use_gpt5)
       
        if args.use_gpt5:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=config.OPENAI_MODEL_GPT5,
                reasoning_effort=config.OPENAI_REASONING_EFFORT,
            )
        else:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=config.OPENAI_MODEL,
                temperature=config.OPENAI_TEMPERATURE, 
            )

        if hasattr(chat_completion, "usage") and chat_completion.usage:
            prompt_tokens = getattr(chat_completion.usage, "prompt_tokens", 0)
            completion_tokens = getattr(chat_completion.usage, "completion_tokens", 0)

        response_message = chat_completion.choices[0].message.content.strip()

        # Parse the response based on prompt type
        if prompt_with_relevance:
            # Extract JSON from response (in case there's extra text before/after)
            # Match both dict format { ... } and list format [ ... ]
            json_match = re.search(r'(\{.*\}|\[.*\])', response_message, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                json_text = response_message
            
            try:
                response_data = json.loads(json_text)
                
                # Handle both list format [] and dict format {'links': [...]}
                if isinstance(response_data, list):
                    # If response is a list, treat it as empty links
                    links = []
                elif isinstance(response_data, dict):
                    # If response is a dict, extract links
                    links = response_data.get('links', [])
                else:
                    links = []
                
                reranked_ids = [item['id'] for item in links if isinstance(item, dict) and 'id' in item]
                reranked_ids_to_relevance = {item['id']: item['relevance'] for item in links if isinstance(item, dict) and 'id' in item and 'relevance' in item}
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"JSON text: {json_text}")
                raise e
        else:
            # Parse the response into a list of commit IDs (original format)
            reranked_ids = [
                line.strip("[]'\" ").split("-", 1)[1]  # 最初の'-'のみで分割
                for line in response_message.split(",") 
                if "-" in line
            ]
        
        if len(reranked_ids) == 0:
            warnings.append({"code": "no_ids_returned"})

        if group_type == GroupType.UR_PR.value:
            id_to_data = {pr_id: (ur_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target) for ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target in top_k_results}
        else:
            id_to_data = {ur_id: (pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target) for ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target in top_k_results}

        final_results, reranked_top_k_results = [], []
        processed_ids = set()  # GPTから返ってきたIDを追跡
        invalid_ids = []
        
        for idx, id_val in enumerate(reranked_ids, start=1):
            if id_val in id_to_data:
                processed_ids.add(id_val)  # 処理済みIDを記録
                if group_type == GroupType.UR_PR.value:
                    ur_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target = id_to_data[id_val]
                    pr_id = id_val
                else:
                    pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target = id_to_data[id_val]
                    ur_id = id_val
                
                result_dict = {
                    "rank": idx,
                    "ur_id": ur_id,
                    "pr_id": pr_id,
                    "similarity_score": float(sim_score),
                    "create_date": create_date,
                    "pull_request_date": pull_request_date,
                    "latest_release_date": latest_release_date,
                    "user_review": user_review,
                    "pull_request": pull_request,
                    "url": url,
                    "author": author,
                    "label": target
                }
                if prompt_with_relevance:
                    result_dict["relevance"] = reranked_ids_to_relevance.get(id_val, '')
                reranked_top_k_results.append(result_dict)
            else:
                invalid_ids.append(id_val)
        
        if invalid_ids:
            warnings.append({"code": "invalid_ids_returned", "details": invalid_ids})
        
        if prompt_type != PromptType.IDENTIFY.value:
            # GPTから返ってこなかったデータを追加（元の順序を保持）
            current_rank = len(reranked_top_k_results) + 1
            
            for ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target in top_k_results:
                if group_type == GroupType.UR_PR.value:
                    check_id = pr_id
                else:
                    check_id = ur_id
                
                # GPTから返ってこなかったデータのみ追加
                if check_id not in processed_ids:
                    result_dict = {
                        "rank": current_rank,
                        "ur_id": ur_id,
                        "pr_id": pr_id,
                        "similarity_score": float(sim_score),
                        "create_date": create_date,
                        "pull_request_date": pull_request_date,
                        "latest_release_date": latest_release_date,
                        "user_review": user_review,
                        "pull_request": pull_request,
                        "url": url,
                        "author": author,
                        "label": target
                    }
                    if prompt_with_relevance:
                        result_dict["relevance"] = ""  # 不足データには関連度なし
                    reranked_top_k_results.append(result_dict)
                    current_rank += 1

        if args.prompt_type == PromptType.IDENTIFY.value:
            final_results = reranked_top_k_results
        else:
            final_results = reranked_top_k_results + [
                {
                    "rank": idx + len(reranked_top_k_results),
                    "ur_id": ur_id,
                    "pr_id": pr_id,
                    "similarity_score": float(sim_score),
                    "create_date": create_date,
                    "pull_request_date": pull_request_date,
                    "latest_release_date": latest_release_date,
                    "user_review": user_review,
                    "pull_request": pull_request,
                    "url": url,
                    "author": author,
                    "label": target
                }
                for idx, (ur_id, pr_id, sim_score, create_date, pull_request_date, latest_release_date, user_review, pull_request, url, author, target) in enumerate(remaining_results, start=1)
            ]

        if len(reranked_top_k_results) == 0:
            warnings.append({"code": "no_results_after_processing"})

        return final_results, prompt_tokens, completion_tokens, warnings

    except Exception as e:
        print(f"Error during OpenAI API call or response parsing: {e}")
        sys.exit(1)
        

if __name__ == "__main__":
    run_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_type', type=str, default='ur_pr', help="Group type: 'ur_pr' or 'pr_ur' (default: 'ur_pr')")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file")
    parser.add_argument('--index_dir', type=str, required=True, help="Directory for FAISS index")
    parser.add_argument('--output_result_path', type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument('--embedding_selected', type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument('--top_k', type=int, required=True, help="Top k results to rerank")
    parser.add_argument('--prompt_type', type=str, required=True, help="Prompt type: 'original' or 'rerank' or 'identify' (default: 'original')")
    parser.add_argument('--prompt_with_relevance', action='store_true', default=False, help="Prompt with relevance (default: False)")
    parser.add_argument('--use_gpt5', action='store_true', default=False, help="Use GPT-5 (default: False)")

    args = parser.parse_args()

    embedding_selected = args.embedding_selected
    if embedding_selected == "OpenAIEmbeddings":
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # already normalized
    elif embedding_selected == "HuggingFace-mpnet":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}  # normalize at source
        )
    elif embedding_selected == "HuggingFace-minilm":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}  # normalize at source
        )
    else:
        raise ValueError(f"Unknown embedding model: {embedding_selected}")

    print("Starting processing...")

    # CSVを読み込み（改行を含むフィールドに対応）
    # pythonエンジンは柔軟性が高く、改行を含むフィールドを正しく処理できる
    df = pd.read_csv(
        args.csv_file, 
        engine='python',
        encoding='utf-8',
        on_bad_lines='warn'
    )
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Unique UR IDs: {df['ur_id'].nunique()}")
    print(f"Unique PR IDs: {df['pr_id'].nunique()}")

    grouped = df.groupby('ur_id') if args.group_type == GroupType.UR_PR.value else df.groupby('pr_id')

    total_prompt_tokens = 0
    total_completion_tokens = 0

    group_key = get_group_key(args.group_type)
    csv_path = f"{args.output_result_path}.csv"
    json_path = f"{args.output_result_path}.json"
    summary_path = f"{args.output_result_path}_summary.json"

    processed_groups = set()
    csv_header_written = False
    if os.path.exists(csv_path):
        csv_header_written = True
        try:
            existing_groups_df = pd.read_csv(csv_path, usecols=[group_key])
            for value in existing_groups_df[group_key].dropna():
                normalized = normalize_group_id(value)
                if normalized is not None:
                    processed_groups.add(normalized)
        except (ValueError, pd.errors.EmptyDataError):
            pass

    # Process each group (either ur_id or pr_id based on group_type)
    for group_id, group in grouped:
        normalized_group_id = normalize_group_id(group_id)
        if normalized_group_id in processed_groups:
            if args.group_type == GroupType.UR_PR.value:
                print(f"Skipping already processed user review ID: {group_id}")
            else:
                print(f"Skipping already processed pull request ID: {group_id}")
            continue

        if args.group_type == GroupType.UR_PR.value:
            print(f"Processing user review ID: {group_id}")
        else:
            print(f"Processing pull request ID: {group_id}")

        documents = []
        last_row = None
        for _, row in tqdm(group.iterrows(), total=len(group), desc="Processing group"):
            last_row = row
            if args.group_type == GroupType.UR_PR.value:
                pr_id = row['pr_id']
                create_date = row['create_date']
                pull_request_date = row['pull_request_date']
                latest_release_date = row['latest_release_date']
                url = row['url']
                author = row['author']
                target = row['target']
                cleaned_content = clean_text(str(row['title'])) + " " + clean_text(str(row['description']))
                documents.append(Document(page_content=cleaned_content, metadata={'pr_id': pr_id, 'create_date': create_date, 'pull_request_date': pull_request_date, 'latest_release_date': latest_release_date, 'url': url, 'author': author, 'target': target}))
            else:
                ur_id = row['ur_id']
                create_date = row['create_date']
                pull_request_date = row['pull_request_date']
                latest_release_date = row['latest_release_date']
                url = row['url']
                author = row['author']
                target = row['target']
                cleaned_content = clean_text(str(row['review']))
                documents.append(Document(page_content=cleaned_content, metadata={'ur_id': ur_id, 'create_date': create_date, 'pull_request_date': pull_request_date, 'latest_release_date': latest_release_date, 'url': url, 'author': author, 'target': target}))

        if last_row is None:
            print(f"No records found for group {group_id}, skipping.")
            continue

        # Index documents in FAISS
        faiss_index_dir = os.path.join(args.index_dir, f"faiss_index_{args.group_type}_{group_id}")
        if os.path.exists(faiss_index_dir):
            faiss_index = FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            texts = [doc.page_content for doc in documents]
            vecs = np.asarray(embeddings.embed_documents(texts), dtype="float32")  # no manual normalize

            assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3), "Embeddings are not unit-normalized!"

            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product == cosine for unit vectors
            index.add(vecs)

            ids = [str(i) for i in range(len(documents))]
            docstore = InMemoryDocstore({ids[i]: documents[i] for i in range(len(documents))})
            id_map = {i: ids[i] for i in range(len(documents))}

            faiss_index = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=id_map,
            )
            faiss_index.save_local(faiss_index_dir)

        if args.group_type == GroupType.UR_PR.value:
            user_review = clean_text(str(last_row['review']))
            query = f"{user_review}"
        else:
            pull_request = clean_text(str(last_row['title'])) + " " + clean_text(str(last_row['description']))
            query = f"{pull_request}"

        results = faiss_index.similarity_search_with_score(query, k=len(documents))
        
        input_results = []
        for result, score in sorted(results, key=lambda x: x[1], reverse=True):
            if args.group_type == GroupType.UR_PR.value:
                # For ur_pr: extract PR metadata
                pr_id = result.metadata.get('pr_id', None)
                sim_score = score
                create_date = result.metadata.get('create_date', None)
                pull_request_date = result.metadata.get('pull_request_date', None)
                latest_release_date = result.metadata.get('latest_release_date', None)
                pull_request = result.page_content
                url = result.metadata.get('url', None)
                author = result.metadata.get('author', None)
                target = result.metadata.get('target', None)
                input_results.append((
                    group_id,  # ur_id
                    pr_id,
                    sim_score,
                    create_date,
                    pull_request_date,
                    latest_release_date,
                    user_review,
                    pull_request,
                    url,
                    author,
                    target)
                )
            else:
                # For pr_ur: extract UR metadata
                ur_id = result.metadata.get('ur_id', None)
                sim_score = score
                create_date = result.metadata.get('create_date', None)
                pull_request_date = result.metadata.get('pull_request_date', None)
                latest_release_date = result.metadata.get('latest_release_date', None)
                user_review = result.page_content
                url = result.metadata.get('url', None)
                author = result.metadata.get('author', None)
                target = result.metadata.get('target', None)
                input_results.append((
                    ur_id,
                    group_id,  # pr_id
                    sim_score,
                    create_date,
                    pull_request_date,
                    latest_release_date,
                    user_review,
                    pull_request,
                    url,
                    author,
                    target)
                )

        reranked_results, in_tok, out_tok, warnings = results_with_gpt(df, query, input_results, args.top_k, args.prompt_type, args.group_type, args.prompt_with_relevance, args.use_gpt5)
        total_prompt_tokens += in_tok
        total_completion_tokens += out_tok

        for warning in warnings:
            if warning["code"] == "no_ids_returned":
                print(f"[warning] Group {group_id}: LLM returned no IDs.")
            elif warning["code"] == "invalid_ids_returned":
                invalid_list = ", ".join(map(str, warning.get("details", [])))
                print(f"[warning] Group {group_id}: LLM returned IDs not in candidates -> {invalid_list}")
            elif warning["code"] == "no_results_after_processing":
                print(f"[warning] Group {group_id}: No reranked entries remained after processing.")

        group_df = results_to_df_llm(reranked_results, args.group_type)
        if not group_df.empty:
            group_df.to_csv(
                csv_path,
                mode='a',
                header=not csv_header_written,
                index=False
            )
            csv_header_written = True

            existing_records = []
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_records = json.load(f)
                except json.JSONDecodeError:
                    existing_records = []
            existing_records.extend(group_df.to_dict('records'))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=2)

            if normalized_group_id is not None:
                processed_groups.add(normalized_group_id)
        else:
            if not warnings:
                print(f"No reranked results produced for group {group_id}.")
        
        if os.path.exists(faiss_index_dir):
            shutil.rmtree(faiss_index_dir)

    if not os.path.exists(csv_path):
        print("No results were written; exiting without computing metrics.")
        sys.exit(0)

    results_df = pd.read_csv(csv_path)
    if results_df.empty:
        print("Results file is empty; exiting without computing metrics.")
        sys.exit(0)

    if args.prompt_with_relevance and 'relevance' in results_df.columns:
        results_df = results_df.loc[results_df['relevance'].isin(['high', 'medium'])]

    results_df = results_df.sort_values(by=[group_key, 'rank']).reset_index(drop=True)

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict('records'), f, ensure_ascii=False, indent=2)

    print(f"Results saved to {csv_path} and {json_path}")

    metrics = None  
    if args.prompt_type == PromptType.IDENTIFY.value:      
        recall = recall(df, results_df)
        print("  Final test recall %f" % (recall))
        precision = precision(results_df)
        print("  Final test precision %f" % (precision))
        F1_score = F1_score(precision, recall)
        print("  Final test F1_score %f" % (F1_score))
        
        metrics = {
            "Recall": float(recall),
            "Precision": float(precision),
            "F1_score": float(F1_score),
        }

    else:
        print("begin calculate metrics")
        Hit = Hit_at_K(results_df, 1, args.group_type)
        print("  Final test Hit@1 %f" % (Hit) )
        Hit10 = Hit_at_K(results_df, 10, args.group_type)
        print("  Final test Hit@10 %f" % (Hit10) )
        precision = precision_at_K(results_df, 1, args.group_type)
        print("  Final test precision@1 %f" % (precision) )  
        precision10 = precision_at_K(results_df, 10, args.group_type)
        print("  Final test precision@10 %f" % (precision10))
        recall10 = recall_at_K(results_df, 10, args.group_type)
        print("  Final test recall@10 %f" % (recall10))
        mrr = MRR(results_df, args.group_type)
        print("  Final test MRR %f" % (mrr)) 
        ngcg = NDCG_at_K(results_df, k=1, group_type=args.group_type)
        print("  Final test ndcg@1 %f" % (ngcg))
        ngcg10 = NDCG_at_K(results_df, k=10, group_type=args.group_type)
        print("  Final test ndcg@10 %f" % (ngcg10)) 

        metrics = {
            "Hit@1": float(Hit),
            "Hit@10": float(Hit10),
            "Precision@1": float(precision),
            "Precision@10": float(precision10),
            "Recall@10": float(recall10),
            "MRR": float(mrr),
            "NDCG@1": float(ngcg),
            "NDCG@10": float(ngcg10),
        }
    
    runtime_seconds = time.time() - run_start

    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
        except json.JSONDecodeError:
            existing_summary = {}

    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "csv_file": args.csv_file,
        "index_dir": args.index_dir,
        "output_result_path": args.output_result_path,
        "embedding_selected": args.embedding_selected,
        "num_rows": int(len(results_df)),
        "num_ur_ids": int(df["ur_id"].nunique()) if "ur_id" in df.columns else None,
        "runtime_seconds": round(existing_summary.get("runtime_seconds", 0) + runtime_seconds, 3),
        "metrics": metrics,
        "gpt_prompt_tokens": int(existing_summary.get("gpt_prompt_tokens", 0) + total_prompt_tokens),
        "gpt_completion_tokens": int(existing_summary.get("gpt_completion_tokens", 0) + total_completion_tokens),
        "gpt_total_tokens": int(existing_summary.get("gpt_total_tokens", 0) + total_prompt_tokens + total_completion_tokens),
        "preprocessing": {
            "prompt_type": args.prompt_type,
            "prompt_with_relevance": args.prompt_with_relevance,
            "use_gpt5": args.use_gpt5,
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved processing summary to {summary_path}")
    

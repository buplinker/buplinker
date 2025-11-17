#!/bin/bash

# ----- Embedding -----
EMBEDDING="HuggingFace-minilm"
# EMBEDDING="HuggingFace-mpnet"
# EMBEDDING="OpenAIEmbeddings"


echo "Starting process..."
echo "Using embedding: $EMBEDDING"

# ===== Test Ground Truth =====
BASE_PATH_GROUP1="../data/random"
NAMES_GROUP1=(random)

# ===== Ground Truth =====
BASE_PATH_GROUP2="../data/all"
NAMES_GROUP2=(1_aeharding.voyager 2_commons-app.apps-android-commons 4_crawl.crawl 5_d4rken-org.sdmaid-se)

BASE_PATH_GROUP3="../data/all"
NAMES_GROUP3=(6_element-hq.element-android 7_element-hq.element-x-android 8_fossasia.pstab-app 9_horizontalsystems.unstoppable-wallet-android)

BASE_PATH_GROUP4="../data/all"
NAMES_GROUP4=(10_hydgard.ppsspp 11_immich-app.immich 12_jellyfin.jellyfin-androidtv 13_Jigsaw-Code.outline-apps)

BASE_PATH_GROUP5="../data/all"
NAMES_GROUP5=(15_johannesjo.super-productivity 16_laurent22.joplin 17_LemmyNet.jerboa 18_luanti-org.luanti)

BASE_PATH_GROUP6="../data/all"
NAMES_GROUP6=(19_mullvad.mullvadvpn-app 20_nextcloud.talk-android 21_organicmaps.organicmaps 23_owntracks.android)

BASE_PATH_GROUP7="../data/all"
NAMES_GROUP7=(25_simplex-chat.simplex-chat 27_streetwriters.notesnook 28_tasks.tasks 30_xbmc.xbmc)


PROMPT_TYPE="identify" # "rerank" or "identify"

# ===== Prompt With Reason =====
PROMPT_WITH_RELEVANCE=true

USE_GPT5=false
GROUP_TYPE="pr_ur"
TOP_K=5

run_group () {
  local BASE_PATH="$1"
  shift 1
  local NAMES=("$@")

  for DATA_NAME in "${NAMES[@]}"; do
    TEST_RAW_DATA_PATH="${BASE_PATH}/${GROUP_TYPE}/${DATA_NAME}_ground_truth.csv"

    echo "--------------------------------------"
    echo "Processing dataset: $DATA_NAME"
    echo "Group type: $GROUP_TYPE"
    echo "Data path: $TEST_RAW_DATA_PATH"
    echo "Prompt Type: $PROMPT_TYPE"
    echo "Prompt With Relevance: $PROMPT_WITH_RELEVANCE"
    echo "Use GPT-5: $USE_GPT5"

    if [ "$PROMPT_TYPE" = "rerank" -o "$PROMPT_TYPE" = "identify" ]; then
      if [ "$PROMPT_WITH_RELEVANCE" = true ]; then
        if [ "$USE_GPT5" = true ]; then
          PREPROCESS_SUFFIX="prompt_${PROMPT_TYPE}_with_reason_gpt5"
        else
          PREPROCESS_SUFFIX="prompt_${PROMPT_TYPE}_with_relevance"
        fi
      else
        PREPROCESS_SUFFIX="prompt_${PROMPT_TYPE}"
      fi
    else
      PREPROCESS_SUFFIX="original"
    fi

    INDEX_DIR="./indexs/index_${DATA_NAME}_${EMBEDDING}"
    
    # グループタイプに応じてパスを設定
    if [[ "$BASE_PATH" == *"test"* ]]; then
      RESULT_PATH="./${GROUP_TYPE}/test_results/${EMBEDDING}/${PREPROCESS_SUFFIX}/${DATA_NAME}_result_${EMBEDDING}_buplinker_${TOP_K}"
      LOG_PATH="./${GROUP_TYPE}/test_logs/${EMBEDDING}/${PREPROCESS_SUFFIX}/${DATA_NAME}_output_${EMBEDDING}_buplinker_${TOP_K}.log"
    else
      RESULT_PATH="./${GROUP_TYPE}/results/${EMBEDDING}/${PREPROCESS_SUFFIX}/${DATA_NAME}_result_${EMBEDDING}_buplinker_${TOP_K}"
      LOG_PATH="./${GROUP_TYPE}/logs/${EMBEDDING}/${PREPROCESS_SUFFIX}/${DATA_NAME}_output_${EMBEDDING}_buplinker_${TOP_K}.log"
    fi

    mkdir -p "$(dirname "$INDEX_DIR")"
    mkdir -p "$(dirname "$RESULT_PATH")"
    mkdir -p "$(dirname "$LOG_PATH")"


    # Run
    if [ "$PROMPT_WITH_RELEVANCE" = true ]; then
      if [ "$USE_GPT5" = true ]; then
        python buplinker.py \
          --group_type "$GROUP_TYPE" \
          --csv_file "$TEST_RAW_DATA_PATH" \
          --index_dir "$INDEX_DIR" \
          --output_result_path "$RESULT_PATH" \
          --embedding_selected "$EMBEDDING" \
          --top_k "$TOP_K" \
          --prompt_type "$PROMPT_TYPE" \
          --prompt_with_relevance \
          --use_gpt5 \
          2>&1 | tee "$LOG_PATH"
      else
        python buplinker.py \
          --group_type "$GROUP_TYPE" \
          --csv_file "$TEST_RAW_DATA_PATH" \
          --index_dir "$INDEX_DIR" \
          --output_result_path "$RESULT_PATH" \
          --embedding_selected "$EMBEDDING" \
          --top_k "$TOP_K" \
          --prompt_type "$PROMPT_TYPE" \
          --prompt_with_relevance \
          2>&1 | tee "$LOG_PATH"
        fi
    else
      python buplinker.py \
        --group_type "$GROUP_TYPE" \
        --csv_file "$TEST_RAW_DATA_PATH" \
        --index_dir "$INDEX_DIR" \
        --output_result_path "$RESULT_PATH" \
        --embedding_selected "$EMBEDDING" \
        --top_k "$TOP_K" \
        --prompt_type "$PROMPT_TYPE" \
        2>&1 | tee "$LOG_PATH"
    fi

    # リポジトリの処理が完了したら、インデックスディレクトリ全体をクリーンアップ
    if [ -d "$INDEX_DIR" ]; then
      echo "Cleaning up index directory: $INDEX_DIR"
      rm -rf "$INDEX_DIR"
      echo "Index directory cleaned up for $DATA_NAME"
    fi

    echo "DONE: $DATA_NAME with $EMBEDDING"
  done
}

# Run both groups
run_group "$BASE_PATH_GROUP1" "${NAMES_GROUP1[@]}"
# run_group "$BASE_PATH_GROUP2" "${NAMES_GROUP2[@]}"
# run_group "$BASE_PATH_GROUP3" "${NAMES_GROUP3[@]}"
# run_group "$BASE_PATH_GROUP4" "${NAMES_GROUP4[@]}"
# run_group "$BASE_PATH_GROUP5" "${NAMES_GROUP5[@]}"
# run_group "$BASE_PATH_GROUP6" "${NAMES_GROUP6[@]}"
# run_group "$BASE_PATH_GROUP7" "${NAMES_GROUP7[@]}"

echo "--------------------------------------"
echo "All datasets processed."

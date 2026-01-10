#!/bin/bash

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting process..."

TOP_K=5
LIMITED=false # trueはsaner用

if [ "$LIMITED" = "true" ]; then
  BASE_PATH="limited_years"
else
  BASE_PATH="all_years"
fi

# ===== Input Pairs =====
NAMES_GROUP1=(1_aeharding.voyager 2_commons-app.apps-android-commons 4_crawl.crawl 5_d4rken-org.sdmaid-se)
NAMES_GROUP2=(6_element-hq.element-android 7_element-hq.element-x-android 8_fossasia.pslab-app 9_horizontalsystems.unstoppable-wallet-android 10_hrydgard.ppsspp)
NAMES_GROUP3=(11_immich-app.immich 12_jellyfin.jellyfin-androidtv 13_Jigsaw-Code.outline-apps 15_johannesjo.super-productivity)
NAMES_GROUP4=(16_laurent22.joplin 17_LemmyNet.jerboa 18_luanti-org.luanti 19_mullvad.mullvadvpn-app 20_nextcloud.talk-android)
NAMES_GROUP5=(21_organicmaps.organicmaps 22_owncloud.android 23_owntracks.android 24_persian-calendar.persian-calendar 25_simplex-chat.simplex-chat)
NAMES_GROUP6=(26_standardnotes.app 27_streetwriters.notesnook 28_tasks.tasks 29_wireapp.wire-android 30_xbmc.xbmc)
NAMES_GROUP7=(31_zulip.zulip-flutter)

run_group () {
  local GROUP_TYPE="$1"
  shift 1
  local NAMES=("$@")

  for DATA_NAME in "${NAMES[@]}"; do
    INPUT_PAIRS_DATA_PATH="${SCRIPT_DIR}/../dataset/input_pairs/${GROUP_TYPE}/${BASE_PATH}/${DATA_NAME}_input_pairs.csv"

    echo "--------------------------------------"
    echo "Processing dataset: $DATA_NAME"
    echo "Group type: $GROUP_TYPE"
    echo "Data path: $INPUT_PAIRS_DATA_PATH"

    INDEX_DIR="${SCRIPT_DIR}/output/indexs/${BASE_PATH}/index_${DATA_NAME}"
    RESULT_PATH="${SCRIPT_DIR}/output/${GROUP_TYPE}/${BASE_PATH}/results/${DATA_NAME}_result_buplinker_${TOP_K}"
    LOG_PATH="${SCRIPT_DIR}/output/${GROUP_TYPE}/${BASE_PATH}/logs/${DATA_NAME}_output_buplinker_${TOP_K}.log"

    mkdir -p "$(dirname "$INDEX_DIR")"
    mkdir -p "$(dirname "$RESULT_PATH")"
    mkdir -p "$(dirname "$LOG_PATH")"

    # Run
    python ${SCRIPT_DIR}/buplinker.py \
      --group_type "$GROUP_TYPE" \
      --csv_file "$INPUT_PAIRS_DATA_PATH" \
      --index_dir "$INDEX_DIR" \
      --output_result_path "$RESULT_PATH" \
      --top_k "$TOP_K" \
      2>&1 | tee "$LOG_PATH"

    # リポジトリの処理が完了したら、インデックスディレクトリ全体をクリーンアップ
    if [ -d "$INDEX_DIR" ]; then
      echo "Cleaning up index directory: $INDEX_DIR"
      rm -rf "$INDEX_DIR"
      echo "Index directory cleaned up for $DATA_NAME"
    fi

    echo "DONE: $DATA_NAME with $EMBEDDING"
  done
}

# Run both groups for each GROUP_TYPE
for GROUP_TYPE in "ur_pr" "pr_ur"; do
  run_group "$GROUP_TYPE" "${NAMES_GROUP1[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP2[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP3[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP4[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP5[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP6[@]}"
  run_group "$GROUP_TYPE" "${NAMES_GROUP7[@]}"
done

echo "--------------------------------------"
echo "All datasets processed."
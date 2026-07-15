#!/usr/bin/env bash
# RoboCasa365 Lifelong Learning 数据集下载脚本 (Ubuntu / Linux)
#
# 按 4 个阶段下载终身学习所需的训练数据。
# 数据来源: pretrain 分割, source = human + mimicgen (对应 dataset_soup 的 source="all")
# 需先 pip install -e . 安装好 robocasa 并激活对应 conda 环境后运行。
#
# 用法:
#   chmod +x download_datasets.sh
#   ./download_datasets.sh 1        # 下载阶段 1
#   ./download_datasets.sh all      # 下载全部 4 个阶段

set -euo pipefail

PHASE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_FILE="${SCRIPT_DIR}/phase_tasks.json"

case "${PHASE}" in
    1)   PHASES=("lifelong_learning_phase1") ;;
    2)   PHASES=("lifelong_learning_phase2") ;;
    3)   PHASES=("lifelong_learning_phase3") ;;
    4)   PHASES=("lifelong_learning_phase4") ;;
    all|ALL)
        PHASES=(
            "lifelong_learning_phase1"
            "lifelong_learning_phase2"
            "lifelong_learning_phase3"
            "lifelong_learning_phase4"
        )
        ;;
    *)
        echo "用法: $0 {1|2|3|4|all}"
        echo "示例: $0 1     # 下载阶段 1"
        echo "      $0 all   # 下载全部 4 个阶段"
        exit 1
        ;;
esac

if [[ ! -f "${TASKS_FILE}" ]]; then
    echo "错误: 找不到 phase_tasks.json 文件: ${TASKS_FILE}" >&2
    exit 1
fi

# 从 JSON 中提取指定阶段的任务名
get_tasks() {
    local phase_name="$1"
    python3 -c "
import json, sys
with open('${TASKS_FILE}', 'r') as f:
    data = json.load(f)
tasks = data.get('${phase_name}', [])
if not tasks:
    print('错误: 阶段 ${phase_name} 未找到任务列表', file=sys.stderr)
    sys.exit(1)
print(' '.join(tasks))
"
}

download_phase() {
    local phase_name="$1"
    local task_list
    task_list="$(get_tasks "${phase_name}")"
    local task_count
    task_count="$(echo "${task_list}" | wc -w)"

    echo ""
    echo "========================================"
    echo "  下载 ${phase_name} (${task_count} 个任务)"
    echo "  source: human + mimicgen"
    echo "  split: pretrain"
    echo "========================================"
    echo ""

    # source=all 对应同时下载 human 和 mimicgen
    # split=pretrain 对应预训练数据
    # shellcheck disable=SC2086  # 需要单词分割以展开任务列表
    python -m robocasa.scripts.download_datasets \
        --tasks ${task_list} \
        --source human mimicgen \
        --split pretrain

    echo ""
    echo "${phase_name} 下载完成"
}

for phase in "${PHASES[@]}"; do
    download_phase "${phase}"
done

echo ""
echo "全部完成。数据默认存放在 robocasa/datasets/ 目录下。"
echo "可用以下命令验证:"
echo "  python robocasa/scripts/dataset_scripts/get_dataset_info.py --dataset <ds-path>"

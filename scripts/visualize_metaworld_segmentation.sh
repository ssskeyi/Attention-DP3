#!/usr/bin/env bash
set -euo pipefail

# 可视化Metaworld GS2分割结果
# 将结果保存在项目根目录的vis_res_metaworld文件夹下

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
VIS_ROOT="${VIS_ROOT:-${ROOT}/vis_res_metaworld}"

# 实验配置（与make_metaworld_datasets.sh保持一致）
TASKS="${TASKS:-hammer pick-place window-open window-close sweep sweep-into stick-push stick-pull soccer shelf-place box-close bin-picking disassemble reach}"
MAX_EP="${MAX_EP:-10}"

log() { echo -e "[visualize_metaworld] $*"; }

# 创建可视化输出目录
create_dirs() {
    mkdir -p "${VIS_ROOT}"
    log "可视化结果将保存到: ${VIS_ROOT}"
}

# 可视化GS2分割结果
visualize_gs2_seg() {
    log "=== 开始可视化Metaworld GS2分割结果 ==="

    for task in ${TASKS}; do
        log "可视化 ${task} 的GS2分割..."

        gs2_root="${ROOT}/3D-Diffusion-Policy/export_gs2/metaworld_${task}"
        frames_root="${ROOT}/3D-Diffusion-Policy/export/metaworld_${task}_frames"

        # 检查目录是否存在
        if [[ ! -d "$gs2_root" ]]; then
            log "警告: GS2结果目录不存在: $gs2_root，跳过 ${task}"
            continue
        fi

        if [[ ! -d "$frames_root" ]]; then
            log "警告: 帧目录不存在: $frames_root，跳过 ${task}"
            continue
        fi

        python "${ROOT}/scripts/visualize_gs2_results.py" \
            --frames_root "$frames_root" \
            --gs2_root "$gs2_root" \
            --output_root "${VIS_ROOT}/gs2_seg_${task}" \
            --score_thr 0.3 \
            --alpha 0.5

        log "${task} GS2分割可视化完成"
    done
}

# 可视化原始帧（可选，用于对比）
visualize_raw_frames() {
    log "=== 开始可视化Metaworld原始帧 ==="

    for task in ${TASKS}; do
        log "可视化 ${task} 的原始帧..."

        frames_root="${ROOT}/3D-Diffusion-Policy/export/metaworld_${task}_frames"

        if [[ ! -d "$frames_root" ]]; then
            log "警告: 帧目录不存在: $frames_root，跳过 ${task}"
            continue
        fi

        python "${ROOT}/scripts/visualize_raw_frames.py" \
            --frames_root "$frames_root" \
            --output_root "${VIS_ROOT}/raw_frames_${task}" \
            --max_episodes "${MAX_EP}"

        log "${task} 原始帧可视化完成"
    done
}

# 主函数
main() {
    log "开始可视化Metaworld分割实验结果"
    log "配置:"
    log "  ROOT=${ROOT}"
    log "  VIS_ROOT=${VIS_ROOT}"
    log "  TASKS=${TASKS}"
    log "  MAX_EP=${MAX_EP}"

    create_dirs
    visualize_gs2_seg
    visualize_raw_frames

    log "所有可视化完成！"
    log "结果保存在: ${VIS_ROOT}"
    log ""
    log "可视化结果结构:"
    log "  ${VIS_ROOT}/"
    for task in ${TASKS}; do
        log "  ├── gs2_seg_${task}/     # ${task} GS2分割可视化"
        log "  └── raw_frames_${task}/  # ${task} 原始帧可视化"
    done
}

main "$@"

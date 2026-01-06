#!/usr/bin/env bash
set -euo pipefail

# 可视化GS2和env分割结果
# 将结果保存在项目根目录的vis_res文件夹下

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
VIS_ROOT="${VIS_ROOT:-${ROOT}/vis_res}"

# 实验配置（与run_adroit_segmentation_experiment.sh保持一致）
TASKS="${TASKS:-pen door hammer}"
SEG_TYPES="${SEG_TYPES:-env gs2}"
MAX_EP="${MAX_EP:-10}"

log() { echo -e "[visualize] $*"; }

# 创建可视化输出目录
create_dirs() {
    mkdir -p "${VIS_ROOT}"
    log "可视化结果将保存到: ${VIS_ROOT}"
}

# 可视化环境分割结果
visualize_env_seg() {
    log "=== 开始可视化环境分割结果 ==="

    for task in ${TASKS}; do
        log "可视化 ${task} 的环境分割..."

        # env分割可视化（使用targets mask可视化）
        python "${ROOT}/tools/visualize_targets_mask.py" \
            --task "${task}" \
            --zarr_path "${ROOT}/3D-Diffusion-Policy/data/adroit_${task}_expert_env.zarr" \
            --output_dir "${VIS_ROOT}/env_seg_${task}" \
            --max_episodes 5 \
            --frames_per_ep 10

        log "${task} 环境分割可视化完成"
    done
}

# 可视化GS2分割结果
visualize_gs2_seg() {
    log "=== 开始可视化GS2分割结果 ==="

    for task in ${TASKS}; do
        log "可视化 ${task} 的GS2分割..."

        gs2_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_gs2"
        frames_root="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_gs2_frames"

        python "${ROOT}/scripts/visualize_gs2_results.py" \
            --frames_root "$frames_root" \
            --gs2_root "$gs2_root" \
            --output_root "${VIS_ROOT}/gs2_seg_${task}" \
            --score_thr 0.3 \
            --alpha 0.5

        log "${task} GS2分割可视化完成"
    done
}


# 主函数
main() {
    log "开始可视化分割实验结果"
    log "配置:"
    log "  ROOT=${ROOT}"
    log "  VIS_ROOT=${VIS_ROOT}"
    log "  TASKS=${TASKS}"
    log "  SEG_TYPES=${SEG_TYPES}"

    create_dirs
    visualize_env_seg
    visualize_gs2_seg

    log "所有可视化完成！"
    log "结果保存在: ${VIS_ROOT}"
    log ""
    log "可视化结果结构:"
    log "  ${VIS_ROOT}/"
    log "  ├── env_seg_pen/     # pen环境分割可视化 (targets mask)"
    log "  ├── env_seg_door/    # door环境分割可视化 (targets mask)"
    log "  ├── env_seg_hammer/  # hammer环境分割可视化 (targets mask)"
    log "  ├── gs2_seg_pen/     # pen GS2分割可视化"
    log "  ├── gs2_seg_door/    # door GS2分割可视化"
    log "  └── gs2_seg_hammer/  # hammer GS2分割可视化"
}

main "$@"

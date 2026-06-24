#!/bin/bash

# ==============================================================
# Task-Conditioned Diffusion 自动化评估脚本
# ==============================================================

CUDA_ID=0
# 👇 请确保这里的名字与你运行训练时指定的 --exp 和 --norm 一致
EXP_NAME="satc_ttacd_la02_instancenorm" 
CKPT_DIR="logs/${EXP_NAME}/ckpts"

# 临时工作目录（每个任务独立，避免覆盖）
BASE_EVAL_DIR="logs/eval_temp_taskcond"

# 结果保存目录
RESULT_DIR="logs/${EXP_NAME}/eval_results"
mkdir -p ${RESULT_DIR}

SUMMARY_FILE="${RESULT_DIR}/TaskCond_Final_Scores.txt"
echo -e "Task\tTarget Split\tAvg Dice" > $SUMMARY_FILE

# 获取当前目录
CURRENT_DIR=$(pwd)

evaluate_task() {

    local task=$1
    local code_dir=$2
    local eval_script=$3
    local split=$4
    local extra_eval_args=$5

    echo "======================================================"
    echo "🚀 开始自动化评估 Task-Conditioned 任务: $task"
    echo "======================================================"
    
    # 1. 定位该任务专属的 Best Model
    local ckpt_path="${CKPT_DIR}/best_model_${task}.pth"
    local eval_dir="${BASE_EVAL_DIR}_${task}"
    local full_eval_path="logs/${eval_dir}"
    
    if [ ! -f "$ckpt_path" ]; then
        echo "❌ 警告: 找不到权重文件 $ckpt_path, 跳过..."
        return
    fi
    
    echo ">>> 正在加载权重: $ckpt_path"
    
    # 清空并创建临时目录
    rm -rf ${full_eval_path}
    mkdir -p ${full_eval_path}/ckpts
    mkdir -p ${full_eval_path}/predictions
    
    # 复制权重
    cp "$ckpt_path" "${full_eval_path}/ckpts/best_model.pth"
    echo "✅ 权重已复制到: ${full_eval_path}/ckpts/best_model.pth"
    
    # 2. 调用测试脚本 (注入 task_id)
    echo ">>> 运行推理..."
    python ${code_dir}/test_Taskcond.py \
        --exp ${eval_dir} \
        -g ${CUDA_ID} \
        --split ${split} \
        -t ${task} \
        --norm instancenorm
    
    # 检查预测文件
    if [ ! -d "${full_eval_path}/predictions" ] || [ -z "$(ls -A ${full_eval_path}/predictions 2>/dev/null)" ]; then
        echo "❌ 错误: 预测文件未生成！"
        return
    fi
    echo "✅ 预测文件已生成: $(ls ${full_eval_path}/predictions | wc -l) 个文件"
    
    # 创建评估所需的目录结构
    mkdir -p ${full_eval_path}/fold1/predictions
    cp ${full_eval_path}/predictions/* ${full_eval_path}/fold1/predictions/ 2>/dev/null
    
    # 3. 运行评估计算最终指标
    echo "📊 计算最终指标..."
    local result_file="${RESULT_DIR}/${task}_result.txt"
    
    python ${code_dir}/${eval_script} \
        --exp ${eval_dir} \
        --folds 1 \
        --split ${split} \
        -t ${task} ${extra_eval_args} | tee ${result_file}
    
    # 提取Dice分数
    AVG_DICE=$(grep "Final Avg Dice" ${result_file} | awk -F': ' '{print $2}' | awk -F'±' '{print $1}')
    echo "✅ ${task} 评估完成! Avg Dice: ${AVG_DICE}%"
    echo -e "$task\t$split\t$AVG_DICE" >> $SUMMARY_FILE
    echo "======================================================"
    echo ""
}

# ==========================================
# 依次执行评估
# ==========================================

echo "======================================================"
echo "Task-Conditioned Diffusion 模型评估"
echo "实验名称: ${EXP_NAME}"
echo "权重目录: ${CKPT_DIR}"
echo "======================================================"

# 评估 LA 任务
evaluate_task "la_10" "code1" "evaluate_la.py" "test" ""

# 评估 MMWHS 任务
evaluate_task "mmwhs_mr2ct" "code1" "evaluate.py" "test_ct" "--modality CT"

# 评估 MNMS 任务
evaluate_task "mnms_toB_10" "code1" "evaluate.py" "test_toB_0.05" ""

echo ""
echo "======================================================"
echo "🎉 所有评估完成！汇总结果:"
echo "======================================================"
cat $SUMMARY_FILE
echo ""
echo "详细结果保存在: ${RESULT_DIR}"
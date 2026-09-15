# SteamInMicrowave 官方站位扰动评测

这套评测把官方复合任务的 Pick & Place 演示拆成独立的 PickObject 和
PlaceObject 策略调用。导航不交给 π0.5：程序先恢复官方 MuJoCo 场景与状态，
提取官方操作站位，再通过底盘真实动作产生有误差但物理有效的起始状态。baseline
和 refiner 都从同一个保存状态开始，因此可以直接比较多视角 VLM 站位优化的收益。

## 数据约定

默认数据目录：

    datasets/v1.0/pretrain/composite/SteamInMicrowave/20250714/lerobot

程序读取官方的 annotation.human.subtask、subtask_name、subtask_stage 和
subtask_idx，提取每条演示中的四个操作入口：

1. 从水槽 Pick 蔬菜；
2. 向碗中 Place 蔬菜；
3. 从台面 Pick 碗；
4. 向微波炉中 Place 碗。

原始数据、提取状态、扰动状态、图片和结果均被 .gitignore 排除，不会上传到
GitHub。不要把 API key 写入脚本或配置文件。

## 1. 提取官方操作站位

先用一个 episode 验证：

    /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.extract_expert_work_poses \
      --episodes 0

提取前 20 条演示：

    /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.extract_expert_work_poses \
      --episode_start 0 \
      --episode_count 20

将 --episode_count 设为 0 会处理全部 episode。索引默认写入：

    robocasa/outputs_work_pose_benchmark/SteamInMicrowave/expert/index.json

## 2. 生成物理有效的差站位

先只生成确定性的扰动规格，不启动 MuJoCo：

    /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.generate_degraded_work_poses \
      --difficulties mild moderate severe \
      --samples_per_stage 3 \
      --specs_only

正式物化状态：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.generate_degraded_work_poses \
      --difficulties mild moderate severe \
      --samples_per_stage 3 \
      --max_candidate_attempts 5

扰动以官方机器人坐标系采样：

- mild：平移 0.10–0.20 m，旋转 5–10°；
- moderate：平移 0.20–0.40 m，旋转 10–25°；
- severe：平移 0.40–0.60 m，旋转 25–40°。
- stress：平移 0.60–0.90 m，旋转 40–60°，用于验证 VLM 是否真的能控制底盘。

底盘必须通过环境原生 12 维动作真正到达目标，不能直接篡改位姿。Place 样本还
必须在移动前后持续夹住目标物体。不可达、碰撞受限或掉落的候选写入
perturbations.json 的 rejected，只有通过检查的状态进入 samples。

## 3. 跑 baseline

baseline 不调用站位 VLM，并默认冻结 π0.5 的底盘动作，只衡量当前差站位下的
Pick/Place 成功率：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_refinement \
      --condition baseline \
      --pi05_host 172.16.36.10 \
      --pi05_port 8000 \
      --pi05_horizon 600

## 4. 跑多视角站位优化

    export VLM_API_KEY='your-key'

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_refinement \
      --condition refiner \
      --local_pose_base_url http://172.16.11.115:11434/v1 \
      --local_pose_model qwen2.5vl:3b \
      --local_pose_api_key "$VLM_API_KEY" \
      --local_pose_cameras \
        robot0_topview \
        robot0_frontview \
        robot0_agentview_left \
        robot0_agentview_right \
        robot0_eye_in_hand \
      --pi05_host 172.16.36.10 \
      --pi05_port 8000 \
      --pi05_horizon 600

初步验证 VLM 是否真的能控制底盘时，建议先只跑一个 mild Pick，并显式使用
10 cm 的非持物修正：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_refinement \
      --condition refiner \
      --operation pick \
      --difficulty mild \
      --limit 1 \
      --local_pose_base_url http://172.16.11.115:11434/v1 \
      --local_pose_model qwen2.5vl:3b \
      --local_pose_api_key ollama \
      --local_pose_translation_distance_m 0.10 \
      --local_pose_rotation_angle_deg 8 \
      --local_pose_held_translation_distance_m 0.01 \
      --local_pose_held_rotation_angle_deg 1 \
      --pi05_host 172.16.11.115 \
      --pi05_port 8000 \
      --pi05_horizon 600 \
      --video

VLM 现在输出 distance、lateral、yaw 三轴对齐判断，程序按 yaw、lateral、
distance 的优先级映射成离散动作。每个动作以真实底盘位姿闭环到 10 cm 或 8°；
Place 等持物阶段自动缩小为 1 cm 或 1°。无效 STOP 最多重新询问 2 次。
result.json 会保存 requested/actual 位移、目标/最终位姿、收敛状态和逐步 trace。

旧版官方 model.xml.gz 没有顶部和前方相机时，恢复代码会把相机注入机器人底盘
或手腕对应 body，不修改原始数据文件。

### 大扰动底盘控制验证

不要覆盖已有的 mild/moderate/severe manifest。先单独生成 5 个非持物 Pick 的
stress 状态；物理物化会排除碰撞、不可达和状态恢复失败的候选：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.generate_degraded_work_poses \
      --operation pick \
      --difficulties stress \
      --samples_per_stage 5 \
      --max_candidate_attempts 10 \
      --max_move_steps 1200 \
      --output robocasa/outputs_work_pose_benchmark/SteamInMicrowave/degraded_stress_pick

在同一个 manifest 上先跑 baseline。保持 π0.5 底盘冻结，避免把操作策略自己的
底盘动作误算成站位优化效果：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_refinement \
      --manifest robocasa/outputs_work_pose_benchmark/SteamInMicrowave/degraded_stress_pick/perturbations.json \
      --output_root robocasa/outputs_work_pose_benchmark/SteamInMicrowave/evaluation_stress \
      --condition baseline \
      --operation pick \
      --difficulty stress \
      --limit 5 \
      --pi05_base_action_mode frozen \
      --pi05_host 172.16.11.115 \
      --pi05_port 8000 \
      --pi05_horizon 600 \
      --video

然后在完全相同的 5 个状态上运行 refiner。stress 最大可偏移 0.90 m，而默认累计
修正上限只有 0.30 m，因此这里显式提高最大决策次数和累计移动/旋转上限：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_refinement \
      --manifest robocasa/outputs_work_pose_benchmark/SteamInMicrowave/degraded_stress_pick/perturbations.json \
      --output_root robocasa/outputs_work_pose_benchmark/SteamInMicrowave/evaluation_stress \
      --condition refiner \
      --operation pick \
      --difficulty stress \
      --limit 5 \
      --local_pose_base_url http://172.16.11.115:11434/v1 \
      --local_pose_model qwen2.5vl:3b \
      --local_pose_api_key ollama \
      --local_pose_max_decisions 16 \
      --local_pose_translation_distance_m 0.10 \
      --local_pose_rotation_angle_deg 10 \
      --local_pose_max_total_translation_m 1.20 \
      --local_pose_max_total_rotation_deg 90 \
      --local_pose_max_invalid_stops 3 \
      --pi05_base_action_mode frozen \
      --pi05_host 172.16.11.115 \
      --pi05_port 8000 \
      --pi05_horizon 600 \
      --video

判断 VLM 底盘控制是否有效时，不应只比较最终操作成功率，还要配对比较同一
sample ID 的以下字段：

- `pose_error_before.translation_m` 与 `pose_error_after_refinement.translation_m`；
- `pose_error_before.yaw_deg` 与 `pose_error_after_refinement.yaw_deg`；
- `num_executed_actions` 和各离散动作的 requested/actual 位移；
- baseline/refiner 的 Pick 成功率；
- `STOP`、无效决策和修正方向反复切换的比例。

建议先完成这个非持物 Pick 对照，再测试 Place。否则大扰动持物移动造成的掉落会
同时影响结果，无法单独说明 VLM 的底盘站位控制是否有效。

## 5. 公平比较

两种 condition 应使用同一个 perturbations.json，并保持以下参数一致：

- --sample_ids 或 --limit；
- --operation 和 --difficulty；
- π0.5 服务、模型、horizon、replan 与 verifier 参数；
- --pi05_base_action_mode frozen。

每个样本都会重新加载同一个 model.xml.gz 和同一个压缩 MuJoCo state。结果中的
pose_error_before、pose_error_after_refinement 和 success 分别衡量优化前误差、
优化后误差与操作成功。最终汇总保存在各次运行的 summary.json。

建议分别报告：

- Pick 与 Place 的成功率；
- mild、moderate、severe、stress 各难度成功率；
- 平移和角度误差下降量；
- VLM 输出 stay 的比例、有效底盘动作数和物体掉落率；
- 相同 sample ID 上 refiner 相对 baseline 的配对成功率变化。

如果只想快速验证单个 Place 样本，生成命令添加：

    --operation place --difficulties mild --limit 1

评测命令添加：

    --operation place --difficulty mild --limit 1

## 6. 连续运行前 5 个完整 episode

单阶段评测会为每个 Pick/Place 独立恢复状态，不能代表完整任务成功率。完整流程
评测只在 episode 开始时恢复一次状态，之后连续执行：

1. Pick 蔬菜；
2. Place 蔬菜到碗；
3. Pick 碗；
4. 使用数据集专家站位完成持物导航；
5. Place 碗到微波炉；
6. CloseMicrowave；
7. TurnOnMicrowave。

首先重新提取前 5 个 episode。该命令会重写 expert/index.json：

    /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.extract_expert_work_poses \
      --episode_start 0 \
      --episode_count 5
默认 VLA 收尾只使用提取出的专家底盘操作站位，不使用结束 EEF 位姿。提取器仍会
保存 expert_handoff_poses，供显式的 legacy
`--operation_completion_mode expert` 对照实验使用；普通 baseline/refiner
运行无需消费这些 EEF 位姿。


然后为 5 × 4 个 Pick/Place 阶段各生成一个 mild 扰动状态。不要添加 --limit 1：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.generate_degraded_work_poses \
      --difficulties mild \
      --samples_per_stage 1 \
      --max_candidate_attempts 5

清单应显示 requested_samples=20、num_valid=20。任何 episode 缺少阶段 0、1、2
或 4 时，完整流程入口会直接报错，不会用其他 episode 的样本补齐。

运行 5 个 baseline 完整 episode：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_full_episode \
      --condition baseline \
      --difficulty mild \
      --episode_start 0 \
      --episode_count 5 \
      --pi05_host 172.16.36.10 \
      --pi05_port 8000 \
      --pi05_horizon 600 \
      --pi05_task_retries 2 \
      --video

运行相同 5 个 episode 的多视角 refiner：

    MUJOCO_GL=egl /opt/conda/envs/robocasa/bin/python -m \
      robocasa.scripts.work_pose.eval_work_pose_full_episode \
      --condition refiner \
      --difficulty mild \
      --episode_start 0 \
      --episode_count 5 \
      --local_pose_base_url http://172.16.11.115:11434/v1 \
      --local_pose_model qwen2.5vl:3b \
      --pi05_host 172.16.36.10 \
      --pi05_port 8000 \
      --pi05_horizon 600 \
      --pi05_task_retries 2 \
      --video

`--pi05_task_retries 2` 表示每个 Pick、Place 或 fixture 操作最多执行 3 次
（首次执行加 2 次 retry）。只有 verifier 标记为 `retryable` 的策略失败才会
重新执行操作，并且从当前环境状态继续，不恢复 episode、不重复站位导航。

默认使用 `--operation_completion_mode vla`。Pick/Place 和 fixture Prompt 都会
明确要求 VLA 在完成操作后抬升或撤离到稳定姿态；Verifier 首次检测成功后，pi0.5
继续执行 50 步（`--pi05_success_handoff_steps 50`），而不是切换到专家 EEF
控制器。Pick 还必须同时满足持物和 EEF 离开来源 fixture，Place 继续要求 released
和 EEF 离开目标 receptacle。收尾期间 Held Object Guard 仍然有效，发生掉落会按
retryable 策略失败处理。若需要复现旧方案，可显式添加
`--operation_completion_mode expert`，该模式会关闭 VLA continuation 并读取
expert_handoff_poses。

操作在 retry 用尽或遇到不可重试失败后，或者数据集站位移动失败后，episode 才会
停止，并在 episode_result.json 的 stopped_at 中记录阶段。每个操作结果还会记录
num_attempts 和 attempt_results，episode 汇总记录 num_atomic_attempts。

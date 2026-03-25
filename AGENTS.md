# AGENTS 指南

## 仓库概览
- 本仓库包含多条医学影像扩散模型流水线（`ddpm`、`diffusion`、`ds_diff`、`ds_diff_gaussian`），不是单一应用。
- 训练与推理主逻辑在 `training_project/`、`trainers/`、`inference/`，共享扩散主干在 `ldm/` 与 `Disc_diff/`。
- 已扫描 AI 约定来源（`README.md`、`AGENTS*.md`、Copilot/Claude/Cursor/Windsurf 规则），当前未发现可继承的既有规范文件。

## 整体架构
- 入口脚本负责组装配置与 Lightning Trainer，再实例化 `trainers/` 中的训练器类：
  - `training_project/train_main_DS_diffusion.py` -> `trainers/trainer_ds_diff.py`（`DSDiffModel`）
  - `training_project/train_main_with_gaussian_diff.py` -> `trainers/trainer_use_gaussian_diff.py`（`TryTrainerDiffusion`）
  - `inference/inference_2d_with_gaussian_main.py` -> `trainer.predict(...)` + `inference/get_metric.py`
- `DSDiffModel` 继承 `ldm.models.diffusion.ddpm.DDPM`，并叠加解纠缠损失（`c-s`、`s-a-l`）与 MONAI 数据流程。
- `TryTrainerDiffusion` 将 OpenAI 风格扩散调度（`Disc_diff/guided_diffusion/*`）接入 `ldm.models.diffusion.ddpm.DiffusionWrapper`。
- 采样后端由配置切换（`ddim`/`dpm`），实现位于 `trainers/trainer_ds_diff.py::log_images`。

## 数据与张量约定
- 训练器与 transform 默认依赖 2D H5 数据目录约定：
  - 训练目录：`<h5_2d_img_dir>/images_tr_256/<case>/<slice>.h5`
  - 测试目录：`<h5_2d_img_dir>/images_ts_256/<case>/<slice>.h5`
- `training_project/training_transform.py::get_2d_train_transform_diff` 要求 `keys=config.train_keys`，其中：
  - `keys[:-1]` 会拼接为 `batch["image"]`（条件输入通道）
  - `keys[-1]` 在训练器中视为目标真值（ground truth）
- 启用 `use_edge` 时，会通过 `GetEdgeMap`（`training_project/utils/my_transform.py`）注入 `edge` 通道。

## 项目约定（非通用）
- 任务/运行目录命名必须保持：`{Task_name}_{Task_id}_{net_mode}_fold{fold_K}-{fold_idx}`；多个脚本按该格式推导 checkpoint 与指标路径。
- 断点续训默认匹配 `checkpoint*.ckpt`；最佳模型通常为 `best-{epoch}.ckpt`，监控指标为 `val/ssim`。
- 非 BraTs 数据集使用 `images_tr_256` 上的 KFold；BraTs 使用显式验证目录 `images_val` 或 `images_val_256`。
- 配置优先级常见模式：`configs/train_config.py`（argparse）-> YAML 加载 -> 入口脚本中可选 `OmegaConf.merge`。

## 关键工作流（真实脚本路径）
- 训练 DS-Diff：
  - `python training_project/train_main_DS_diffusion.py --config_file /abs/path/to/configs/train_config.yaml`
- 训练 Gaussian DS-Diff 变体：
  - `python training_project/train_main_with_gaussian_diff.py --config_file /abs/path/to/configs/train_config.yaml`
- 推理与指标导出：
  - `python inference/inference_2d_with_gaussian_main.py`
  - 指标表由 `inference/get_metric.py` 输出为 `pred_nii_*_metric.xlsx`

## 集成与环境注意事项
- 多数配置/脚本假设 NAS 风格绝对路径（`/nas_3/...`），并在运行时做前缀拼接；迁移环境时需同步更新 YAML 路径与脚本前缀逻辑。
- `trainers/*` 依赖 MONAI + Lightning + SimpleITK + 自定义 `ldm/Disc_diff` 内部模块；常见问题多由路径或 key 不匹配引起，而非模型本身。
- 指标流程 `inference/get_metric.py` 默认真值文件名为 `S_Data2.nii.gz`，预测文件后缀为 `_pred.nii.gz`。
- 仓库没有统一测试框架；最稳妥的验证方式是入口脚本里做一次短程 Lightning 冒烟运行（`limit_train_batches`、`limit_val_batches`）。


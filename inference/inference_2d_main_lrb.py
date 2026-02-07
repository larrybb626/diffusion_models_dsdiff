import os
import re
import sys
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from monai.utils import set_determinism
from omegaconf import OmegaConf

# 导入你训练时修改好的模型类
from trainers.trainer_ds_diff import DSDiffModel
import inference.get_metric as get_metric

if __name__ == "__main__":
    # 1. 加载配置
    config = OmegaConf.load("../configs/inference_config.yaml")

    # 2. 基础环境设置
    set_determinism(config["seed"])
    seed_everything(config["seed"], workers=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 3. 路径修正：直接指向你服务器上的真实路径
    # 考虑到之前的断电重启，建议这里写绝对路径或确保 dir_prefix 正确
    # dir_prefix = "/nas_3/LaiRuiBin/Changhai"

    # 4. 构建任务名称和路径
    # 注意：如果 Task_id 在 YAML 中是 146，这里读出来是整数，如果是 "0146"，读出来是字符串
    task_name = "{}_{}_{}_fold5-{}".format(config.Task_name, config.Task_id, config.net_mode, config.fold_idx)

    # 5. 寻找最优或指定 Checkpoint
    ckpt_dir = os.path.join(config.result_path, task_name, "checkpoint")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"找不到权重目录: {ckpt_dir}")

    ckpt_list = os.listdir(ckpt_dir)
    ckpt_name = config.ckpt_name

    # 自动匹配最新的 v 版本或 epoch 版本
    if ckpt_name == "best":
        pattern = r"best-epoch=(\d+)\.ckpt"
        matches = [re.search(pattern, f) for f in ckpt_list if re.match(pattern, f)]
        if not matches: raise FileNotFoundError("未找到 best 权重")
        sorted_ckpts = sorted(matches, key=lambda x: int(x.group(1)))
        ckpt_to_resume = sorted_ckpts[-1].group(0)
    else:
        # 匹配 checkpoint.ckpt 或 checkpoint-v1.ckpt 等
        pattern = r"{}(-v\d+)?\.ckpt".format(ckpt_name)
        matches = [f for f in ckpt_list if re.match(pattern, f)]
        if not matches: raise FileNotFoundError(f"未找到 {ckpt_name} 权重")
        # 如果有多个版本，取最新的一个
        ckpt_to_resume = sorted(matches)[-1]

    ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    print(f"正在加载权重: {ckpt_path}")

    # 6. 加载模型
    # 注意：由于训练时用了 DeepSpeed，load_from_checkpoint 会加载模型权重
    # 推理时我们不需要 DeepSpeed 策略，直接跑单卡即可
    model = DSDiffModel.load_from_checkpoint(ckpt_path, map_location="cpu")

    # 7. 更新模型参数（覆盖训练时的默认值）
    model.sampler_setting = config.sampler_setting
    model.test_batch_size = config.test_batch_size
    model.test_num = config.test_num

    # 设定推理结果输出目录
    model.pred_result_dir = os.path.join(result_path, task_name,
                                         f"pred_{config.sampler_setting.sampler}_{config.sampler_setting.sample_steps}")
    if not os.path.exists(model.pred_result_dir):
        os.makedirs(model.pred_result_dir)

    # 8. 初始化推理器
    # 即使之前是多卡 DDP 训练，推理只需单卡运行
    print(f"使用 GPU: {config.cuda_idx} 进行推理...")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[config.cuda_idx],  # 确保 config.cuda_idx 是 0 或 2
        enable_progress_bar=True,
    )

    # 9. 执行预测
    # 会自动调用 DSDiffModel 中的 predict_step 和 on_predict_end
    trainer.predict(model)

    # 10. 计算指标（PET-CT 的 MAE/SSIM 等）
    print('推理完成，正在计算指标...')
    get_metric.main()
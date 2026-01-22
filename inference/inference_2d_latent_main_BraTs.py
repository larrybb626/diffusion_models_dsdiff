import os.path
import re
import sys

from omegaconf import OmegaConf

from inference.test_param import config
import lightning.pytorch as pl
from trainers.trainer_latent_diffusion import LatentDiffusionModel
from monai.utils import set_determinism
from lightning.pytorch import seed_everything
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load("../configs/inference_config_BraTs.yaml")
    assert config.Task_name == "BraTs_synthesis"
    set_determinism(config["seed"])
    seed_everything(config["seed"], workers=True)
    # ==========path============
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name

    dir_prefix = sys.argv[0].split("/newnas")[0]
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, config.net_mode, fold_idx)
    result_path = config.result_path
    # ================search for best==============================
    ckpt_dir = os.path.join(result_path, task_name, "checkpoint")
    ckpt_list = os.listdir(ckpt_dir)
    if ckpt_name == "best":
        pattern = r"{}(-epoch=\d+)?\.ckpt".format(ckpt_name)
        ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
        versions = [re.search(r"epoch=(\d+)", file).group(1) for file in ckpt_file if re.search(r"epoch=\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"{ckpt_name.split('.')[0]}-epoch={sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name
    else:
        pattern = r"{}(-v\d+)?\.ckpt".format(ckpt_name)
        ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"{ckpt_name}-v{sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name + ".ckpt"

    ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    print(ckpt_to_resume)
    model = LatentDiffusionModel.load_from_checkpoint(ckpt_path, map_location="cuda:{}".format(int(config.cuda_idx)))
    # 下面是临时改变VAE的操作
    if config.vae_local_pretrained:
        vae_ckpt = os.path.join(dir_prefix,"newnas/MJY_file/CE-MRI/train_result", "VAE_4_fold5-1", "checkpoint", "checkpoint-v1.ckpt")
        model.instantiate_first_stage(config=None, ckpt=vae_ckpt)

    # reset perd dir for different machine
    model.pred_result_dir = os.path.join(dir_prefix, "newnas_1", model.pred_result_dir.split('newnas_1/')[-1] + "_" +
                                         f"{config.sampler_setting.sampler}_{config.sampler_setting.sample_steps}_" +
                                         f"eta{config.sampler_setting.ddim_eta}_{ckpt_name.split('.')[0]}")
    model.sampler_setting = config.sampler_setting
    model.test_batch_size = config.test_batch_size
    model.test_num = config.test_num
    if not os.path.exists(model.pred_result_dir):
        os.makedirs(model.pred_result_dir)
    # ===============以防万一地址变动================
    # root_dir = os.path.dirname(ckpt_dir)
    # model.template_dir = os.path.join(dir_prefix, 'newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm')
    # model.data_dir = os.path.join(dir_prefix, "newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre")
    # model.train_dir = os.path.join(dir_prefix, 'newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre/images_tr')
    # model.test_dir = os.path.join(dir_prefix, 'newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre_320320_01norm/images_ts')
    # model.log_pic_dir = os.path.join(root_dir, "loss_pic")
    # model.result_dir = root_dir
    # model.record_file = os.path.join(root_dir, "log_txt.txt")
    # model.num_workers = 0
    # ==========PL MODEL============
    torch.set_float32_matmul_precision('high')
    print("========================{}==========================".format(task_name))
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[config.cuda_idx],
        enable_progress_bar=False,
        # limit_predict_batches=2
    )
    predictions = trainer.predict(model)

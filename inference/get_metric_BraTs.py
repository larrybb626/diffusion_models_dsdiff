import os.path
import sys
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from inference.test_metrics import *

import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
torch.set_num_threads(32)
def main(config=None):
    # ==========path============
    dir_prefix = sys.argv[0].split("/newnas")[0]
    if config is None:
        config = OmegaConf.load("../configs/inference_config_BraTs.yaml")
    assert config.Task_name == "BraTs_synthesis"
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # txt_path = os.path.join(dir_prefix, "newnas/MJY_file/CE-MRI", "peizhun_error.txt")
    # with open(txt_path, "r") as f:
    #     # 读取全部内容
    #     file_contents = f.read()
    # bad_list = file_contents.split("\n")
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name
    net_mode = config.net_mode
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, net_mode, fold_idx)
    result_path = config.result_path
    gt_dir = os.path.join(config.filepath_img, "images_ts")
    pred_dir = os.path.join(result_path, task_name, "pred_nii_" +
                            f"{config.sampler_setting.sampler}_{config.sampler_setting.sample_steps}_" +
                            f"eta{config.sampler_setting.ddim_eta}_{ckpt_name}")
    # pred_dir = os.path.join("/data/newnas_1/MJY_file/SOTA_models/DiscDiff_test/BraTs_v_predict_7e4_uknow/itk_save_dir","")
    # pred_dir = os.path.join(
    #     "/data/newnas_1/MJY_file/SOTA_models/SOTA_GAN/ResVit_brats/result", "")
    # pred_dir = os.path.join('/data/newnas_1/MJY_file/SOTA_models/DiT_BraTs_test/011-DiT-XL-2/dpm20/itk_result','')
    # pred_dir = os.path.join('/data/newnas_1/MJY_file/SOTA_models/controlnet-model-brats/controlnet_result/checkpoint-50000/itk_result', '')
    # pred_dir = os.path.join('/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/SA-GAN/experiment_1/itk_result', '')
    # pred_dir = os.path.join('/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/FGDM/result_cond/itk_result','')
    # pred_dir = os.path.join(
    #     '/data/newnas_1/MJY_file/SOTA_models/mulD_RegGAN/braCE_MRI_simulate_PCa_64_pix2pix_mulD_fold5-1/pred_nii')
    excel_save_dir = (pred_dir + "_metric.xlsx") if not config.use_prostate_mask else (pred_dir + "_metric_mask.xlsx")
    print("====================={}=======================".format(pred_dir))
    # ============================================
    pred_list = os.listdir(pred_dir)
    metrics = []
    mean_nrmse_metric = 0
    mean_smape_metric = 0
    mean_logac_matric = 0
    mean_medsymac_matric = 0
    for idx, filename in enumerate(reversed(pred_list)):
        if not filename.endswith(".nii.gz"):
            continue
        id_num = filename.split(".")[0].split("_")[0]
        # if id_num in bad_list:
        #     continue
        gt_file = os.path.join(gt_dir, id_num, "ce.nii.gz")
        pred_file = os.path.join(pred_dir, filename)
        mask_file = None
        prostate_mask = os.path.join(gt_dir, id_num, "seg.nii.gz")
        gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
        pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        if config.use_prostate_mask:
            mask_img = sitk.GetArrayFromImage(sitk.ReadImage(prostate_mask))
            mask_img = mask_img > 0
        else:
            mask_img = None
        # get error matric
        # normalized root mean square error
        nrmse_metric = nrmse(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # symmetric mean absolute percent error
        smape_metric = smape(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # log accuracy ratio
        logac_matric = logac(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # median symmetric accuracy
        medsymac_matric = medsymac(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # get sim metrics
        # =======neighborhood cross correlation=======
        # cc_metric = cc_py(gt_file, pred_file, mask_file)
        cc_metric = 0
        # =======histogram mutual information=======
        mi_metric = mi_py(gt_file, pred_file, mask_file)
        # mi_metric = nmi(true_array=gt_img,pred_array=pred_img,mask=mask_img)
        # =======ssim=======
        ssim_metric = ssim_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======psnr=======
        psnr_metric = psnr(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======lpips=======
        # lpips=lpips_metric(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        lpips = 0
        # lpips = med_lpips_metric(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======fid=======
        # fid = fid_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img, compute=(idx == (len(pred_list) - 1)))
        fid = 0
        # =======vif=======
        # vif = vif_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        vif = 0
        all_metric = [str(id_num), nrmse_metric, smape_metric, logac_matric, medsymac_matric, cc_metric, mi_metric,
                      ssim_metric, lpips, fid, psnr_metric]
        metrics.append(all_metric)
        print("{}/{} {}".format(idx + 1, len(pred_list), id_num),
              "nrmse:{}, smape:{}, logza:{}, medsymaz:{}, cc:{}, mi:{}, ssim:{}, lpips:{}, fid:{}, psnr:{}".format(
                  *all_metric[1:]))
    # 求平均
    np_vresion = np.array(metrics)[:, 1:].astype(np.float32)
    mean = np.mean(np_vresion, axis=0)
    mean_metric = [0, *mean[:]]
    print("nrmse:{}, smape:{}, logza:{}, medsymaz:{}, cc:{}, mi:{}, ssim:{}, lpips:{}, fid:{}, psnr:{}".format(
        *mean_metric[1:]))
    metrics.insert(0, mean_metric)
    # 保存表格
    result = pd.DataFrame(metrics)
    record = pd.ExcelWriter(excel_save_dir, mode='w')
    header = ['ids'] + ["nrmse", "smape", "logac", "medsymac", "cc", "mi", "ssim", "lpips", "fid", "psnr"]
    result.to_excel(record, header=header, index=False)
    record.close()
if __name__ == "__main__":
    main()

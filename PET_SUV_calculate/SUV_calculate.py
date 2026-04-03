"""PET pred SUV calculator (Scheme A).

Workflow:
1) Read per-patient metadata json exported from gold-standard DICOM.
2) Inverse-normalize pred image using original S_Data2 intensity range.
3) Convert to SUV and export per-case statistics to Excel.

Only Excel is saved by design; SUV NIfTI is not written.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk


META_KEYS: List[str] = [
    "SeriesTime",
    "AcquisitionTime",
    "PatientWeight",
    "RadiopharmaceuticalStartTime",
    "RadionuclideTotalDose",
    "RadionuclideHalfLife",
    "RescaleSlope",
    "RescaleIntercept",
]


def extract_dicom_params(dicom_path: str) -> Dict[str, str]:
    """Keep the blog-style DICOM tag extraction for compatibility."""
    from pydicom import dcmread

    dcm = dcmread(dicom_path)
    radio = dcm.RadiopharmaceuticalInformationSequence[0]
    return {
        "SeriesTime": str(getattr(dcm, "SeriesTime", "")),
        "AcquisitionTime": str(getattr(dcm, "AcquisitionTime", "")),
        "PatientWeight": str(getattr(dcm, "PatientWeight", "")),
        "RadiopharmaceuticalStartTime": str(radio["RadiopharmaceuticalStartTime"].value),
        "RadionuclideTotalDose": str(radio["RadionuclideTotalDose"].value),
        "RadionuclideHalfLife": str(radio["RadionuclideHalfLife"].value),
        "RescaleSlope": str(getattr(dcm, "RescaleSlope", 1)),
        "RescaleIntercept": str(getattr(dcm, "RescaleIntercept", 0)),
    }


def _time_to_seconds(t: str) -> float:
    left, _, frac = str(t).partition(".")
    left = left.zfill(6)
    hh = int(left[0:2])
    mm = int(left[2:4])
    ss = int(left[4:6])
    frac_sec = float("0." + frac) if frac else 0.0
    return hh * 3600 + mm * 60 + ss + frac_sec


def _to_float_params(params: Dict[str, str]) -> Dict[str, float]:
    return {
        "PatientWeight": float(params["PatientWeight"]),
        "RadionuclideTotalDose": float(params["RadionuclideTotalDose"]),
        "RadionuclideHalfLife": float(params["RadionuclideHalfLife"]),
        "RescaleSlope": float(params["RescaleSlope"]),
        "RescaleIntercept": float(params["RescaleIntercept"]),
        "AcquisitionTime": _time_to_seconds(params["AcquisitionTime"]),
        "RadiopharmaceuticalStartTime": _time_to_seconds(params["RadiopharmaceuticalStartTime"]),
    }


def _parse_patient_id(pred_filename: str, task_id: str) -> str:
    stem = pred_filename
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    if stem.endswith("_pred"):
        stem = stem[:-5]

    prefix = f"{task_id}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]

    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def _load_json_params(json_root: str, patient_id: str) -> Dict[str, str]:
    patient_dir = os.path.join(json_root, patient_id)
    target_json = os.path.join(patient_dir, f"{patient_id}_S_Data2.json")
    if os.path.exists(target_json):
        with open(target_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.isdir(patient_dir):
        for name in sorted(os.listdir(patient_dir)):
            if name.lower().endswith(".json"):
                with open(os.path.join(patient_dir, name), "r", encoding="utf-8") as f:
                    return json.load(f)

    raise FileNotFoundError(f"No json metadata found for patient: {patient_id}")


def _get_inverse_norm_range(orig_nii_path: str) -> Tuple[float, float]:
    """Rebuild min/max used by your MinMax normalization path.

    normalization.py logic:
      upper = 0.75 * max(orig)
      orig_clip = min(orig, upper)
      norm = ((orig_clip - min) / (max - min)) * 2 - 1
    """
    img = sitk.ReadImage(orig_nii_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    upper = float(arr.max()) * 0.75
    arr = np.minimum(arr, upper)

    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        raise ValueError(f"Invalid inverse-normalization range from: {orig_nii_path}")
    return min_v, max_v


def _inverse_normalize_pred(pred_arr: np.ndarray, min_v: float, max_v: float, clip_norm: bool) -> np.ndarray:
    pred = pred_arr.astype(np.float32)
    if clip_norm:
        pred = np.clip(pred, -1.0, 1.0)
    return (pred + 1.0) * 0.5 * (max_v - min_v) + min_v


def _suv_from_activity_like(pred_denorm: np.ndarray, params: Dict[str, str]) -> np.ndarray:
    p = _to_float_params(params)

    dt = p["AcquisitionTime"] - p["RadiopharmaceuticalStartTime"]
    if dt < 0:
        dt += 24.0 * 3600.0

    lam = math.log(2.0) / p["RadionuclideHalfLife"]
    dose_at_scan = p["RadionuclideTotalDose"] * math.exp(-lam * dt)

    activity = pred_denorm * p["RescaleSlope"] + p["RescaleIntercept"]
    suv = activity * p["PatientWeight"] / dose_at_scan
    return suv.astype(np.float32)


def _collect_pred_files(pred_path: str) -> List[str]:
    if os.path.isfile(pred_path):
        return [pred_path]

    files: List[str] = []
    for n in sorted(os.listdir(pred_path)):
        if n.endswith("_pred.nii.gz"):
            files.append(os.path.join(pred_path, n))
    return files


def _default_excel_basename(pred_path: str) -> str:
    if os.path.isdir(pred_path):
        base = os.path.basename(os.path.normpath(pred_path))
    else:
        base = os.path.basename(pred_path).replace(".nii.gz", "")
    return f"{base}_suv_stats1.xlsx"


def _resolve_out_excel(pred_path: str, out_excel: Optional[str]) -> str:
    """Resolve output path when out_excel is None, a directory, or a file path."""
    default_name = _default_excel_basename(pred_path)

    if out_excel is None:
        if os.path.isdir(pred_path):
            out_dir = os.path.dirname(os.path.normpath(pred_path))
        else:
            out_dir = os.path.dirname(pred_path)
        return os.path.join(out_dir, default_name)

    # If user passes a folder path (existing or intended), place auto filename inside it.
    is_existing_dir = os.path.isdir(out_excel)
    looks_like_dir = out_excel.endswith(os.sep) or os.path.splitext(out_excel)[1] == ""
    if is_existing_dir or looks_like_dir:
        return os.path.join(out_excel, default_name)

    return out_excel


def process_pred_dir(
    pred_path: str,
    json_root: str,
    orig_nii_root: str,
    task_id: str,
    clip_norm: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pred_files = _collect_pred_files(pred_path)

    base_cols = [
        "PatientID",
        "PredFile",
        "SUV_Mean",
        "SUV_Max",
        "SUV_Min_Positive",
        "SUV_Std_Positive",
        "Norm_Min",
        "Norm_Max",
        "Status",
        "Error",
    ]

    if not pred_files:
        raise RuntimeError(f"No pred files found in: {pred_path}")

    for idx, pred_file in enumerate(pred_files, start=1):
        pred_name = os.path.basename(pred_file)
        patient_id = _parse_patient_id(pred_name, task_id)

        try:
            params = _load_json_params(json_root, patient_id)
            orig_nii = os.path.join(orig_nii_root, patient_id, "S_Data2.nii.gz")
            if not os.path.exists(orig_nii):
                raise FileNotFoundError(f"Missing original S_Data2 for inverse-normalization: {orig_nii}")

            min_v, max_v = _get_inverse_norm_range(orig_nii)

            pred_img = sitk.ReadImage(pred_file)
            pred_arr = sitk.GetArrayFromImage(pred_img)
            pred_denorm = _inverse_normalize_pred(pred_arr, min_v, max_v, clip_norm)
            suv = _suv_from_activity_like(pred_denorm, params)

            pos = suv[suv > 0]
            row = {
                "PatientID": patient_id,
                "PredFile": pred_name,
                "SUV_Mean": float(np.mean(pos)) if pos.size else 0.0,
                "SUV_Max": float(np.max(suv)),
                "SUV_Min_Positive": float(np.min(pos)) if pos.size else 0.0,
                "SUV_Std_Positive": float(np.std(pos)) if pos.size else 0.0,
                "Norm_Min": float(min_v),
                "Norm_Max": float(max_v),
                "Status": "OK",
                "Error": "",
            }
            for k in META_KEYS:
                row[k] = str(params.get(k, ""))
        except Exception as exc:
            row = {
                "PatientID": patient_id,
                "PredFile": pred_name,
                "SUV_Mean": 0.0,
                "SUV_Max": 0.0,
                "SUV_Min_Positive": 0.0,
                "SUV_Std_Positive": 0.0,
                "Norm_Min": 0.0,
                "Norm_Max": 0.0,
                "Status": "FAIL",
                "Error": str(exc),
            }
            for k in META_KEYS:
                row[k] = ""

        rows.append(row)
        if row["Status"] == "FAIL":
            print(f"[{idx}/{len(pred_files)}] {pred_name} -> {row['Status']} (Error: {row['Error']})")
        else:
            print(f"[{idx}/{len(pred_files)}] {pred_name} -> {row['Status']}")

    df = pd.DataFrame(rows)
    ordered_cols = base_cols + META_KEYS
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = "" if col in META_KEYS else 0.0
    return df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SUV stats Excel for pred NIfTI files (Scheme A)")

    parser.add_argument(
        "--pred_path",
        type=str,
        default="/nas_3/LaiRuiBin/Changhai/results/2026_0204/result/PET_synthesis_0309_ds_diff_gaussian_fold5-1/pred_nii_ddim_20_eta0_checkpoint",
        help="Pred directory or one pred file",
    )
    parser.add_argument(
        "--json_root",
        type=str,
        default="/nas_3/LiuWenxi/Changhai/data/suv_nii/SSA",
        help="Root folder that contains per-patient json from gold DICOM",
    )
    parser.add_argument(
        "--orig_nii_root",
        type=str,
        default='/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_ts_256',
        help="Original (non-normalized) NIfTI root: {root}/{patient}/S_Data2.nii.gz",
    )
    parser.add_argument("--task_id", type=str, default="0309", help="Task id prefix in pred filename")
    parser.add_argument("--clip_norm", action="store_true", help="Clip pred values to [-1,1] before inverse-normalization")
    parser.add_argument(
        "--out_excel",
        type=str,
        default='/nas_3/LaiRuiBin/Changhai/results/2026_0204/result/PET_synthesis_0309_ds_diff_gaussian_fold5-1/',
        help="Output Excel file path or folder path. If folder, auto filename is generated.",
    )

    args = parser.parse_args()

    args.out_excel = _resolve_out_excel(args.pred_path, args.out_excel)

    df = process_pred_dir(
        pred_path=args.pred_path,
        json_root=args.json_root,
        orig_nii_root=args.orig_nii_root,
        task_id=args.task_id,
        clip_norm=args.clip_norm,
    )

    os.makedirs(os.path.dirname(args.out_excel) or ".", exist_ok=True)
    df.to_excel(args.out_excel, index=False)

    ok_count = int((df["Status"] == "OK").sum())
    fail_count = int((df["Status"] == "FAIL").sum())
    print(f"[DONE] Excel saved: {args.out_excel}")
    print(f"[DONE] OK={ok_count}, FAIL={fail_count}")


if __name__ == "__main__":
    main()

"""Compute SUV stats for predicted PET after inverse-normalization with gold min/max.

Pipeline per patient:
1) Read predicted normalized PET (usually in [-1, 1]).
2) Read gold PET NIfTI and get min/max (optionally with 0.75*max clipping rule).
3) Inverse-normalize pred back to PET intensity domain.
4) Read one DICOM in patient S/Data2, extract dose/time tags, convert PET to SUV.
5) Export per-case SUV statistics to Excel.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pydicom import dcmread


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


def parse_patient_id(pred_filename: str, task_id: str) -> str:
    stem = pred_filename
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    if stem.endswith("_pred"):
        stem = stem[:-5]

    if task_id:
        prefix = f"{task_id}_"
        if stem.startswith(prefix):
            return stem[len(prefix):]

    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 and task_id else stem


def collect_pred_files(pred_path: str) -> List[str]:
    if os.path.isfile(pred_path):
        return [pred_path]

    files: List[str] = []
    for name in sorted(os.listdir(pred_path)):
        if name.endswith("_pred.nii.gz"):
            files.append(os.path.join(pred_path, name))
    return files


def resolve_gold_nii(gold_nii_root: str, patient_id: str) -> str:
    candidates = [
        os.path.join(gold_nii_root, patient_id, f"{patient_id}_S_Data2.nii.gz"),
        os.path.join(gold_nii_root, patient_id, "S_Data2.nii.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Gold NIfTI not found for {patient_id}: {candidates}")


def get_gold_minmax(gold_nii: str, use_clip_075: bool) -> Tuple[float, float]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(gold_nii)).astype(np.float32)
    if use_clip_075:
        upper = float(arr.max()) * 0.75
        arr = np.minimum(arr, upper)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        raise ValueError(f"Invalid min/max from gold image: {gold_nii}")
    return min_v, max_v


def inverse_normalize(pred_arr: np.ndarray, min_v: float, max_v: float, clip_norm: bool) -> np.ndarray:
    pred = pred_arr.astype(np.float32)
    if clip_norm:
        pred = np.clip(pred, -1.0, 1.0)
    return (pred + 1.0) * 0.5 * (max_v - min_v) + min_v


def find_one_dicom_file(dicom_dir: str) -> str:
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    for root, _, files in os.walk(dicom_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            try:
                _ = dcmread(path, stop_before_pixels=True, force=True)
                return path
            except Exception:
                continue

    raise FileNotFoundError(f"No readable DICOM found under: {dicom_dir}")


def extract_dicom_params(dicom_path: str) -> Dict[str, str]:
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


def dicom_hhmmss(t: str) -> float:
    left, _, frac = str(t).partition(".")
    left = left.zfill(6)
    h = float(left[0:2])
    m = float(left[2:4])
    s = float(left[4:6])
    frac_s = float("0." + frac) if frac else 0.0
    return h * 3600.0 + m * 60.0 + s + frac_s


def pet_to_suv(data: List[str], pet: np.ndarray, use_rescale: bool) -> np.ndarray:
    st, _at, pw, rst, rtd, rhl, rs, ri = data

    decay_time = dicom_hhmmss(st) - dicom_hhmmss(rst)
    if decay_time < 0:
        decay_time += 24.0 * 3600.0

    decay_dose = float(rtd) * pow(2.0, -float(decay_time) / float(rhl))
    suv_bw_scale_factor = (1000.0 * float(pw)) / decay_dose

    pet_f = pet.astype(np.float32)
    if use_rescale:
        pet_f = pet_f * float(rs) + float(ri)
    return (pet_f * suv_bw_scale_factor).astype(np.float32)


def suv_stats(arr: np.ndarray) -> Dict[str, float]:
    pos = arr[arr > 0]
    return {
        "SUV_Mean": float(np.mean(pos)) if pos.size else 0.0,
        "SUV_Max": float(np.max(arr)),
        "SUV_Min_Positive": float(np.min(pos)) if pos.size else 0.0,
        "SUV_Std_Positive": float(np.std(pos)) if pos.size else 0.0,
    }


def default_out_excel(pred_path: str, out_excel: str) -> str:
    if out_excel:
        return out_excel
    if os.path.isdir(pred_path):
        base = os.path.basename(os.path.normpath(pred_path))
        out_dir = os.path.dirname(os.path.normpath(pred_path))
    else:
        base = os.path.basename(pred_path).replace(".nii.gz", "")
        out_dir = os.path.dirname(pred_path)
    return os.path.join(out_dir, f"{base}_suv_from_gold_minmax.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverse-normalize pred with gold min/max, then compute SUV stats"
    )
    parser.add_argument("--pred_path", type=str, required=True, help="Pred folder or one *_pred.nii.gz file")
    parser.add_argument(
        "--gold_nii_root",
        type=str,
        required=True,
        help="Gold NIfTI root with {patient}/{patient}_S_Data2.nii.gz",
    )
    parser.add_argument(
        "--dicom_root",
        type=str,
        required=True,
        help="DICOM root with {patient}/S/Data2",
    )
    parser.add_argument(
        "--dicom_rel",
        type=str,
        default=os.path.join("S", "Data2"),
        help="Relative DICOM path under each patient folder",
    )
    parser.add_argument("--task_id", type=str, default="", help="Optional task prefix in pred filename")
    parser.add_argument("--out_excel", type=str, default="", help="Output Excel path")
    parser.add_argument("--clip_norm", action="store_true", help="Clip pred to [-1,1] before inverse-normalization")
    parser.add_argument(
        "--gold_clip_075",
        action="store_true",
        help="Use 0.75*max clipping on gold image before min/max (match old normalization path)",
    )
    parser.add_argument(
        "--no_rescale",
        action="store_true",
        help="Disable DICOM rescale (PET*RS+RI) before SUV scaling",
    )

    args = parser.parse_args()
    out_excel = default_out_excel(args.pred_path, args.out_excel)
    dicom_rel = str(args.dicom_rel)

    pred_files = collect_pred_files(args.pred_path)
    if not pred_files:
        raise RuntimeError(f"No pred files found in: {args.pred_path}")

    rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    for idx, pred_file in enumerate(pred_files, start=1):
        pred_name = os.path.basename(pred_file)
        patient_id = parse_patient_id(pred_name, args.task_id)

        row: Dict[str, object] = {
            "PatientID": patient_id,
            "PredFile": pred_name,
            "GoldNii": "",
            "DicomPath": "",
            "PredNormMin": 0.0,
            "PredNormMax": 0.0,
            "PredDenormMin": 0.0,
            "PredDenormMax": 0.0,
            "GoldMin": 0.0,
            "GoldMax": 0.0,
            "SUV_Mean": 0.0,
            "SUV_Max": 0.0,
            "SUV_Min_Positive": 0.0,
            "SUV_Std_Positive": 0.0,
            "Status": "FAIL",
            "Error": "",
        }
        for k in META_KEYS:
            row[k] = ""

        try:
            gold_nii = resolve_gold_nii(args.gold_nii_root, patient_id)
            dicom_path = find_one_dicom_file(os.path.join(args.dicom_root, patient_id, dicom_rel))
            params = extract_dicom_params(dicom_path)

            pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred_file)).astype(np.float32)
            min_v, max_v = get_gold_minmax(gold_nii, use_clip_075=args.gold_clip_075)
            pred_denorm = inverse_normalize(pred_arr, min_v=min_v, max_v=max_v, clip_norm=args.clip_norm)

            data = [
                params["SeriesTime"],
                params["AcquisitionTime"],
                params["PatientWeight"],
                params["RadiopharmaceuticalStartTime"],
                params["RadionuclideTotalDose"],
                params["RadionuclideHalfLife"],
                params["RescaleSlope"],
                params["RescaleIntercept"],
            ]
            suv = pet_to_suv(data=data, pet=pred_denorm, use_rescale=(not args.no_rescale))
            pred_stats = suv_stats(suv)

            gold_arr = sitk.GetArrayFromImage(sitk.ReadImage(gold_nii)).astype(np.float32)
            gold_suv = pet_to_suv(data=data, pet=gold_arr, use_rescale=(not args.no_rescale))
            gold_stats = suv_stats(gold_suv)

            row.update(
                {
                    "GoldNii": gold_nii,
                    "DicomPath": dicom_path,
                    "PredNormMin": float(pred_arr.min()),
                    "PredNormMax": float(pred_arr.max()),
                    "PredDenormMin": float(pred_denorm.min()),
                    "PredDenormMax": float(pred_denorm.max()),
                    "GoldMin": float(min_v),
                    "GoldMax": float(max_v),
                    "SUV_Mean": pred_stats["SUV_Mean"],
                    "SUV_Max": pred_stats["SUV_Max"],
                    "SUV_Min_Positive": pred_stats["SUV_Min_Positive"],
                    "SUV_Std_Positive": pred_stats["SUV_Std_Positive"],
                    "Status": "OK",
                    "Error": "",
                }
            )
            for k in META_KEYS:
                row[k] = str(params.get(k, ""))

            error_rows.append(
                {
                    "PatientID": patient_id,
                    "SE_SUV_Mean": (pred_stats["SUV_Mean"] - gold_stats["SUV_Mean"]) ** 2,
                    "SE_SUV_Max": (pred_stats["SUV_Max"] - gold_stats["SUV_Max"]) ** 2,
                    "SE_SUV_Min_Positive": (pred_stats["SUV_Min_Positive"] - gold_stats["SUV_Min_Positive"]) ** 2,
                    "SE_SUV_Std_Positive": (pred_stats["SUV_Std_Positive"] - gold_stats["SUV_Std_Positive"]) ** 2,
                }
            )
        except Exception as exc:
            row["Error"] = str(exc)

        rows.append(row)
        if row["Status"] == "OK":
            print(f"[{idx}/{len(pred_files)}] {pred_name} -> OK")
        else:
            print(f"[{idx}/{len(pred_files)}] {pred_name} -> FAIL (Error: {row['Error']})")

    df = pd.DataFrame(rows)
    ordered_cols = [
        "PatientID",
        "PredFile",
        "GoldNii",
        "DicomPath",
        *META_KEYS,
        "PredNormMin",
        "PredNormMax",
        "PredDenormMin",
        "PredDenormMax",
        "GoldMin",
        "GoldMax",
        "SUV_Mean",
        "SUV_Max",
        "SUV_Min_Positive",
        "SUV_Std_Positive",
        "Status",
        "Error",
    ]
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[ordered_cols]

    os.makedirs(os.path.dirname(out_excel) or ".", exist_ok=True)
    df.to_excel(out_excel, index=False)

    error_out_excel = out_excel.replace(".xlsx", "_error.xlsx")
    err_cols = [
        "PatientID",
        "SE_SUV_Mean",
        "SE_SUV_Max",
        "SE_SUV_Min_Positive",
        "SE_SUV_Std_Positive",
    ]
    df_err = pd.DataFrame(error_rows)
    if not df_err.empty:
        mse_row = {
            "PatientID": "MSE",
            "SE_SUV_Mean": float(df_err["SE_SUV_Mean"].mean()),
            "SE_SUV_Max": float(df_err["SE_SUV_Max"].mean()),
            "SE_SUV_Min_Positive": float(df_err["SE_SUV_Min_Positive"].mean()),
            "SE_SUV_Std_Positive": float(df_err["SE_SUV_Std_Positive"].mean()),
        }
        rmse_row = {
            "PatientID": "RMSE",
            "SE_SUV_Mean": float(np.sqrt(mse_row["SE_SUV_Mean"])),
            "SE_SUV_Max": float(np.sqrt(mse_row["SE_SUV_Max"])),
            "SE_SUV_Min_Positive": float(np.sqrt(mse_row["SE_SUV_Min_Positive"])),
            "SE_SUV_Std_Positive": float(np.sqrt(mse_row["SE_SUV_Std_Positive"])),
        }
        df_err = pd.concat([df_err, pd.DataFrame([mse_row, rmse_row])], ignore_index=True)
    else:
        df_err = pd.DataFrame(columns=err_cols)
    df_err = df_err[err_cols]
    df_err.to_excel(error_out_excel, index=False)

    ok = int((df["Status"] == "OK").sum())
    fail = int((df["Status"] == "FAIL").sum())
    print(f"[DONE] Excel saved: {out_excel}")
    print(f"[DONE] Error Excel saved: {error_out_excel}")
    print(f"[DONE] OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    main()


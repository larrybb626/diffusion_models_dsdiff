"""Recompute gold-standard SUV from PET NIfTI + DICOM dose metadata.

Method follows the user-provided recipe:
- Read one DICOM under each patient `S/Data2`.
- Extract Series/Acquisition time, weight, injected dose, half-life, rescale info.
- Convert PET voxel values to SUV using PET2SUV formula.
- Save per-patient metadata and SUV statistics to Excel.
"""

import argparse
import os
from typing import Dict, List

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


def find_one_dicom_file(dicom_dir: str) -> str:
    """Return the first readable DICOM file from a directory tree."""
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
    """Convert DICOM time HHMMSS(.ffffff) to seconds."""
    left, _, frac = str(t).partition(".")
    left = left.zfill(6)
    if len(left) == 5:
        left = "0" + left

    h_t = float(left[0:2])
    m_t = float(left[2:4])
    s_t = float(left[4:6])
    frac_s = float("0." + frac) if frac else 0.0
    return h_t * 3600.0 + m_t * 60.0 + s_t + frac_s


def pet_to_suv(data: List[str], pet: np.ndarray, norm: bool) -> np.ndarray:
    """Apply the provided PET2SUV formula."""
    st, _at, pw, rst, rtd, rhl, rs, ri = data

    decay_time = dicom_hhmmss(st) - dicom_hhmmss(rst)
    if decay_time < 0:
        decay_time += 24.0 * 3600.0

    decay_dose = float(rtd) * pow(2.0, -float(decay_time) / float(rhl))
    suv_bw_scale_factor = (1000.0 * float(pw)) / decay_dose

    pet_f = pet.astype(np.float32)
    if norm:
        pet_suv = (pet_f * float(rs) + float(ri)) * suv_bw_scale_factor
    else:
        pet_suv = pet_f * suv_bw_scale_factor
    return pet_suv.astype(np.float32)


def _build_row_template(patient_id: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "PatientID": patient_id,
        "NiiPath": "",
        "DicomPath": "",
        "SUV_Mean": 0.0,
        "SUV_Max": 0.0,
        "SUV_Min_Positive": 0.0,
        "SUV_Std_Positive": 0.0,
        "Status": "FAIL",
        "Error": "",
    }
    for k in META_KEYS:
        row[k] = ""
    return row


def resolve_patient_nii_path(nii_root: str, patient_id: str) -> str:
    """Resolve per-patient PET NIfTI using {patient_id}_S_Data2.nii.gz."""
    patient_dir = os.path.join(nii_root, patient_id)
    target = os.path.join(patient_dir, f"{patient_id}_S_Data2.nii.gz")
    if os.path.exists(target):
        return target

    raise FileNotFoundError(
        f"Missing NIfTI for patient '{patient_id}': {target}"
    )


def process_one_patient(
    patient_id: str,
    nii_root: str,
    dicom_root: str,
    dicom_rel: str,
    use_norm_formula: bool,
) -> Dict[str, object]:
    row = _build_row_template(patient_id)

    try:
        nii_path = resolve_patient_nii_path(nii_root=nii_root, patient_id=patient_id)

        dicom_dir = os.path.join(dicom_root, patient_id, dicom_rel)
        dicom_path = find_one_dicom_file(dicom_dir)

        params = extract_dicom_params(dicom_path)
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

        pet_img = sitk.ReadImage(nii_path)
        pet_arr = sitk.GetArrayFromImage(pet_img)
        suv_arr = pet_to_suv(data=data, pet=pet_arr, norm=use_norm_formula)

        pos = suv_arr[suv_arr > 0]
        row.update(
            {
                "NiiPath": nii_path,
                "DicomPath": dicom_path,
                "SUV_Mean": float(np.mean(pos)) if pos.size else 0.0,
                "SUV_Max": float(np.max(suv_arr)),
                "SUV_Min_Positive": float(np.min(pos)) if pos.size else 0.0,
                "SUV_Std_Positive": float(np.std(pos)) if pos.size else 0.0,
                "Status": "OK",
                "Error": "",
            }
        )
        for k in META_KEYS:
            row[k] = str(params.get(k, ""))

    except Exception as exc:
        row["Error"] = str(exc)

    return row


def collect_patient_ids(nii_root: str) -> List[str]:
    if not os.path.isdir(nii_root):
        raise FileNotFoundError(f"NIfTI root not found: {nii_root}")

    patient_ids: List[str] = []
    for name in sorted(os.listdir(nii_root)):
        full = os.path.join(nii_root, name)
        if os.path.isdir(full):
            patient_ids.append(name)

    if not patient_ids:
        raise RuntimeError(f"No patient directories found in: {nii_root}")
    return patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute gold SUV from S_Data2.nii.gz using DICOM dose metadata"
    )
    parser.add_argument(
        "--nii_root",
        type=str,
        default="/nas_3/LaiRuiBin/Changhai/data/ori_nii/SSA",
        help="Root folder of patient data with {patient}/{patient}_S_Data2.nii.gz",
    )
    parser.add_argument(
        "--dicom_root",
        type=str,
        default="/nas_3/LiuWenxi/Changhai/图像生成-长海医院/图像生成/SSA",
        help="Root folder of patient-name DICOM folders",
    )
    parser.add_argument(
        "--dicom_rel",
        type=str,
        default=os.path.join("S", "Data2"),
        help="Relative path under each patient DICOM folder",
    )
    parser.add_argument(
        "--out_excel",
        type=str,
        default="/nas_3/LaiRuiBin/Changhai/gold_suv_from_dicom.xlsx",
        help="Output Excel path",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Use standard formula: (PET*RS+RI)*SUVbwScaleFactor",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Optional single patient id/name to run",
    )

    args = parser.parse_args()

    if args.patient:
        patient_ids = [args.patient]
    else:
        patient_ids = collect_patient_ids(args.nii_root)

    rows: List[Dict[str, object]] = []
    total = len(patient_ids)
    for idx, pid in enumerate(patient_ids, start=1):
        row = process_one_patient(
            patient_id=pid,
            nii_root=args.nii_root,
            dicom_root=args.dicom_root,
            dicom_rel=args.dicom_rel,
            use_norm_formula=args.norm,
        )
        rows.append(row)

        if row["Status"] == "OK":
            print(f"[{idx}/{total}] {pid} -> OK")
        else:
            print(f"[{idx}/{total}] {pid} -> FAIL (Error: {row['Error']})")

    df = pd.DataFrame(rows)
    ordered_cols = [
        "PatientID",
        "NiiPath",
        "DicomPath",
        *META_KEYS,
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

    out_dir = os.path.dirname(args.out_excel) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(args.out_excel, index=False)

    ok = int((df["Status"] == "OK").sum())
    fail = int((df["Status"] == "FAIL").sum())
    print(f"[DONE] Excel saved: {args.out_excel}")
    print(f"[DONE] OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    main()


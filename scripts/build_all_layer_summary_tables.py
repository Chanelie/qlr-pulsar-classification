from pathlib import Path
import pandas as pd

TABLES = Path("tables")

N_VALUES = [200, 500, 1000]
LAYERS = [1, 2, 3, 5, 10]

BASELINE_ORDER = ["LogReg", "CSVM", "XGBoost", "QSVM"]
QLR_ENCODING_ORDER = ["amplitude", "angle", "data_reuploading"]

ENCODING_TO_MODEL = {
    "amplitude": "QLR_amplitude",
    "angle": "QLR_angle",
    "data_reuploading": "QLR_data_reuploading",
}

SUMMARY_COLS = ["ROC_AUC", "PR_AUC", "F1", "Precision", "Recall"]
STRICT_FPR_COLS = ["Recall_FPR1", "Precision_FPR1", "Recall_FPR5"]
CALIBRATION_COLS = ["Brier", "ECE"]
RUNTIME_COLS = ["Train_s", "Pred_s"]


def fmt(mean, se, digits=3):
    if pd.isna(mean):
        return ""
    if pd.isna(se):
        se = 0.0
    return f"{float(mean):.{digits}f} ± {float(se):.{digits}f}"


def add_baseline_rows(n, outputs):
    mean_path = TABLES / f"baselines_mean_N{n}.csv"
    se_path = TABLES / f"baselines_se_N{n}.csv"

    bl_mean = pd.read_csv(mean_path)
    bl_se = pd.read_csv(se_path)

    for model in BASELINE_ORDER:
        row_m = bl_mean[bl_mean["model"] == model]
        row_s = bl_se[bl_se["model"] == model]

        if row_m.empty or row_s.empty:
            print(f"[WARN] Missing baseline row for {model}, N={n}")
            continue

        row_m = row_m.iloc[0]
        row_s = row_s.iloc[0]

        for name, cols in [
            ("summary", SUMMARY_COLS),
            ("strict_fpr", STRICT_FPR_COLS),
            ("calibration", CALIBRATION_COLS),
            ("runtime", RUNTIME_COLS),
        ]:
            out_row = {"Model": model}
            for col in cols:
                if col in row_m.index and col in row_s.index:
                    digits = 2 if name == "runtime" else 3
                    out_row[col] = fmt(row_m[col], row_s[col], digits=digits)
            outputs[name].append(out_row)


def add_qlr_rows(n, outputs):
    for L in LAYERS:
        mean_path = TABLES / f"qlr_mean_N{n}_L{L}.csv"
        se_path = TABLES / f"qlr_se_N{n}_L{L}.csv"

        if not mean_path.exists() or not se_path.exists():
            print(f"[WARN] Missing QLR table for N={n}, L={L}")
            continue

        qlr_mean = pd.read_csv(mean_path)
        qlr_se = pd.read_csv(se_path)

        for enc in QLR_ENCODING_ORDER:
            row_m = qlr_mean[qlr_mean["encoding"] == enc]
            row_s = qlr_se[qlr_se["encoding"] == enc]

            if row_m.empty or row_s.empty:
                print(f"[WARN] Missing QLR row for {enc}, N={n}, L={L}")
                continue

            row_m = row_m.iloc[0]
            row_s = row_s.iloc[0]
            model = f"{ENCODING_TO_MODEL.get(enc, 'QLR_' + enc)} (L={L})"

            for name, cols in [
                ("summary", SUMMARY_COLS),
                ("strict_fpr", STRICT_FPR_COLS),
                ("calibration", CALIBRATION_COLS),
                ("runtime", RUNTIME_COLS),
            ]:
                out_row = {"Model": model}
                for col in cols:
                    if col in row_m.index and col in row_s.index:
                        digits = 2 if name == "runtime" else 3
                        out_row[col] = fmt(row_m[col], row_s[col], digits=digits)
                outputs[name].append(out_row)


def build_for_n(n):
    outputs = {
        "summary": [],
        "strict_fpr": [],
        "calibration": [],
        "runtime": [],
    }

    add_baseline_rows(n, outputs)
    add_qlr_rows(n, outputs)

    for name, rows in outputs.items():
        out_path = TABLES / f"{name}_N{n}_all_layers.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}")


def main():
    for n in N_VALUES:
        build_for_n(n)


if __name__ == "__main__":
    main()

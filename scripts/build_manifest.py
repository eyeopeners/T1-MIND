
import os, argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", default="")
    args = ap.parse_args()
    root = args.root
    t1_dir = os.path.join(root, "t1")
    mind_dir = os.path.join(root, "mind_csv")
    labels_csv = os.path.join(root, "labels_filtered_new1107.csv")
    overlay_path = os.path.join(root, args.overlay) if args.overlay else ""
    df = pd.read_csv(labels_csv)
    df["label"] = df["label"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["age"] = df["age"].astype(float)
    df["edu"] = df["edu"].astype(float)
    df["race"] = df["race"].astype(int)
    rows = []
    for sid in df["subject_id"].astype(str).values:
        t1 = os.path.join(t1_dir, f"{sid}.nii.gz")
        mind = os.path.join(mind_dir, f"{sid}_MIND_HCP-MMP-360.csv")
        if not (os.path.exists(t1) and os.path.exists(mind)):
            continue
        rows.append({"subject_id": sid, "t1w_path": t1, "mind_path": mind,
                     "overlay_path": overlay_path if os.path.exists(overlay_path) else ""})
    df_paths = pd.DataFrame(rows)
    out = df.merge(df_paths, on="subject_id", how="inner")
    out.rename(columns={"edu":"edu_years"}, inplace=True)
    out.to_csv(args.out, index=False)
    print(f"[build_manifest] {len(out)} subjects -> {args.out}")
if __name__ == "__main__": main()

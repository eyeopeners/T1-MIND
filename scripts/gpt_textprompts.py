# scripts/gpt_textprompts.py
# -*- coding: utf-8 -*-
import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="原始 textprompt.csv：列 class,text")
    ap.add_argument("--aug", required=True, help="mind_group_stats 生成的 textprompt_mind_aug.csv")
    ap.add_argument("--out", required=True, help="合并结果")
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    aug  = pd.read_csv(args.aug)
    if "class" not in base.columns or "text" not in base.columns:
        raise ValueError("--base 必须包含列 class,text")
    if "class" not in aug.columns or "text" not in aug.columns:
        raise ValueError("--aug 必须包含列 class,text")

    merged = base.merge(aug, on="class", how="left", suffixes=("_base","_aug"))
    merged["text"] = merged["text_base"].fillna("") + " " + merged["text_aug"].fillna("")
    merged[["class","text"]].to_csv(args.out, index=False, encoding="utf-8")
    print(f"[gpt_textprompts] merged -> {args.out}")

if __name__ == "__main__":
    main()

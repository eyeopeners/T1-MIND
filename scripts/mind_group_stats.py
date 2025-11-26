# scripts/mind_group_stats.py
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def read_mind_csv(path):
    df = pd.read_csv(path, index_col=0)
    mat = df.values.astype(np.float32)
    mat = (mat + mat.T)/2.0
    np.fill_diagonal(mat, 1.0)
    labels = list(df.columns)
    return mat, labels

def upper_tri_indices(N): return np.triu_indices(N, k=1)
def upper_tri_flat(mat):
    iu = upper_tri_indices(mat.shape[0]); return mat[iu]

def residualize_cols(Y, X):
    if X.size == 0: return Y
    Xm = X.mean(0, keepdims=True); Xs = X.std(0, keepdims=True)+1e-8
    Xn = (X - Xm)/Xs
    reg = LinearRegression()
    Yr = np.zeros_like(Y, dtype=np.float32)
    for e in tqdm(range(Y.shape[1]), desc="Residualizing by covariates"):
        reg.fit(Xn, Y[:,e]); Yr[:,e] = Y[:,e] - reg.predict(Xn)
    return Yr

def cohens_d(a,b):
    nx,ny=len(a),len(b); mx,my=a.mean(),b.mean()
    vx,vy=a.var(ddof=1),b.var(ddof=1)
    sp = np.sqrt(((nx-1)*vx+(ny-1)*vy)/(nx+ny-2+1e-8))
    return (mx-my)/(sp+1e-8)

def effect_edges(A,B):
    d = np.zeros(A.shape[1], dtype=np.float32)
    for e in tqdm(range(A.shape[1]), desc="Cohen d per edge"):
        d[e] = cohens_d(A[:,e], B[:,e])
    return d

def roi_summary(d_edge, N, labels):
    iu = np.triu_indices(N, k=1)
    roi_imp = np.zeros(N, dtype=np.float32)
    for idx,val in enumerate(d_edge):
        i,j = iu[0][idx], iu[1][idx]
        roi_imp[i] += abs(val); roi_imp[j] += abs(val)
    roi_imp = roi_imp/(N-1.0)
    out = pd.DataFrame({"roi": np.arange(N), "importance": roi_imp, "name": labels})
    return out.sort_values("importance", ascending=False)

def compare(edges, covs, k1, k2, N, labels, outdir, name, control_cov):
    A,B = edges[k1], edges[k2]
    AB = np.vstack([A,B]); X = np.vstack([covs[k1], covs[k2]])
    if control_cov and len(AB)>0:
        AB = residualize_cols(AB, X); A = AB[:len(A)]; B = AB[len(A):]
    d_edge = effect_edges(A,B)
    iu = np.triu_indices(N, k=1)
    pd.DataFrame({"i":iu[0],"j":iu[1],"cohend":d_edge}).to_csv(os.path.join(outdir,f"edges_{name}.csv"), index=False)
    roi = roi_summary(d_edge, N, labels)
    roi.to_csv(os.path.join(outdir, f"roi_{name}.csv"), index=False)
    return roi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--control_covariates", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.manifest)
    groups = {c: df[df["label"]==c] for c in [0,1,2]}
    seed = 0 if len(groups[0])>0 else (1 if len(groups[1])>0 else 2)
    seed_mat, labels = read_mind_csv(groups[seed].iloc[0]["mind_path"])
    N = seed_mat.shape[0]; iu = np.triu_indices(N,k=1); E = len(iu[0])

    edges = {}; covs={}
    for k in [0,1,2]:
        mats=[]; Xcov=[]
        for _,row in tqdm(groups[k].iterrows(), total=len(groups[k]), desc=f"Load group {k}"):
            m,_ = read_mind_csv(row["mind_path"]); mats.append(upper_tri_flat(m))
            Xcov.append([row["age"], row["sex"], row["edu_years"], row["race"]])
        edges[k] = np.vstack(mats).astype(np.float32) if mats else np.zeros((0,E), dtype=np.float32)
        covs[k]  = np.array(Xcov, dtype=np.float32) if Xcov else np.zeros((0,4), dtype=np.float32)

    roi_nc_ad  = compare(edges, covs, 0,2,N,labels,args.outdir,"NC_vs_AD", args.control_covariates)
    roi_nc_mci = compare(edges, covs, 0,1,N,labels,args.outdir,"NC_vs_MCI",args.control_covariates)

    # Sentences
    def topk(df_roi, title, k=10):
        names = df_roi["name"].tolist()[:k]
        return f"{title}：本组与对照的形态学协变差异主要集中在如下皮层区：{'、'.join(names)}。" if len(names)>0 else f"{title}：当前训练折未统计到显著差异。"

    line_ad  = topk(roi_nc_ad,  "AD 组相对于 NC")
    line_mci = topk(roi_nc_mci, "MCI 组相对于 NC")
    line_nc  = "NC 组：相较病程组，整体差异较小，作为对照组用于呈现基本皮层协变基线模式。"
    with open(os.path.join(args.outdir, "textprompt_mind_aug.csv"), "w", encoding="utf-8") as f:
        f.write("class,text\n")
        f.write(f'NC,"{line_nc}"\n')
        f.write(f'MCI,"{line_mci}"\n')
        f.write(f'AD,"{line_ad}"\n')
    print("[mind_group_stats] done.")

if __name__ == "__main__":
    main()
# scripts/dump_preds.py
import argparse, os, sys, glob, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# --- 把项目根目录加入 sys.path，避免 "No module named 'src'" ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 你的 yaml_config.py 里函数名是 load_config（不是 load_yaml）
from src.config.yaml_config import load_config
from src.data.dataset_2d import Neuro2DDataset
from src.models.ppal_mind import PPAL_MIND


def _cfg_get(cfg: dict, dotted_key: str, default=None):
    """支持 'a.b.c' 的安全取值；如果不存在就给 default。"""
    cur = cfg
    for k in dotted_key.split('.'):
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def _find_latest_ckpt(runs_dir='runs'):
    pats = [os.path.join(runs_dir, '**', '*.pt'),
            os.path.join(runs_dir, '**', '*.pth')]
    cands = []
    for p in pats:
        cands += glob.glob(p, recursive=True)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mind_ext.yaml')
    ap.add_argument('--ckpt', required=False, help='checkpoint 路径（若不提供，将自动在 runs/ 下寻找最新的）')
    ap.add_argument('--split', default='val', choices=['train','val','test'])
    ap.add_argument('--out', default='runs/val_preds.csv')
    ap.add_argument('--fold', type=int, default=-1)  # -1=全部折；>=0 指定单折
    args = ap.parse_args()

    cfg_raw = load_config(args.config)  # 你的函数名
    # -------- 兼容扁平/分组两种写法，统一取值 ----------
    manifest   = _cfg_get(cfg_raw, 'data.manifest',   cfg_raw.get('manifest'))
    slices     = _cfg_get(cfg_raw, 'data.slices',     cfg_raw.get('slices', 64))
    axis       = _cfg_get(cfg_raw, 'data.axis',       cfg_raw.get('axis', 'axial'))
    img_size   = _cfg_get(cfg_raw, 'data.img_size',   cfg_raw.get('img_size', 224))
    num_workers= _cfg_get(cfg_raw, 'train.num_workers', cfg_raw.get('num_workers', 4))
    eval_bs    = _cfg_get(cfg_raw, 'train.eval_bs',     cfg_raw.get('batch_size', 4))
    n_rois     = _cfg_get(cfg_raw, 'mind.n_rois', 360)  # 你的 YAML 里没给就默认 360

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 读 manifest，并在 train 上 fit scaler，保证 cov 标准化与训练一致 ---
    df_all = pd.read_csv(manifest)
    # split 列若不存在，则默认全是 'val'
    if 'split' not in df_all.columns:
        df_all['split'] = 'val'
    df_train = df_all[df_all['split'] == 'train'].copy()
    if df_train.empty:
        # 如果没有标注 train，至少用全体去 fit 一个 scaler（不会报错）
        df_train = df_all.copy()

    if args.split == 'train':
        df_split = df_train
    else:
        df_split = df_all[df_all['split'] == args.split].copy()

    # 支持按照 fold 过滤（manifest 里没有 fold 列就忽略）
    if (args.fold >= 0) and ('fold' in df_split.columns):
        df_split = df_split[df_split['fold'] == args.fold].copy()

    # 先在训练集 fit scaler
    scaler = StandardScaler()
    _ = Neuro2DDataset(df_train,
                       slices=slices,
                       axis=axis,
                       img_size=img_size,
                       n_rois=n_rois,
                       fit_scaler=True,
                       scaler=scaler)

    ds = Neuro2DDataset(df_split,
                        slices=slices,
                        axis=axis,
                        img_size=img_size,
                        n_rois=n_rois,
                        fit_scaler=False,
                        scaler=scaler)
    dl = DataLoader(ds,
                    batch_size=eval_bs,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=True)

    # --- 加载模型与权重 ---
    model = PPAL_MIND(cfg_raw).to(device).eval()
    ckpt_path = args.ckpt or _find_latest_ckpt('runs')
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'], strict=False)
            print(f"[info] loaded checkpoint: {ckpt_path}")
        else:
            model.load_state_dict(state, strict=False)
            print(f"[info] loaded state_dict: {ckpt_path}")
    else:
        print("[warn] 未提供 ckpt，且 runs/ 下未发现权重文件；将使用随机初始化模型导出预测（仅调试用途）")

    rows = []
    for batch in dl:
        img = batch['img'].to(device)         # (B,S,1,H,W)
        mind = batch['mind'].to(device)       # (B,R,R)
        cover = batch['slice_cover'].to(device)
        cov = batch['cov'].to(device)         # (B,10)
        y = batch['y'].cpu().numpy()
        sid = batch['subject_id']

        out = model({'img':img,'mind':mind,'slice_cover':cover,'cov':cov})
        prob = torch.softmax(out['logits_3c'], dim=-1).cpu().numpy()
        pNC,pMCI,pAD = prob[:,0],prob[:,1],prob[:,2]

        def renorm(a,b):
            s=a+b
            s[s==0]=1e-6
            return a/s
        nc_ad  = renorm(pNC,pAD)
        nc_mci = renorm(pNC,pMCI)
        mci_ad = renorm(pMCI,pAD)

        for i in range(len(y)):
            rows.append(dict(
                subject_id=sid[i],
                y=int(y[i]),
                p_nc=float(pNC[i]), p_mci=float(pMCI[i]), p_ad=float(pAD[i]),
                p_nc_ad=float(nc_ad[i]),
                p_nc_mci=float(nc_mci[i]),
                p_mci_ad=float(mci_ad[i])
            ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[done] saved: {args.out}")

if __name__ == "__main__":
    main()

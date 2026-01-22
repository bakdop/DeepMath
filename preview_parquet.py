from pathlib import Path
import pandas as pd
import numpy as np

PARQUET_PATH = "/data/boyan/DeepMath/data/train.parquet"  # <- 改这里
MAX_CHARS = 400

def short(v, max_chars=MAX_CHARS):
    # 1) 先处理 None
    if v is None:
        return "None"

    # 2) 处理 numpy/pandas 的缺失值（仅对标量）
    try:
        if np.isscalar(v) and pd.isna(v):
            return "NA"
    except Exception:
        pass

    # 3) 如果是 dict/list/tuple/set/ndarray 等，先给个简短结构信息
    if isinstance(v, dict):
        keys = list(v.keys())
        head = keys[:10]
        s = f"dict(len={len(keys)}, keys={head}{'...' if len(keys)>10 else ''})"
        body = str(v)
        if len(body) > max_chars:
            body = body[:max_chars] + f"... [truncated, len={len(str(v))}]"
        return s + "\n" + body

    if isinstance(v, (list, tuple, set)):
        vv = list(v)
        preview = vv[:10]
        s = f"{type(v).__name__}(len={len(vv)}, head={preview}{'...' if len(vv)>10 else ''})"
        body = str(v)
        if len(body) > max_chars:
            body = body[:max_chars] + f"... [truncated, len={len(str(v))}]"
        return s + "\n" + body

    if isinstance(v, np.ndarray):
        s = f"ndarray(shape={v.shape}, dtype={v.dtype})"
        body = np.array2string(v, threshold=20)
        if len(body) > max_chars:
            body = body[:max_chars] + f"... [truncated, len={len(np.array2string(v, threshold=np.inf))}]"
        return s + "\n" + body

    # 4) 普通标量：转字符串并截断
    s = str(v)
    if len(s) > max_chars:
        return s[:max_chars] + f"... [truncated, len={len(s)}]"
    return s

p = Path(PARQUET_PATH)
df = pd.read_parquet(p)

print(f"=== File: {p} ===")
print(f"Rows: {len(df)}  Cols: {df.shape[1]}\n")

print("=== Columns & dtypes ===")
for c in df.columns:
    print(f"- {c}: {df[c].dtype}")
print()

print("=== First 3 rows (field-by-field) ===")
n = min(3, len(df))
for i in range(n):
    print(f"\n--- row {i} (index={df.index[i]}) ---")
    row = df.iloc[i]
    for c in df.columns:
        print(f"{c}: {short(row[c])}")

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

# =============================
# Tokenize
# =============================
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

# =============================
# Suffix repetition detector
# =============================
def detect_suffix_repetition(
    tokens: List[str],
    min_pat_len: int = 6,
    max_pat_len: int = 500,   # <-- as requested
    max_repeat_times: int = 10
) -> Optional[Dict[str, Any]]:
    n = len(tokens)
    if n < 2 * min_pat_len:
        return None

    max_pat_len = min(max_pat_len, n // 2)

    # Try longer patterns first (more meaningful suffix repetition)
    for pat_len in range(max_pat_len, min_pat_len - 1, -1):
        pattern = tokens[-pat_len:]
        max_r = min(max_repeat_times, n // pat_len)

        r = 1
        while r < max_r:
            start = n - (r + 1) * pat_len
            mid   = n - r * pat_len
            if start < 0:
                break
            if tokens[start:mid] == pattern:
                r += 1
            else:
                break

        if r >= 2:
            cover = r * pat_len
            return {
                "pattern_len": pat_len,
                "repeat_times": r,
                "cover_tokens": cover,
                "suffix_start_idx": n - cover,
            }

    return None

# =============================
# Analyze one jsonl
# =============================
def repetition_ratio_for_jsonl(
    jsonl_path: str,
    min_pat_len: int = 6,
    max_pat_len: int = 500
) -> Tuple[int, int, float]:
    total = 0
    repeated = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            resp = obj.get("response", "")
            tokens = tokenize(resp)

            info = detect_suffix_repetition(tokens, min_pat_len=min_pat_len, max_pat_len=max_pat_len)
            total += 1
            if info is not None:
                repeated += 1

    ratio = (repeated / total) if total else 0.0
    return total, repeated, ratio

# =============================
# Scan root folder: global_step_x/val.jsonl
# =============================
_STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")

def scan_steps(root_dir: str) -> List[Dict[str, Any]]:
    """
    Returns list of dict:
      {
        "step": int,
        "total": int,
        "repeated": int,
        "ratio": float,
        "val_jsonl": str
      }
    Sorted by step.
    """
    rows: List[Dict[str, Any]] = []

    for name in os.listdir(root_dir):
        m = _STEP_DIR_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        jsonl_path = os.path.join(root_dir, name, "val.jsonl")
        if not os.path.isfile(jsonl_path):
            continue

        total, repeated, ratio = repetition_ratio_for_jsonl(
            jsonl_path,
            min_pat_len=6,
            max_pat_len=500
        )
        rows.append({
            "step": step,
            "total": total,
            "repeated": repeated,
            "ratio": ratio,
            "val_jsonl": jsonl_path,
        })

    rows.sort(key=lambda d: d["step"])
    return rows

# =============================
# Save results
# =============================
def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# =============================
# Plot & save
# =============================
def save_plot(rows: List[Dict[str, Any]], out_png: str, out_pdf: Optional[str] = None) -> None:
    steps = [r["step"] for r in rows]
    ratios = [r["ratio"] for r in rows]

    plt.figure()
    plt.plot(steps, ratios, marker="o")
    plt.xlabel("global_step")
    plt.ylabel("repetition ratio")
    plt.title("Suffix Repetition Ratio vs Global Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    if out_pdf:
        plt.savefig(out_pdf)
    plt.close()

# =============================
# Main
# =============================
if __name__ == "__main__":
    # 改成你的总目录（里面包含 global_step_x 子文件夹）
    ROOT_DIR = "/data/boyan/DeepMath/models/deepmath/qwen3-1.7B-Base-qwen3-4B-nonthinking"

    # 输出目录（默认放到 ROOT_DIR 下面）
    OUT_DIR = ROOT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = scan_steps(ROOT_DIR)
    if not rows:
        raise RuntimeError(f"No valid global_step_x/val.jsonl found under: {ROOT_DIR}")

    # 1) 存统计
    out_jsonl = os.path.join(OUT_DIR, "repetition_by_step.jsonl")
    out_json  = os.path.join(OUT_DIR, "repetition_by_step.json")
    # save_jsonl(out_jsonl, rows)
    save_json(out_json, rows)

    # 2) 存图（不show）
    out_png = os.path.join(OUT_DIR, "repetition_ratio_curve.png")
    # out_pdf = os.path.join(OUT_DIR, "repetition_ratio_curve.pdf")  # 不想要pdf就改成 None
    save_plot(rows, out_png=out_png, out_pdf=None)

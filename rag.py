"""小型日本語 RAG プロトタイプ（学習用）

流れ：データ読み込み → 分割 → 埋め込み → 保存 → 検索 Top-k → プロンプト組立 → 生成。
すべてローカル：Ollama の bge-m3 で埋め込み、gemma2:9b で生成。

使い方（本ディレクトリで）：
  uv run --with pandas --with pyarrow --with numpy python rag.py build      # インデックス構築
  uv run --with pandas --with pyarrow --with numpy python rag.py ask "質問"  # 質問
  uv run --with pandas --with pyarrow --with numpy python rag.py eval       # 検索評価 + 誤変換デモ

データ：JQaRA（hotchpotch/JQaRA）の dev.parquet。
  デフォルトでは ./data/dev.parquet を読む。環境変数 JQARA_PATH で別の場所を指定可。
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 設定 ----
DATA = Path(os.environ.get("JQARA_PATH") or Path(__file__).parent / "data" / "dev.parquet")
N_QUESTIONS = 60        # JQaRA 先頭 60 問。1 問あたり 50 の候補段落 → 約 3,000 段
CHUNK_SIZE = 400        # 1 チャンク最大 400 字
CHUNK_OVERLAP = 50      # 隣接チャンクの重なり 50 字（文が切れても両側が不完全にならないように）
EMBED_MODEL = "bge-m3"  # 埋め込みモデル：多言語（日本語対応）
GEN_MODEL = "gemma2:9b" # 生成モデル
TOP_K = 5               # 上位 5 件を取得
OLLAMA = "http://localhost:11434"
INDEX_DIR = Path(__file__).parent / "index"


def ollama(path, payload):
    """ローカル Ollama の HTTP API を呼ぶ。"""
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


# ---- 第 1 ステップ：データ読み込み ----
def load_passages():
    if not DATA.exists():
        sys.exit(f"データが見つかりません: {DATA}\n"
                 f"JQaRA の dev.parquet をダウンロードして上記のパスに置くか、"
                 f"環境変数 JQARA_PATH で別の場所を指定してください（DATA.md 参照）。")
    df = pd.read_parquet(DATA)
    qids = sorted(df.q_id.unique())[:N_QUESTIONS]
    df = df[df.q_id.isin(qids)]
    # 同じ段落が複数の問題の候補になることがあるので重複を除く
    passages = df.drop_duplicates("passage_row_id")[["passage_row_id", "title", "text"]]
    questions = df.drop_duplicates("q_id")[["q_id", "question", "answers"]]
    # 各問題の正解段落（JQaRA の人手アノテーション label=1）：評価で使う
    gold = df[df.label == 1].groupby("q_id").passage_row_id.apply(lambda x: set(map(int, x))).to_dict()
    return passages, questions, gold


# ---- 第 2 ステップ：分割 ----
def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """固定文字数で分割、隣接チャンクが overlap 字重なる。size 以下ならそのまま 1 段を返す。"""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text) - overlap, step)]


# ---- 第 3 ステップ：埋め込み ----
def embed(texts, batch=32):
    """テキストをベクトル化し、長さ 1 に正規化（内積＝コサイン類似度になる）。"""
    vecs = []
    for i in range(0, len(texts), batch):
        out = ollama("/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + batch]})
        vecs.extend(out["embeddings"])
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build():
    passages, _, _ = load_passages()
    chunks = []  # どの段落から来たかを記録
    for pid, title, text in passages.itertuples(index=False):
        for c in chunk(text):
            chunks.append({"passage_row_id": int(pid), "title": title, "text": c})
    print(f"段落 {len(passages)} 件、分割後 {len(chunks)} 段。埋め込み開始…")
    t = time.time()
    vecs = embed([f"{c['title']}\n{c['text']}" for c in chunks])
    print(f"埋め込み完了：{vecs.shape[0]} 件 × {vecs.shape[1]} 次元、{time.time() - t:.0f} 秒")
    # ---- 第 4 ステップ：保存 ----
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vecs)
    with open(INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"保存先：{INDEX_DIR}/")


def load_index():
    vecs = np.load(INDEX_DIR / "vectors.npy")
    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl")]
    return vecs, chunks


# ---- 第 5 ステップ：検索 Top-k ----
def search(query, vecs, chunks, k=TOP_K):
    q = embed([query])[0]
    scores = vecs @ q                      # 正規化済なので内積＝コサイン類似度
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


# ---- 第 6・7 ステップ：プロンプト組立、生成 ----
def ask(query):
    vecs, chunks = load_index()
    hits = search(query, vecs, chunks)
    print("【検索された段落】")
    for s, c in hits:
        print(f"  {s:.3f}  {c['title']}：{c['text'][:60]}…")
    context = "\n\n".join(f"[{i + 1}] {c['title']}\n{c['text']}" for i, (_, c) in enumerate(hits))
    prompt = ("以下の資料だけを使って、質問に日本語で簡潔に答えてください。"
              "資料に答えがなければ「資料からは分かりません」と答えてください。\n\n"
              f"資料：\n{context}\n\n質問：{query}\n答え：")
    out = ollama("/api/generate", {"model": GEN_MODEL, "prompt": prompt, "stream": False})
    print("\n【回答】\n" + out["response"].strip())


# ---- 評価：検索が正解段落を見つけられたか ----
def evaluate():
    vecs, chunks = load_index()
    _, questions, gold = load_passages()
    hit = {1: 0, 5: 0, 20: 0}
    for q_id, question, _ in questions.itertuples(index=False):
        found = [c["passage_row_id"] for _, c in search(question, vecs, chunks, k=20)]
        for k in hit:
            if gold.get(q_id, set()) & set(found[:k]):
                hit[k] += 1
    n = len(questions)
    print(f"{n} 問について、正解段落が上位 k に入った割合（Hit@k）：")
    for k, h in hit.items():
        print(f"  Hit@{k:<2} = {h / n:.2f}（{h}/{n}）")

    # 誤変換デモ：同じ検索語を、正しい表記・同音誤変換・かな止まりの 3 通りで検索
    print("\n【入力誤りデモ】")
    demo = [("疲労骨折とは", "披露骨折とは", "ひろうこっせつとは"),
            ("浸透圧の仕組み", "新党圧の仕組み", "しんとうあつの仕組み"),
            ("ハンガリーの首都", "ハンガリーの主都", "ハンガリーのしゅと")]
    for right, wrong, kana in demo:
        for label, q in (("正しい", right), ("誤変換", wrong), ("かな", kana)):
            top = search(q, vecs, chunks, k=3)
            print(f"  {label}：{q}")
            for s, c in top:
                print(f"     {s:.3f} {c['title']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "ask" and len(sys.argv) > 2:
        ask(sys.argv[2])
    elif cmd == "eval":
        evaluate()
    else:
        print(__doc__)

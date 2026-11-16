# 小型日本語 RAG プロトタイプ

ローカルで動く、約 200 行の最小構成 RAG（Retrieval-Augmented Generation）。
日本語 Wikipedia 約 3,000 段落を bge-m3 でベクトル化し、質問とのコサイン類似度で上位 5 件を取り出して Gemma 2 に回答させる。
入力誤り（同音誤変換、かな止まり）が検索結果にどう影響するかを観察するデモ付き。

**本プロジェクトは学習・研究目的のプロトタイプで、プロダクトではない。**

## 動作確認済みの結果

実機（Mac, Ollama）で本リポジトリのコードを上記「実行」の 3 コマンドで実行した結果（2026-11-16。2026-11-09 の初回実行と同じ数値）：

- 資料：JQaRA dev の先頭 60 問の候補段落、重複除去後 **2,989 段**（400 字超の段落はなく、分割後も 2,989 段）
- 埋め込み：bge-m3、**1,024 次元**、**88 秒**
- 検索：コサイン類似度、Top-5
- 検索評価（60 問）：**Hit@1 = 0.95（57/60）、Hit@5 = Hit@20 = 0.98（59/60）**
  - これは資料が 3,000 段と小規模で、かつ JQaRA の候補段落自体が埋め込み検索で選別されているため埋め込み検索に有利な評価。「精度 95%」のように外には言えない。あくまで「小規模条件下で正解段落を上位に検索できることを確認した」程度。
- 回答例：質問「『流氷の天使』と呼ばれる動物は何ですか」 → 検索で「オホーツク水族館」「クリオネ」等を取得、Gemma 2 は「クリオネ」と回答。
- 誤変換デモ（Top-3 のタイトル、括弧内は Top-1 のコサイン類似度）：

| 検索語 | 正しい表記 | 同音誤変換 | かな止まり |
|---|---|---|---|
| 疲労骨折とは | 疲労骨折 ×3（0.76） | 披露骨折：疲労骨折 / 半月板損傷 / 疲労骨折（0.52） | ひろうこっせつ：疲労骨折 / ぱいどん / こげぱん（0.49） |
| 浸透圧の仕組み | 受動輸送 / 塩生植物 …（0.65） | 新党圧：圧力 / ワシントンの辞任挨拶 …（0.45） | しんとうあつ：ぜんまい仕掛け / オナモミ …（0.51） |
| ハンガリーの首都 | ハンガリー / ブダペスト …（0.69） | 主都：ほぼ変化なし（0.66） | しゅと：ハンガリー / ドゥナウーイヴァーロシュ …（0.47） |

誤変換・かな止まりは類似度を下げ、無関係な段落が混ざることがある。影響は語によって異なる（「首都→主都」のように字形が近く他の手がかりが多い場合はほぼ影響しない）。

## 動かし方

### 前提

- macOS / Linux（Ollama が動く環境）
- [Ollama](https://ollama.com/)
- [uv](https://github.com/astral-sh/uv)（任意、推奨）

### セットアップ

```bash
# モデル取得
ollama pull bge-m3
ollama pull gemma2:9b

# データ取得（約 40 MB、ライセンスは DATA.md）
mkdir -p data
curl -L -o data/dev.parquet https://huggingface.co/datasets/hotchpotch/JQaRA/resolve/main/data/dev-00000-of-00001.parquet
```

### 実行

```bash
# インデックス構築（1 回だけ、約 90 秒）
uv run --with pandas --with pyarrow --with numpy python rag.py build

# 質問
uv run --with pandas --with pyarrow --with numpy python rag.py ask "「流氷の天使」と呼ばれる動物は何ですか"

# 検索評価 + 誤変換デモ
uv run --with pandas --with pyarrow --with numpy python rag.py eval
```

## 構成

全体で約 200 行、7 ステップ：

1. **読み込み**（`load_passages`）：JQaRA から段落を取り出す
2. **分割**（`chunk`）：長文を 400 字 / 重なり 50 字で分割（JQaRA は元から 400 字以下なので実質分割なし）
3. **埋め込み**（`embed`）：bge-m3 で 1,024 次元に、正規化（内積＝コサイン類似度）
4. **保存**（`build` 後半）：`index/vectors.npy` と `index/chunks.jsonl`
5. **検索**（`search`）：1 回の行列積 `vecs @ q` で Top-k
6. **プロンプト組立**（`ask`）：検索段落と質問を日本語テンプレートに
7. **生成**（`ask`）：Gemma 2 に渡して回答

コード内の `第 N ステップ` コメントが上の 7 ステップに対応。

## 設計上の選択

- **外部サービスを使わない**：ベクトル DB も API も立てない。`numpy` 配列をファイルに保存するだけ。目的は RAG の中身を見えやすくすること。
- **埋め込みと生成を同じ Ollama で統一**：環境構築を最小限に。
- **評価も入れる**：Hit@k は検索評価の最も基本的な指標。数字そのものより「どう計測するか」を示すのが目的。
- **誤変換デモ**：日本語入力の特性（同音異字、かな漢字変換）が検索に与える影響を具体例で示す。本格的な研究には規模・統計が必要だが、問題の実在は示せる。

## 使っているもの

- 埋め込みモデル：[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)（MIT）
- 生成モデル：[Gemma 2 9B](https://ai.google.dev/gemma)（[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）
- データ：[hotchpotch/JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA)（CC BY-SA 4.0 ほか、詳細は [DATA.md](DATA.md)）
- Python 依存：`pandas`、`pyarrow`、`numpy`

## ライセンス

MIT（コード本体）。詳細は [LICENSE](LICENSE)。
データのライセンスは [DATA.md](DATA.md) を参照。

---

## In English (short)

A ~200-line, local-only RAG prototype in Japanese. Embeds ~3,000 Japanese
Wikipedia passages from [JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA)
with [bge-m3](https://huggingface.co/BAAI/bge-m3), retrieves top-5 by cosine
similarity, and answers with Gemma 2 9B via [Ollama](https://ollama.com/).
Includes a Hit@k evaluation and a small demo showing how Japanese typo-level
input (homophone miss-conversions, kana-only queries) affects retrieval.

See `DATA.md` for how to fetch the dataset, and run:

```bash
ollama pull bge-m3
ollama pull gemma2:9b
uv run --with pandas --with pyarrow --with numpy python rag.py build
uv run --with pandas --with pyarrow --with numpy python rag.py ask "..."
uv run --with pandas --with pyarrow --with numpy python rag.py eval
```

Code: MIT. Data: see `DATA.md`.

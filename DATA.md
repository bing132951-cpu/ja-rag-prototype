# データとライセンス / Data and Licenses

本プロジェクトはデータをリポジトリに含めない。各自が元データを取得して利用する。

This repository contains code only. Download the data yourself.

## JQaRA

- 配布元：[hotchpotch/JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA) （Hugging Face）
- 作者：Yuichi Tateno（hotchpotch）
- ライセンス：
  - `question` / `answers` 列：元の [JAQKET](https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/) に由来し **CC BY-SA 4.0**
  - `title` / `text` 列（Wikipedia 段落）：Wikipedia の規定により **CC BY-SA 4.0 または GFDL**
- 本プロジェクトで使うのは `dev.parquet`。

### ダウンロード例

```bash
mkdir -p data
curl -L -o data/dev.parquet https://huggingface.co/datasets/hotchpotch/JQaRA/resolve/main/data/dev-00000-of-00001.parquet
```

（約 40 MB。2026-11-16 にこの URL から取得したファイルで動作確認済み。）

もしくは環境変数で場所を指定：

```bash
export JQARA_PATH=/path/to/dev.parquet
uv run --with pandas --with pyarrow --with numpy python rag.py build
```

## モデル

- 埋め込み：`bge-m3`（[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), MIT License）
- 生成：`gemma2:9b`（[Gemma Terms of Use](https://ai.google.dev/gemma/terms) に従う）

どちらも Ollama 経由で取得・実行する：

```bash
ollama pull bge-m3
ollama pull gemma2:9b
```

## 本リポジトリのコード

- ライセンス：MIT（`LICENSE` 参照）

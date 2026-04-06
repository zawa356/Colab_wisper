# WhisperX + pyannote 文字起こし・話者分離ツール

Google Colab（GPU）上で **WhisperX**（large-v3）と **pyannote.audio** を使い、
動画・音声ファイルの文字起こしと話者分離を行うツールです。

## 出力形式

| 形式 | 内容 |
| ---- | ---- |
| `.txt` | シンプルなテキスト（タイムスタンプ・話者名付き） |
| `.json` | タイムスタンプ・話者・単語情報をすべて含む構造化データ |
| `.srt` | 動画プレイヤー向け字幕ファイル |
| `.md` | 話者ごとに見出しを付けた Markdown |

---

## 使い方（Google Colab）

### ステップ 1: HuggingFace トークンの登録

> **重要**: トークンをコードに直接書かないでください。

1. [HuggingFace](https://huggingface.co/settings/tokens) でアクセストークン（read 権限）を発行する
2. Colab の左サイドバーから **「シークレット」（鍵アイコン）** を開く
3. 「+ 新しいシークレットを追加」をクリック
4. 名前: `HF_TOKEN`、値: 発行したトークン を入力して保存
5. 「ノートブックへのアクセス」トグルをオンにする

### ステップ 2: HuggingFace モデルの利用規約に同意

以下の 2 つのモデルページにアクセスし、それぞれ利用規約に同意してください（要ログイン）。

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> 同意しないと話者分離の実行時にエラーになります。

### ステップ 3: リポジトリのクローン・インストール・ドライブマウント（セル 1）

Colab の **セル 1** に以下を貼り付けて実行します。

> **仕組み**: `whisperx` も `pyannote.audio` も torch/numpy/pandas をダウングレードする
> 依存を持っています。両方を `--no-deps` で入れ、numpy/pandas は Colab 互換バージョンに
> 固定することで Colab の CUDA 最適化済み torch（2.10.0+cu128）を守ります。
> セル 1 は「初回のみインストール → 自動再起動 → 2回目以降はスキップ」という設計です。

```python
import os, sys

# ---- ドライブマウント ----
from google.colab import drive
drive.mount('/content/drive')

# ---- リポジトリをクローン（既にあればスキップ） ----
TOOL_DIR = "/content/whisper_tool"
if not os.path.exists(TOOL_DIR):
    os.system(f"git clone https://github.com/zawa356/Colab_wisper.git {TOOL_DIR}")
else:
    print(f"{TOOL_DIR} は既に存在します。クローンをスキップします。")

os.chdir(TOOL_DIR)
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

# ---- インストール（初回のみ） ----
try:
    import whisperx
    import pyannote.audio
    print("✓ インストール済みです。次のセルに進んでください。")
except ImportError:
    print("インストールします（10〜15分かかります）...")

    # Step 1: 本体のみ --no-deps でインストール（torch/numpy/pandas を守る）
    os.system("pip install -q whisperx --no-deps")
    os.system("pip install -q 'pyannote.audio>=3.1.0' --no-deps")

    # Step 2: numpy/pandas を Colab 互換バージョンに固定
    # （google-colab は pandas==2.2.2、numba は numpy<2.1 を要求）
    os.system("pip install -q 'numpy==2.0.2' 'pandas==2.2.2'")

    # Step 3: 音声処理系（torch 非依存）
    os.system("pip install -q 'faster-whisper>=1.2.0' 'ctranslate2>=4.5.0' 'av>=11' onnxruntime ffmpeg-python")

    # Step 4: NLP 系
    os.system("pip install -q nltk omegaconf tqdm")

    # Step 5: HuggingFace 系（バージョン固定）
    os.system("pip install -q 'transformers>=4.48.0,<5.1' 'huggingface-hub<1.0'")

    # Step 6: pyannote の依存パッケージ（torch 系は除外）
    os.system("pip install -q asteroid-filterbanks pyannote-core pyannote-database pyannote-metrics pyannote-pipeline pyannoteai-sdk pytorch-metric-learning torch-audiomentations torchmetrics optuna colorlog")
    os.system("pip install -q 'lightning>=2.4' lightning-utilities pytorch-lightning")

    print("\nインストール完了。ランタイムを再起動します...")
    os.kill(os.getpid(), 9)  # ← 初回のみここに到達して再起動
```

> **再起動後の動作**: セル 1 を再実行しても `インストール済み` と表示され、
> `os.kill` には到達しません。そのままセル 2 に進んでください。
>
> **警告について**: `whisperx requires torch~=2.8.0` などの WARNING が出ますが、
> 実際の動作には影響ありません（Colab の torch 2.10.0+cu128 で正常に動きます）。

### ステップ 4: ファイルのアップロード＋実行（セル 2）

Colab の **セル 2** に以下を貼り付けて実行します。
ファイル選択ダイアログが表示され、アップロード後にパスが自動取得されてそのまま処理が始まります。

対応形式: `mp4` / `wav` / `mp3` / `m4a`

```python
import os, sys
from google.colab import files

# ランタイム再起動後もスクリプトのディレクトリを確実に参照する
TOOL_DIR = "/content/whisper_tool"
os.chdir(TOOL_DIR)
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

# ---- ファイルをアップロード（ダイアログが開く） ----
uploaded = files.upload()
if not uploaded:
    raise ValueError("ファイルがアップロードされませんでした。")

# アップロードされたファイルのパスを自動取得
filename = list(uploaded.keys())[0]
INPUT_FILE = os.path.abspath(filename)

# 対応形式チェック
supported = {".mp4", ".wav", ".mp3", ".m4a"}
ext = os.path.splitext(filename)[1].lower()
if ext not in supported:
    raise ValueError(f"非対応の形式です: {ext}  対応形式: {supported}")

print(f"入力ファイル: {INPUT_FILE}")

# ---- 話者数の設定（0 = 自動推定） ----
NUM_SPEAKERS = 0  # 人数がわかる場合は数字を入力（例: 2）

# ---- 実行（import で直接呼び出し、エラーがそのまま表示される） ----
import importlib, run as _run
importlib.reload(_run)  # 再起動後のキャッシュをクリア

# sys.argv を差し替えて argparse に渡す
sys.argv = ["/content/whisper_tool/run.py", INPUT_FILE]
if NUM_SPEAKERS > 0:
    sys.argv += ["--speakers", str(NUM_SPEAKERS)]

_run.main()
```

#### `NUM_SPEAKERS` の設定方法

| 値 | 動作 |
| -- | ---- |
| `0` | pyannote が話者数を自動推定（デフォルト） |
| `2` | 2 人として固定（精度が上がる場合あり） |
| `3` | 3 人として固定 |

### ステップ 5: 出力ファイルの確認・ダウンロード

処理が完了すると、以下の 4 ファイルが自動的にダウンロードされます。

```text
your_audio.txt
your_audio.json
your_audio.srt
your_audio.md
```

ダウンロードされない場合は、Colab の **「ファイル」サイドバー**から
`/content/output/` フォルダを確認し、右クリックでダウンロードしてください。

---

## オプション引数

```bash
python run.py <input_file> [--speakers N] [--output_dir DIR] [--model MODEL] [--language LANG]
```

| 引数 | デフォルト | 説明 |
| ---- | --------- | ---- |
| `--speakers` | 自動 | 話者数を固定指定 |
| `--output_dir` | `/content/output` | 出力先ディレクトリ |
| `--model` | `large-v3` | Whisper モデルサイズ |
| `--language` | `ja` | 言語コード |

---

## セッション切れ対策

> Colab の無料プランはアイドル状態が続くと自動的にセッションが切れます。

- モデルは Google ドライブにキャッシュされるため、**セッション再起動後も再ダウンロード不要**です。
- 長時間の音声を処理する場合は、**Colab Pro / Pro+** の使用を推奨します。
- ランタイムが切れた場合は、セル 1（ドライブマウント + インストール）から再実行してください。
- 処理の進捗はセルの出力ログで確認できます。

---

## ライセンス

本リポジトリのコードは MIT ライセンスです。
使用しているモデル・ライブラリのライセンスは各プロジェクトを参照してください。

- [WhisperX](https://github.com/m-bain/whisperX) — BSD-4-Clause
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — MIT
- [OpenAI Whisper](https://github.com/openai/whisper) — MIT

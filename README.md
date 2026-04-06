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

### ステップ 1: Google ドライブのマウント

Colab のセルで以下を実行し、ドライブをマウントします。
モデルのキャッシュをドライブに保存することで、セッション再起動後の再ダウンロードを防ぎます。

```python
from google.colab import drive
drive.mount('/content/drive')
```

### ステップ 2: HuggingFace トークンの登録

> **重要**: トークンをコードに直接書かないでください。

1. [HuggingFace](https://huggingface.co/settings/tokens) でアクセストークン（read 権限）を発行する
2. Colab の左サイドバーから **「シークレット」（鍵アイコン）** を開く
3. 「+ 新しいシークレットを追加」をクリック
4. 名前: `HF_TOKEN`、値: 発行したトークン を入力して保存
5. 「ノートブックへのアクセス」トグルをオンにする

### ステップ 3: HuggingFace モデルの利用規約に同意

以下の 2 つのモデルページにアクセスし、それぞれ利用規約に同意してください（要ログイン）。

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> 同意しないと話者分離の実行時にエラーになります。

### ステップ 4: リポジトリのクローンとインストール

Colab の **セル 1** に以下を貼り付けて実行します。

> **重要**: `whisperx` をそのまま `pip install` すると、Colab にプリインストールされている
> CUDA 最適化済みの `torch` が古いバージョンに**ダウングレード**されてしまいます。
> 以下のコマンドでは `--no-deps` を使ってダウングレードを防いでいます。

```python
# ドライブマウント
from google.colab import drive
drive.mount('/content/drive')

# リポジトリをクローン
!git clone https://github.com/zawa356/Colab_wisper.git /content/whisper_tool
%cd /content/whisper_tool

# ① whisperx 本体を torch 上書きなしでインストール
!pip install whisperx --no-deps

# ② whisperx の依存パッケージ（torch 系・numpy・pandas は除外して Colab のものを使う）
!pip install \
    faster-whisper>=1.2.0 \
    ctranslate2>=4.5.0 \
    nltk>=3.9.1 \
    omegaconf>=2.3.0 \
    "transformers>=4.48.0,<5.0" \
    "huggingface-hub<1.0.0" \
    ffmpeg-python \
    tqdm

# ③ pyannote.audio（torch は既存のものを使う）
!pip install "pyannote.audio>=3.1.0"

# ④ インストール後にランタイムを再起動（必須）
import os
os.kill(os.getpid(), 9)
```

> セル実行後にランタイムが自動的に再起動されます。**再起動後はセル 1 の先頭から再実行不要です。**
> 次のセル（セル 2）から続けて実行してください。

### ステップ 5 & 6: ファイルのアップロード＋実行

Colab の **セル 2** に以下を貼り付けて実行します。
ファイル選択ダイアログが表示され、アップロード後にパスが自動取得されてそのまま処理が始まります。

対応形式: `mp4` / `wav` / `mp3` / `m4a`

```python
import os, subprocess, sys
from google.colab import files

# ---- ファイルをアップロード（ダイアログが開く） ----
uploaded = files.upload()
if not uploaded:
    raise ValueError("ファイルがアップロードされませんでした。")

# アップロードされたファイルのパスを自動取得
filename = list(uploaded.keys())[0]
INPUT_FILE = os.path.abspath(filename)  # /content/<filename> として解決される

# 対応形式チェック
supported = {".mp4", ".wav", ".mp3", ".m4a"}
ext = os.path.splitext(filename)[1].lower()
if ext not in supported:
    raise ValueError(f"非対応の形式です: {ext}  対応形式: {supported}")

print(f"入力ファイル: {INPUT_FILE}")

# ---- 話者数の設定（0 = 自動推定） ----
NUM_SPEAKERS = 0  # 人数がわかる場合は数字を入力（例: 2）

# ---- 実行 ----
cmd = [sys.executable, "run.py", INPUT_FILE]
if NUM_SPEAKERS > 0:
    cmd += ["--speakers", str(NUM_SPEAKERS)]

subprocess.run(cmd, check=True)
```

#### `NUM_SPEAKERS` の設定方法

| 値 | 動作 |
| -- | ---- |
| `0` | pyannote が話者数を自動推定（デフォルト） |
| `2` | 2 人として固定（精度が上がる場合あり） |
| `3` | 3 人として固定 |

### ステップ 7: 出力ファイルの確認・ダウンロード

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

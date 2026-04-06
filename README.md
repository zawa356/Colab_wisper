# WhisperX + pyannote 文字起こし・話者分離ツール

Google Colab（GPU）上で **WhisperX**（large-v3）と **pyannote.audio** を使い、
動画・音声ファイルの文字起こしと話者分離を行うツールです。

## 出力形式

| 形式 | 内容 |
|------|------|
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
from google.drive import drive
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

```python
# ドライブマウント
from google.colab import drive
drive.mount('/content/drive')

# リポジトリをクローン
!git clone https://github.com/your-username/your-repo-name.git /content/whisper_tool
%cd /content/whisper_tool

# 依存パッケージをインストール
!pip install -r requirements.txt
```

> `your-username/your-repo-name` は実際のリポジトリ名に変更してください。

### ステップ 5: 音声・動画ファイルのアップロード

Colab の左サイドバーにある **「ファイル」アイコン** から、
処理したいファイルを `/content/` 以下にアップロードします。

対応形式: `mp4` / `wav` / `mp3` / `m4a`

### ステップ 6: 実行

Colab の **セル 2** に以下を貼り付けて実行します。

```python
# 設定
INPUT_FILE = "/content/your_audio.mp4"  # アップロードしたファイルのパスに変更
NUM_SPEAKERS = 0  # 0 = 自動推定、人数がわかる場合は数字を入力（例: 2）

# 実行
import subprocess, sys

cmd = [sys.executable, "run.py", INPUT_FILE]
if NUM_SPEAKERS > 0:
    cmd += ["--speakers", str(NUM_SPEAKERS)]

subprocess.run(cmd, check=True)
```

#### `NUM_SPEAKERS` の設定方法

| 値 | 動作 |
|----|------|
| `0` | pyannote が話者数を自動推定（デフォルト） |
| `2` | 2 人として固定（精度が上がる場合あり） |
| `3` | 3 人として固定 |

### ステップ 7: 出力ファイルの確認・ダウンロード

処理が完了すると、以下の 4 ファイルが自動的にダウンロードされます。

```
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
|------|-----------|------|
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

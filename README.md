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
    # 既にある場合は最新コードを取得
    os.system(f"cd {TOOL_DIR} && git pull")

os.chdir(TOOL_DIR)
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

# ---- インストール（初回のみ。2回目以降はスキップ） ----
# インストール済みフラグファイルで管理する
FLAG = "/tmp/whisperx_installed"
if os.path.exists(FLAG):
    print("✓ インストール済みです。セル 2 に進んでください。")
else:
    print("パッケージをインストールします（5〜10分かかります）...")
    ret = os.system("pip install -r requirements.txt")
    if ret != 0:
        raise RuntimeError("pip install に失敗しました。上のログを確認してください。")
    open(FLAG, "w").close()  # フラグを立てる
    print("\n✓ インストール完了。ランタイムを再起動します...")
    os.kill(os.getpid(), 9)  # 再起動（フラグがあるので次回は再起動しない）
```

> **仕組み**: `/tmp/whisperx_installed` というファイルをフラグとして使います。
>
> - **初回**: インストール → 自動再起動（フラグが残る）
> - **再起動後**: フラグあり → スキップしてセル 2 へ
> - **新しいセッション**: `/tmp` がリセットされるので自動的に再インストール
>
> **WARNING が多数出ますが無視して問題ありません。**
> `whisperx requires torch~=2.8.0` などは機能に影響しない互換性警告です。

### ステップ 4: ファイル選択＋実行（セル 2）

Colab の **セル 2** に以下を貼り付けて実行します。
Google ドライブ内の対応ファイルをドロップダウンで選択し「▶ 実行」ボタンを押すだけです。

対応形式: `mp4` / `wav` / `mp3` / `m4a`

```python
import os, sys, importlib
import ipywidgets as widgets
from IPython.display import display

# ランタイム再起動後もスクリプトのディレクトリを確実に参照する
TOOL_DIR = "/content/whisper_tool"
os.chdir(TOOL_DIR)
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

# ================================================================
# ★ここを編集★ 検索するドライブのフォルダ（省略するとマイドライブ全体）
SEARCH_DIR = "/content/drive/MyDrive"
# ================================================================

SUPPORTED = {".mp4", ".wav", ".mp3", ".m4a"}

# ---- ドライブ内の対応ファイルを検索 ----
print(f"検索中: {SEARCH_DIR}")
found = []
for root, dirs, files in os.walk(SEARCH_DIR):
    dirs[:] = [d for d in dirs if not d.startswith('.')]  # 隠しフォルダをスキップ
    for f in files:
        if os.path.splitext(f)[1].lower() in SUPPORTED:
            full = os.path.join(root, f)
            label = full.replace(SEARCH_DIR + "/", "")  # 表示を短縮
            found.append((label, full))

if not found:
    print(f"対応ファイルが見つかりません（{SEARCH_DIR}）")
else:
    found.sort()
    print(f"{len(found)} 件見つかりました。")

    # ---- ウィジェット ----
    file_picker = widgets.Dropdown(
        options=found,
        description="ファイル:",
        layout=widgets.Layout(width="90%"),
        style={"description_width": "80px"},
    )
    speaker_slider = widgets.IntSlider(
        value=0, min=0, max=10, step=1,
        description="話者数:",
        style={"description_width": "80px"},
    )
    speaker_label = widgets.Label("← 0 = 自動推定")
    run_btn = widgets.Button(
        description="▶ 実行",
        button_style="success",
        layout=widgets.Layout(width="120px", margin="10px 0 0 0"),
    )
    out = widgets.Output()

    def on_run(_):
        with out:
            out.clear_output()
            INPUT_FILE = file_picker.value
            NUM_SPEAKERS = speaker_slider.value

            print(f"入力ファイル: {INPUT_FILE}")

            import run as _run
            importlib.reload(_run)

            sys.argv = ["/content/whisper_tool/run.py", INPUT_FILE]
            if NUM_SPEAKERS > 0:
                sys.argv += ["--speakers", str(NUM_SPEAKERS)]

            _run.main()

    run_btn.on_click(on_run)

    display(widgets.VBox([
        file_picker,
        widgets.HBox([speaker_slider, speaker_label]),
        run_btn,
        out,
    ]))
```

> **話者数スライダー**: 0 = pyannote が自動推定。人数が分かる場合は数字を指定すると精度が上がります。

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

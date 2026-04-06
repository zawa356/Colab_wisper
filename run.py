"""
run.py
------
WhisperX + pyannote.audio を使った文字起こし＋話者分離のメインスクリプト。
Google Colab (GPU) 上での実行を想定。
"""

import argparse
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# Google Colab 環境チェック
# -----------------------------------------------------------------------
def is_colab() -> bool:
    """Google Colab 環境で実行中かどうかを判定する。"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_hf_token() -> str:
    """
    HuggingFace トークンを取得する。
    Colab ではシークレット機能から、それ以外は環境変数から取得。
    """
    if is_colab():
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if not token:
                raise ValueError("Colab シークレットに HF_TOKEN が登録されていません。")
            return token
        except Exception as e:
            print(f"[エラー] HF_TOKEN の取得に失敗しました: {e}")
            print("Colab のシークレット機能で HF_TOKEN を登録してください。")
            sys.exit(1)
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[エラー] 環境変数 HF_TOKEN が設定されていません。")
            sys.exit(1)
        return token


def setup_drive_cache():
    """
    Google ドライブをモデルキャッシュ先に設定する。
    Colab 以外の環境ではスキップ。
    """
    if is_colab():
        cache_dir = "/content/drive/MyDrive/hf_cache"
        os.environ["HF_HOME"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[情報] HuggingFace キャッシュ先: {cache_dir}")
    else:
        print("[情報] Colab 以外の環境のため、ドライブキャッシュ設定をスキップします。")


def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    ffmpeg を使って入力ファイルを 16kHz モノラル WAV に変換する。

    Parameters
    ----------
    input_path : str
        入力ファイルパス（mp4/wav/mp3/m4a）
    output_path : str
        変換後の WAV ファイルパス

    Returns
    -------
    str
        変換後のファイルパス
    """
    import subprocess

    print(f"[情報] 音声を変換中: {input_path} -> {output_path}")
    cmd = [
        "ffmpeg",
        "-y",               # 既存ファイルを上書き
        "-i", input_path,
        "-ar", "16000",     # サンプリングレート 16kHz
        "-ac", "1",         # モノラル
        "-c:a", "pcm_s16le",  # 16bit WAV
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[エラー] ffmpeg の変換に失敗しました:\n{result.stderr}")
        sys.exit(1)
    print("[情報] 音声変換完了。")
    return output_path


def download_outputs(output_files: list):
    """
    Colab 環境で出力ファイルを自動ダウンロードする。

    Parameters
    ----------
    output_files : list
        ダウンロードするファイルパスのリスト
    """
    if is_colab():
        from google.colab import files
        for path in output_files:
            if os.path.exists(path):
                print(f"[情報] ダウンロード中: {path}")
                files.download(path)
            else:
                print(f"[警告] ファイルが見つかりません: {path}")
    else:
        print("[情報] Colab 以外の環境のため、自動ダウンロードをスキップします。")
        print("出力ファイル:", output_files)


# -----------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------
def parse_args():
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="WhisperX + pyannote による文字起こし＋話者分離"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="入力ファイルパス（mp4/wav/mp3/m4a）",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="話者数を固定指定する（省略時は pyannote が自動推定）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/output",
        help="出力ディレクトリ（デフォルト: /content/output）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper モデルサイズ（デフォルト: large-v3）",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="文字起こしの言語コード（デフォルト: ja）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ドライブキャッシュの設定（Colab のみ有効）
    setup_drive_cache()

    # HuggingFace トークンの取得
    hf_token = get_hf_token()

    # 入力ファイルの存在確認
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[エラー] 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 対応フォーマットの確認
    supported = {".mp4", ".wav", ".mp3", ".m4a"}
    if input_path.suffix.lower() not in supported:
        print(f"[エラー] 非対応のファイル形式です: {input_path.suffix}")
        print(f"対応形式: {', '.join(supported)}")
        sys.exit(1)

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声を WAV に変換（既に WAV でも再変換して正規化する）
    wav_path = str(output_dir / f"{input_path.stem}_16k.wav")
    convert_to_wav(str(input_path), wav_path)

    # ---------------------------------------------------------------
    # 文字起こし（WhisperX）
    # ---------------------------------------------------------------
    from utils.transcribe import transcribe_audio

    print("\n[ステップ 1/3] 文字起こしを開始します...")
    segments, info = transcribe_audio(
        wav_path,
        model_name=args.model,
        language=args.language,
        device="cuda",
    )
    print(f"[情報] 検出言語: {info.get('language', '不明')}")
    print(f"[情報] セグメント数: {len(segments)}")

    # ---------------------------------------------------------------
    # 話者分離（pyannote.audio）
    # ---------------------------------------------------------------
    from utils.diarize import diarize_audio

    print("\n[ステップ 2/3] 話者分離を開始します...")
    diarized_segments = diarize_audio(
        wav_path,
        segments=segments,
        hf_token=hf_token,
        num_speakers=args.speakers,  # None の場合は自動推定
    )
    print(f"[情報] 話者分離済みセグメント数: {len(diarized_segments)}")

    # ---------------------------------------------------------------
    # 結果の出力（4形式）
    # ---------------------------------------------------------------
    from utils.export import export_all

    print("\n[ステップ 3/3] 結果を出力します...")
    stem = input_path.stem
    output_files = export_all(diarized_segments, output_dir=str(output_dir), stem=stem)

    print("\n[完了] 出力ファイル一覧:")
    for f in output_files:
        print(f"  - {f}")

    # Colab で自動ダウンロード
    download_outputs(output_files)

    print("\n処理が完了しました。")


if __name__ == "__main__":
    main()

"""
utils/transcribe.py
-------------------
WhisperX を使った音声文字起こしモジュール。
アライメント（単語レベルのタイムスタンプ精度向上）も実施する。
"""

from typing import Any


def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",
    language: str = "ja",
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16",
) -> tuple[list[dict], dict]:
    """
    WhisperX で音声ファイルを文字起こしし、単語レベルのアライメントを行う。

    Parameters
    ----------
    audio_path : str
        WAV ファイルのパス（16kHz モノラル推奨）
    model_name : str
        使用する Whisper モデル（デフォルト: large-v3）
    language : str
        言語コード（デフォルト: ja）
    device : str
        使用デバイス（"cuda" または "cpu"）
    batch_size : int
        バッチサイズ（GPU メモリに応じて調整）
    compute_type : str
        演算精度（"float16" or "int8"）

    Returns
    -------
    tuple[list[dict], dict]
        (アライメント済みセグメントリスト, メタ情報)
    """
    import whisperx

    # -------------------------------------------------------------------
    # モデルのロード
    # -------------------------------------------------------------------
    print(f"[情報] WhisperX モデルをロード中: {model_name} (デバイス: {device})")
    try:
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    except Exception as e:
        print(f"[エラー] WhisperX モデルのロードに失敗しました: {e}")
        raise

    # -------------------------------------------------------------------
    # 音声のロード
    # -------------------------------------------------------------------
    print(f"[情報] 音声ファイルをロード中: {audio_path}")
    try:
        audio = whisperx.load_audio(audio_path)
    except Exception as e:
        print(f"[エラー] 音声ファイルのロードに失敗しました: {e}")
        raise

    # -------------------------------------------------------------------
    # 文字起こし（Whisper 推論）
    # -------------------------------------------------------------------
    print("[情報] 文字起こし実行中...")
    try:
        result = model.transcribe(audio, batch_size=batch_size, language=language)
    except Exception as e:
        print(f"[エラー] 文字起こしに失敗しました: {e}")
        raise

    detected_language = result.get("language", language)
    info = {"language": detected_language}
    print(f"[情報] 検出言語: {detected_language}")

    # GPU メモリを解放
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # アライメント（単語レベルのタイムスタンプ精度向上）
    # -------------------------------------------------------------------
    print("[情報] アライメント処理中（単語レベルのタイムスタンプ）...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
    except Exception as e:
        # アライメントに失敗した場合は元のセグメントを使用
        print(f"[警告] アライメントに失敗しました。元のセグメントを使用します: {e}")
        return result["segments"], info

    # GPU メモリを解放
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    segments = result_aligned.get("segments", result["segments"])
    print(f"[情報] アライメント完了。セグメント数: {len(segments)}")
    return segments, info

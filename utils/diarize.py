"""
utils/diarize.py
----------------
pyannote.audio を使った話者分離モジュール。
WhisperX のセグメントと話者ラベルを統合する。
"""

from typing import Optional


def diarize_audio(
    audio_path: str,
    segments: list[dict],
    hf_token: str,
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> list[dict]:
    """
    pyannote.audio で話者分離を行い、WhisperX のセグメントに話者ラベルを付与する。

    Parameters
    ----------
    audio_path : str
        WAV ファイルのパス（16kHz モノラル推奨）
    segments : list[dict]
        WhisperX から得たアライメント済みセグメントリスト
    hf_token : str
        HuggingFace アクセストークン
    num_speakers : int or None
        話者数を固定指定する（None の場合は pyannote が自動推定）
    min_speakers : int
        自動推定時の最小話者数
    max_speakers : int
        自動推定時の最大話者数

    Returns
    -------
    list[dict]
        話者ラベル付きセグメントリスト
        各要素: {"start": float, "end": float, "text": str, "speaker": str}
    """
    import whisperx

    # -------------------------------------------------------------------
    # 話者分離パイプラインのロード
    # -------------------------------------------------------------------
    print("[情報] pyannote.audio パイプラインをロード中...")
    try:
        # whisperx 3.x では DiarizationPipeline はサブモジュールにある
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            # フォールバック: whisperx のバージョンによってはトップレベルにある
            DiarizationPipeline = whisperx.DiarizationPipeline

        # 新しい HuggingFace API は use_auth_token → token に変更された
        try:
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device="cuda",
            )
        except TypeError:
            diarize_model = DiarizationPipeline(
                token=hf_token,
                device="cuda",
            )
    except Exception as e:
        print(f"[エラー] 話者分離パイプラインのロードに失敗しました: {e}")
        print("[ヒント] HuggingFace の利用規約に同意しているか確認してください。")
        print("  - pyannote/speaker-diarization-3.1")
        print("  - pyannote/segmentation-3.0")
        raise

    # -------------------------------------------------------------------
    # 話者分離の実行
    # -------------------------------------------------------------------
    print("[情報] 話者分離を実行中...")
    try:
        if num_speakers is not None:
            # 話者数を固定指定
            print(f"[情報] 話者数を固定指定: {num_speakers} 人")
            diarize_segments = diarize_model(
                audio_path,
                num_speakers=num_speakers,
            )
        else:
            # pyannote の自動推定
            print(f"[情報] 話者数を自動推定（範囲: {min_speakers}〜{max_speakers} 人）")
            diarize_segments = diarize_model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
    except Exception as e:
        print(f"[エラー] 話者分離の実行に失敗しました: {e}")
        raise

    # -------------------------------------------------------------------
    # WhisperX セグメントと話者ラベルの統合
    # -------------------------------------------------------------------
    print("[情報] 文字起こし結果と話者ラベルを統合中...")
    try:
        result = whisperx.assign_word_speakers(diarize_segments, {"segments": segments})
        diarized_segments = result.get("segments", segments)
    except Exception as e:
        print(f"[警告] 話者ラベルの統合に失敗しました。話者なしで続行します: {e}")
        # 話者ラベルなしでフォールバック
        diarized_segments = [
            {**seg, "speaker": "SPEAKER_UNKNOWN"} for seg in segments
        ]

    # speaker キーが存在しないセグメントに デフォルト値を付与
    for seg in diarized_segments:
        if "speaker" not in seg:
            seg["speaker"] = "SPEAKER_UNKNOWN"

    # 話者の種類をログ出力
    speakers = sorted(set(seg["speaker"] for seg in diarized_segments))
    print(f"[情報] 検出された話者: {speakers}")

    return diarized_segments

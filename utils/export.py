"""
utils/export.py
---------------
文字起こし・話者分離の結果を4種類のフォーマットで出力するモジュール。
出力形式: TXT / JSON / SRT / Markdown
"""

import json
import os
from pathlib import Path


def _format_timestamp(seconds: float, srt: bool = False) -> str:
    """
    秒数をタイムスタンプ文字列に変換する。

    Parameters
    ----------
    seconds : float
        秒数
    srt : bool
        True の場合 SRT 形式（hh:mm:ss,mmm）、False の場合 hh:mm:ss

    Returns
    -------
    str
        タイムスタンプ文字列
    """
    if seconds is None:
        seconds = 0.0
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600

    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def export_txt(segments: list[dict], output_path: str) -> str:
    """
    シンプルなテキスト形式で出力する。

    Parameters
    ----------
    segments : list[dict]
        話者分離済みセグメントリスト
    output_path : str
        出力ファイルパス

    Returns
    -------
    str
        出力ファイルパス
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg.get("text", "").strip()
        start = _format_timestamp(seg.get("start", 0))
        end = _format_timestamp(seg.get("end", 0))
        lines.append(f"[{start} --> {end}] {speaker}: {text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[情報] TXT を出力しました: {output_path}")
    return output_path


def export_json(segments: list[dict], output_path: str) -> str:
    """
    タイムスタンプ・話者情報付きの JSON 形式で出力する。

    Parameters
    ----------
    segments : list[dict]
        話者分離済みセグメントリスト
    output_path : str
        出力ファイルパス

    Returns
    -------
    str
        出力ファイルパス
    """
    output_data = []
    for i, seg in enumerate(segments):
        entry = {
            "id": i,
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
            "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
            "text": seg.get("text", "").strip(),
        }
        # 単語レベル情報があれば追加
        if "words" in seg:
            entry["words"] = seg["words"]
        output_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[情報] JSON を出力しました: {output_path}")
    return output_path


def export_srt(segments: list[dict], output_path: str) -> str:
    """
    SRT 字幕形式で出力する。

    Parameters
    ----------
    segments : list[dict]
        話者分離済みセグメントリスト
    output_path : str
        出力ファイルパス

    Returns
    -------
    str
        出力ファイルパス
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg.get("start", 0), srt=True)
        end = _format_timestamp(seg.get("end", 0), srt=True)
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg.get("text", "").strip()

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{speaker}] {text}")
        lines.append("")  # SRT の空行区切り

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[情報] SRT を出力しました: {output_path}")
    return output_path


def export_markdown(segments: list[dict], output_path: str) -> str:
    """
    話者ごとに見出しを付けた Markdown 形式で出力する。

    Parameters
    ----------
    segments : list[dict]
        話者分離済みセグメントリスト
    output_path : str
        出力ファイルパス

    Returns
    -------
    str
        出力ファイルパス
    """
    lines = ["# 文字起こし結果\n"]

    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg.get("text", "").strip()
        start = _format_timestamp(seg.get("start", 0))
        end = _format_timestamp(seg.get("end", 0))

        # 話者が変わったら新しい見出しを挿入
        if speaker != current_speaker:
            lines.append(f"\n## {speaker}\n")
            current_speaker = speaker

        lines.append(f"- `[{start} --> {end}]` {text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[情報] Markdown を出力しました: {output_path}")
    return output_path


def export_all(
    segments: list[dict],
    output_dir: str,
    stem: str,
) -> list[str]:
    """
    4種類すべての形式でファイルを出力する。

    Parameters
    ----------
    segments : list[dict]
        話者分離済みセグメントリスト
    output_dir : str
        出力先ディレクトリ
    stem : str
        出力ファイル名のベース（拡張子なし）

    Returns
    -------
    list[str]
        出力されたファイルパスのリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(output_dir) / stem

    output_files = []

    try:
        output_files.append(export_txt(segments, str(base) + ".txt"))
    except Exception as e:
        print(f"[警告] TXT の出力に失敗しました: {e}")

    try:
        output_files.append(export_json(segments, str(base) + ".json"))
    except Exception as e:
        print(f"[警告] JSON の出力に失敗しました: {e}")

    try:
        output_files.append(export_srt(segments, str(base) + ".srt"))
    except Exception as e:
        print(f"[警告] SRT の出力に失敗しました: {e}")

    try:
        output_files.append(export_markdown(segments, str(base) + ".md"))
    except Exception as e:
        print(f"[警告] Markdown の出力に失敗しました: {e}")

    return output_files

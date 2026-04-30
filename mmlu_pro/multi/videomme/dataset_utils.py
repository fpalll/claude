import functools
import os
import string
import tempfile
import zipfile
import torch
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset

# OpenAI / in-memory path uses this marker dict in message content (JSON-serializable).
# vLLM path must call materialize_zip_videos_in_messages() first to get real paths.
ZipVideoRef = Dict[str, str]  # {"_videomme_zip": path, "_member": "entry.mp4"}


def _read_zip_member(zip_path: str, member: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member)


def read_videomme_zip_video_bytes(ref: ZipVideoRef) -> bytes:
    """Read raw .mp4 bytes from a zip reference (no disk extract)."""
    return _read_zip_member(ref["_videomme_zip"], ref["_member"])


@functools.lru_cache(maxsize=32)
def _zip_mp4_index(zip_path: str) -> Dict[str, str]:
    """Map videoID (basename without .mp4) -> full member path inside the zip."""
    out: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if not name.lower().endswith(".mp4"):
                continue
            vid = os.path.splitext(os.path.basename(name))[0]
            out[vid] = name
    return out


@functools.lru_cache(maxsize=32)
def _zip_srt_index(zip_path: str) -> Dict[str, str]:
    """Map videoID -> .srt member path."""
    out: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if not name.lower().endswith(".srt"):
                continue
            vid = os.path.splitext(os.path.basename(name))[0]
            out[vid] = name
    return out


def resolve_videos_zip_list(videos_zip: Optional[Union[str, List[str]]]) -> List[str]:
    if videos_zip is None:
        return []
    if isinstance(videos_zip, str):
        return [videos_zip]
    return list(videos_zip)


def resolve_video_source(
    video_id: str,
    data_dir: str,
    videos_zip: Optional[Union[str, List[str]]] = None,
) -> Union[str, ZipVideoRef]:
    """Filesystem path under data_dir/videos, or a zip reference if archives are given."""
    path = os.path.join(data_dir, "videos", f"{video_id}.mp4")
    if os.path.isfile(path):
        return path
    for zp in resolve_videos_zip_list(videos_zip):
        if not os.path.isfile(zp):
            continue
        idx = _zip_mp4_index(zp)
        if video_id in idx:
            return {"_videomme_zip": zp, "_member": idx[video_id]}
    return path


def resolve_subtitle_source(
    video_id: str,
    data_dir: str,
    subtitles_zip: Optional[Union[str, List[str]]] = None,
) -> Union[str, ZipVideoRef, None]:
    path = os.path.join(data_dir, "subtitle", f"{video_id}.srt")
    if os.path.isfile(path):
        return path
    for zp in resolve_videos_zip_list(subtitles_zip):
        if not os.path.isfile(zp):
            continue
        idx = _zip_srt_index(zp)
        if video_id in idx:
            return {"_videomme_zip": zp, "_member": idx[video_id]}
    return None


def _video_source_to_temp_path(video_source: Union[str, ZipVideoRef]) -> Tuple[str, Optional[str]]:
    """Return (path_for_decord, temp_path_or_none_to_delete)."""
    if isinstance(video_source, str):
        return video_source, None
    raw = _read_zip_member(video_source["_videomme_zip"], video_source["_member"])
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.write(fd, raw)
    os.close(fd)
    return tmp, tmp


def materialize_zip_videos_in_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Replace zip-backed video dicts with temp .mp4 paths for vLLM / qwen_vl_utils.
    Returns (new_messages, temp_paths_to_delete_after_inference).
    """
    import copy

    out_msgs: List[Dict[str, Any]] = copy.deepcopy(messages)
    cleanup: List[str] = []

    for msg in out_msgs:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "video" not in block:
                continue
            v = block["video"]
            if isinstance(v, dict) and "_videomme_zip" in v:
                raw = _read_zip_member(v["_videomme_zip"], v["_member"])
                fd, tmp = tempfile.mkstemp(suffix=".mp4")
                os.write(fd, raw)
                os.close(fd)
                cleanup.append(tmp)
                block["video"] = tmp

    return out_msgs, cleanup

def load_videomme_dataset(data_dir, duration='short'):
    """
    Load the VideoMME dataset.
    
    Args:
        data_dir: Directory containing VideoMME data
        duration: Video duration type ('short', 'medium', or 'long')
    
    Returns:
        List of data samples
    """
    print(f"Loading VideoMME dataset with duration={duration}")
    
    total_data = []
    for item in load_dataset(data_dir)["test"]:
        if item['duration'] == duration:
            total_data.append(item)
    
    print(f"✓ Loaded {len(total_data)} samples with duration={duration}")
    return total_data

def extract_video_frames_with_timestamps(
    video_source: Union[str, ZipVideoRef],
    fps=2,
    min_frames=4,
    max_frames=512,
):
    """
    Extract frames from video and return their timestamps.

    Args:
        video_source: Path to .mp4, or zip ref dict {_videomme_zip, _member} (temp-extracted).
    """
    from decord import VideoReader

    path, tmp = _video_source_to_temp_path(video_source)
    try:
        video_reader = VideoReader(path, num_threads=1)
        video_len = len(video_reader)
        duration = video_len / video_reader.get_avg_fps()

        nframes = round(duration) * fps
        nframes = min(max(nframes, min_frames), max_frames, video_len // 2 * 2)

        indices = torch.linspace(0, video_len - 1, nframes).round().long().clamp(0, video_len - 1).tolist()
        frame_timestamps = video_reader.get_frame_timestamp(indices)[:, 0].tolist()
        return indices, frame_timestamps
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

def load_subtitles(subtitle_source: Union[str, ZipVideoRef, None], frame_timestamps):
    """
    Load subtitles and match them to video frames.

    Args:
        subtitle_source: Path to .srt, zip ref dict, or None / missing path.
        frame_timestamps: List of frame timestamps in seconds
    """
    import pysubs2

    if subtitle_source is None:
        return ""
    if isinstance(subtitle_source, dict) and "_videomme_zip" in subtitle_source:
        raw = _read_zip_member(subtitle_source["_videomme_zip"], subtitle_source["_member"])
        fd, tmp = tempfile.mkstemp(suffix=".srt")
        try:
            os.write(fd, raw)
            os.close(fd)
            subs = pysubs2.load(tmp, encoding="utf-8")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    else:
        if not isinstance(subtitle_source, str) or not os.path.exists(subtitle_source):
            return ""
        subs = pysubs2.load(subtitle_source, encoding="utf-8")
    subtitles = []
    
    for sub in subs:
        for frame_timestamp in frame_timestamps:
            if sub.start / 1000 < frame_timestamp and sub.end / 1000 > frame_timestamp:
                sub_text = sub.text.replace('\\N', ' ')
                if sub_text.strip():
                    subtitles.append(sub_text)
                    break
    
    return ' '.join(subtitles)

def build_videomme_prompt(
    data,
    data_dir,
    use_subtitle=False,
    fps=2,
    min_frames=4,
    max_frames=512,
    min_pixels=128 * 28 * 28,
    max_pixels=512 * 28 * 28,
    total_pixels=24576 * 28 * 28,
    sys_prompt=None,
    prompt_style: str = "default",
    videos_zip: Optional[Union[str, List[str]]] = None,
    subtitles_zip: Optional[Union[str, List[str]]] = None,
):
    """
    Build VideoMME prompt (consistent with original implementation).

    prompt_style:
      - default: original VideoMME wording ("Select the best answer...", letter only).
      - qwen35: Qwen3.5 demo style — question + Choices (A)/(B)/… + JSON answer hint.

    Args:
        data: Single data sample
        data_dir: VideoMME data directory
        use_subtitle: Whether to include subtitles
        fps: Frames per second
        min_frames: Minimum frames
        max_frames: Maximum frames
        min_pixels: Minimum pixels per frame
        max_pixels: Maximum pixels per frame
        total_pixels: Total pixels across all frames
        sys_prompt: Optional system prompt
        videos_zip: Optional path(s) to .zip archives containing .mp4 files (basename = videoID).
        subtitles_zip: Optional path(s) to .zip archives containing .srt files.

    Returns:
        Tuple of (messages, annotation)
    """
    video_id = data['videoID']
    duration = data['duration']
    domain = data['domain']
    sub_category = data["sub_category"]
    question = data['question']
    choices = data['options']
    answer = data['answer']
    question_id = data['question_id']
    
    video_ref = resolve_video_source(video_id, data_dir, videos_zip)
    subtitle_src = resolve_subtitle_source(video_id, data_dir, subtitles_zip)

    # Build choices text (original layout)
    choice_txt = "\n".join(choices)

    # Build prompt
    prompt = ""
    subtitle_ok = subtitle_src is not None and (
        isinstance(subtitle_src, dict)
        or (isinstance(subtitle_src, str) and os.path.isfile(subtitle_src))
    )
    if use_subtitle and subtitle_ok:
        _, frame_timestamps = extract_video_frames_with_timestamps(
            video_ref, fps=fps, min_frames=min_frames, max_frames=max_frames
        )
        subtitles = load_subtitles(subtitle_src, frame_timestamps)

        if subtitles:
            prompt = "This video's subtitles are listed below:\n"
            prompt += subtitles + "\n"

    if prompt_style == "qwen35":
        parsed_lines = []
        letters = []
        for i, choice in enumerate(choices):
            letter = string.ascii_uppercase[i]
            letters.append(letter)
            body = choice.split(".", 1)[1].strip() if "." in choice else choice.strip()
            parsed_lines.append(f"({letter}) {body}")
        letters_str = ",".join(letters)
        prompt += question.strip() + "\n\n"
        prompt += "Choices:\n" + "\n".join(parsed_lines)
        prompt += (
            "\n\nThink step by step before answering.\n"
            "Please show your choice in the answer field with only the choice letter, "
            'e.g., "answer": "C".\n'
            f'The value of "answer" must be exactly one of: {letters_str}.'
        )
    else:
        prompt += (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option."
        )
        prompt += f"\nQuestion: {question}\n{choice_txt}\nThe best answer is:"
    
    # Build video content
    video_content = {
        "video": video_ref,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "total_pixels": total_pixels,
        "fps": fps,
    }
    
    contents = [
        video_content,
        {
            "text": prompt
        }
    ]
    
    # Build messages
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    
    messages.append({
        "role": "user",
        "content": contents
    })
    
    # Build annotation
    assert answer in ['A', 'B', 'C', 'D', 'E']
    answer_id = ord(answer) - 65
    
    annotation = {
        "question": question,
        "choices": {
            string.ascii_uppercase[i]: choice.split(".", 1)[1].strip() 
            for i, choice in enumerate(choices)
        },
        "answer": answer,
        "answer_id": answer_id,
        "video_path": (
            video_ref
            if isinstance(video_ref, str)
            else f"zip:{os.path.basename(video_ref['_videomme_zip'])}:{video_ref['_member']}"
        ),
        "domain": domain,
        "sub_category": sub_category,
        "question_id": question_id
    }
    
    return messages, annotation


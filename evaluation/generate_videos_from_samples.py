from typing import Optional, Any, Dict
from pathlib import Path
from argparse import ArgumentParser

import torch
from torchaudio import load as read_audio
from torchvision.io import read_video
import numpy as np
import av


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file or dir",
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to the input audio file or dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output dir",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=21.5,
        help="Frames per second for the output video",
    )
    parser.add_argument(
        "--video_codec",
        type=str,
        default="libx264",
        help="Video codec to use",
    )
    parser.add_argument(
        "--audio_codec",
        type=str,
        default="aac",
        help="Audio codec to use",
    )
    parser.add_argument(
        "--audio_fps",
        type=float,
        default=22050,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pix_fmt",
        type=str,
        default="yuv420p",
    )
    return parser.parse_args()


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = a_stream.format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout
            )

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def main():
    args = get_args()
    input_video = Path(args.input_video)
    input_audio = Path(args.input_audio)
    output = Path(args.output)

    if input_audio.is_dir():
        input_audio = sorted(input_audio.rglob("*.wav"))
    else:
        input_audio = [input_audio]

    if input_video.is_dir():
        # search matching video files for audio files
        input_video = [
            Path(str(input_video / audio_path.name).replace("_sample_0.wav", ".mp4"))
            for audio_path in input_audio
        ]
    else:
        input_video = [input_video]

    assert len(input_video) == len(
        input_audio
    ), f"Number of videos ({len(input_video)}) and audios ({len(input_audio)}) must be the same"

    output.mkdir(exist_ok=True, parents=True)

    for video_path, audio_path in zip(input_video, input_audio):
        video, _, _ = read_video(video_path)
        audio, _ = read_audio(audio_path)
        output_path = output / video_path.name
        write_video(
            str(output_path),
            video,
            args.video_fps,
            args.video_codec,
            options={"crf": str(args.crf), "pix_fmt": args.pix_fmt},
            audio_array=audio,
            audio_fps=args.audio_fps,
            audio_codec=args.audio_codec,
        )


if __name__ == "__main__":
    main()

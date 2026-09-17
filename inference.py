#!/usr/bin/env python
# encoding: utf-8
#
# ONNX-only inference pipeline for Basic Pitch (Spotify).
# Adapted from basic_pitch.inference, TensorFlow/CoreML/TFLite paths removed.

import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import librosa
import onnxruntime as ort
import pretty_midi

from .constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from . import note_creation as infer


class Model:
    """Wraps an ONNX InferenceSession for the Basic Pitch model."""

    def __init__(self, model_path: Union[pathlib.Path, str]):
        self.model = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        # Cache output names discovered from the session
        self._output_names = [o.name for o in self.model.get_outputs()]
        self._input_name = self.model.get_inputs()[0].name

    def predict(self, x: npt.NDArray[np.float32]) -> Dict[str, npt.NDArray[np.float32]]:
        # basic-pitch ONNX graph output mapping:
        #   note    -> StatefulPartitionedCall:1
        #   onset   -> StatefulPartitionedCall:2
        #   contour -> StatefulPartitionedCall:0
        outputs = self.model.run(
            [
                "StatefulPartitionedCall:1",
                "StatefulPartitionedCall:2",
                "StatefulPartitionedCall:0",
            ],
            {self._input_name: x},
        )
        return {
            k: v
            for k, v in zip(["note", "onset", "contour"], outputs)
        }


def window_audio_file(
    audio_original: npt.NDArray[np.float32], hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float]]]:
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(
                window,
                pad_width=[[0, AUDIO_N_SAMPLES - len(window)]],
            )
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        yield np.expand_dims(window, axis=-1), window_time


def get_audio_input(
    audio_path: Union[pathlib.Path, str], overlap_len: int, hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float], int]]:
    assert overlap_len % 2 == 0, f"overlap_length must be even, got {overlap_len}"

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    for window, window_time in window_audio_file(audio_original, hop_size):
        yield np.expand_dims(window, axis=0), window_time, original_length


def unwrap_output(
    output: npt.NDArray[np.float32],
    audio_original_length: int,
    n_overlapping_frames: int,
) -> np.array:
    if len(output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]

    output_shape = output.shape
    n_output_frames_original = audio_original_length // FFT_HOP
    unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]


def run_inference(
    audio_path: Union[pathlib.Path, str],
    model: Model,
) -> Dict[str, np.array]:
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    # Advance by an integer number of model frames. Each window contributes
    # exactly 141 frames; the remaining frame overlaps the next window.
    hop_size = 141 * FFT_HOP

    output: Dict[str, Any] = {"note": [], "onset": [], "contour": []}
    for audio_windowed, _, audio_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        for k, v in model.predict(audio_windowed).items():
            output[k].append(v)

    unwrapped_output = {
        k: unwrap_output(np.concatenate([part[:, 15:156, :] for part in output[k]]),
                         audio_original_length, 0) for k in output
    }

    return unwrapped_output


def predict(
    audio_path: Union[pathlib.Path, str],
    model: Model,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = 120,
    include_pitch_bends: bool = True,
    model_output: Optional[Dict[str, np.array]] = None,
) -> Tuple[
    Dict[str, np.array],
    pretty_midi.PrettyMIDI,
    List[Tuple[float, float, int, float, Optional[List[int]]]],
]:
    if model_output is None:
        model_output = run_inference(audio_path, model)
    if model_output["note"].shape[0] < 3:
        return model_output, pretty_midi.PrettyMIDI(initial_tempo=midi_tempo), []
    min_note_len = int(np.ceil(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        include_pitch_bends=include_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=midi_tempo,
    )

    return model_output, midi_data, note_events

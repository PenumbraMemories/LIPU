#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from enum import Enum

FFT_HOP = 256
N_FFT = 8 * FFT_HOP

NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
ANNOTATIONS_BASE_FREQUENCY = 27.5
ANNOTATIONS_N_SEMITONES = 88
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP


def _freq_bins(bins_per_semitone: int, base_frequency: float, n_semitones: int) -> np.array:
    d = 2.0 ** (1.0 / (12 * bins_per_semitone))
    bin_freqs = base_frequency * d ** np.arange(bins_per_semitone * n_semitones)
    return bin_freqs


FREQ_BINS_NOTES = _freq_bins(NOTES_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
FREQ_BINS_CONTOURS = _freq_bins(CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"

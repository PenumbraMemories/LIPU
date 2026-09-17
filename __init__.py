"""Basic Pitch ONNX engine.

A self-contained, TensorFlow-free re-implementation of the Spotify Basic Pitch
inference pipeline using onnxruntime.
"""
import os
import pathlib

from .inference import Model, predict, run_inference

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "nmp.onnx"

__all__ = ["Model", "predict", "run_inference", "MODEL_PATH"]

"""Platform-independent audio-to-MIDI entry point for LiPu 1.0.0."""
import argparse
from pathlib import Path
from bp_core import MODEL_PATH, Model, predict


def main():
    parser = argparse.ArgumentParser(description="璃谱 1.0.0 音频转 MIDI 核心")
    parser.add_argument("audio", type=Path)
    parser.add_argument("midi", type=Path)
    parser.add_argument("--onset", type=float, default=0.5)
    parser.add_argument("--frame", type=float, default=0.3)
    parser.add_argument("--minimum-ms", type=float, default=127)
    parser.add_argument("--minimum-hz", type=float, default=50)
    parser.add_argument("--maximum-hz", type=float, default=2000)
    parser.add_argument("--bends", action="store_true")
    parser.add_argument("--no-melodia", action="store_true")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error("音频文件不存在")
    if not 0.05 <= args.onset <= 1 or not 0.05 <= args.frame <= 1:
        parser.error("阈值须在 0.05 到 1 之间")
    if not 20 <= args.minimum_ms <= 500 or not 20 <= args.minimum_hz < args.maximum_hz <= 4000:
        parser.error("音符长度或频率范围无效")
    if not MODEL_PATH.is_file():
        parser.error("缺少 models/nmp.onnx")
    model = Model(MODEL_PATH)
    _, midi, notes = predict(args.audio, model, onset_threshold=args.onset,
                             frame_threshold=args.frame,
                             minimum_note_length=args.minimum_ms,
                             minimum_frequency=args.minimum_hz,
                             maximum_frequency=args.maximum_hz,
                             include_pitch_bends=args.bends,
                             melodia_trick=not args.no_melodia)
    args.midi.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.midi))
    print(f"已保存 {len(notes)} 个音符：{args.midi}")


if __name__ == "__main__":
    main()

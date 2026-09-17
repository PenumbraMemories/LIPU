璃谱 1.0.0 · 通用核心源码

本目录只包含平台无关的音频转 MIDI 核心、命令行入口、模型与依赖清单。不含 Windows 界面、Android 工程、构建缓存或签名材料。

安装：pip install -r requirements.txt
运行：python core_cli.py 输入音频.wav 输出.mid
查看参数：python core_cli.py --help

models/nmp.onnx 是运行所需的模型资源，不是源代码。核心按 22,050 Hz 音频的完整采样长度逐窗口推理，窗口每次前进 141 个 256 采样帧，输出连续拼接，不重复任何窗口。该模型仍可能对复杂音乐漏识别或误识别音符。

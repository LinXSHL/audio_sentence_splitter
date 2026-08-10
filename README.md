# 音频按句子拆分工具

这是一个本地运行的桌面/命令行工具：使用 Whisper 识别人声和词级时间戳，按句末标点、静音停顿和最长时长切分句子，再从原音频导出每句一个文件。

## 特点

- 中文、英文及 Whisper 支持的其他主流语言
- 本地识别，不调用收费云端 API；模型首次下载后可离线使用
- CPU 可运行，检测到可用 NVIDIA CUDA 时自动加速，CUDA 初始化失败时自动回退 CPU
- 词级时间戳 + 标点/静音联合分句，句间切点不会互相重叠
- 自带跨平台 FFmpeg 二进制依赖，不要求系统单独安装 FFmpeg
- 同时提供 CLI 和 Tkinter 图形界面
- 每句生成纯编号音频和同名 TXT，并额外生成文字总表与结构化 JSON
- 输出 WAV、FLAC 或 MP3；默认 WAV 为 16-bit PCM

## 环境与安装

需要 Python 3.11 或更高版本；本项目已在 Windows 11 / Python 3.12.2 上开发和验证。建议使用虚拟环境：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS/Linux：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

> `faster-whisper` 当前的 CUDA 运行时要求可能随版本变化。自动模式会优先尝试 CUDA，缺少兼容的 NVIDIA CUDA/cuDNN 库时会回退到 CPU；CPU 不需要安装 CUDA。

## 图形界面

安装后运行：

```powershell
audio-sentence-split-gui
```

也可以使用：

```powershell
python -m audio_sentence_splitter --gui
```

不带参数运行 `python -m audio_sentence_splitter` 也会直接打开图形界面。

Windows 用户还可以双击仓库根目录的 `run_gui.bat`（它优先使用 `.venv`）。

## 命令行

最简单的用法：

```powershell
audio-sentence-split "D:\audio\example.mp3"
```

指定输出目录、语言和模型：

```powershell
audio-sentence-split input.m4a -o output --language zh --model small
```

强制 CPU、使用英文并导出 FLAC：

```powershell
audio-sentence-split input.wav --device cpu --language en --format flac
```

模型已经下载后，强制禁止联网查找模型：

```powershell
audio-sentence-split input.wav --offline
```

查看全部参数：

```powershell
audio-sentence-split --help
```

默认输出目录为输入文件旁的 `<文件名>_sentences`：

```text
001_.wav
001_.txt
002_.wav
002_.txt
003_.wav
003_.txt
transcript.txt
segments.json
```

每个逐句 TXT 使用 Windows 记事本友好的 UTF-8 BOM 编码，内容格式如下：

```text
大家好，欢迎来到我的频道。  00:00.000 - 00:05.000
```

`transcript.txt` 按编号汇总全部句子。`segments.json` 保存识别语言、运行设备、模型、每句话的文字、识别时间和带留白的实际导出时间，方便检查和二次处理。

## 模型选择

- `tiny` / `base`：下载小、速度快，准确率较低
- `small`：默认值，CPU 可用，准确率与资源占用较均衡
- `medium` / `large-v3`：更准确，但下载、内存和计算需求更高
- `large-v3-turbo`：速度和准确率兼顾，但资源需求仍高于 `small`

首次运行会从 Hugging Face 下载所选模型。下载完成后缓存由 Hugging Face 管理，后续可以离线运行。

## 提高拆分准确率

- 清晰人声、较少混响和背景音乐会显著改善结果。
- 没有标点的识别文本会按静音间隔拆分；可用 `--sentence-gap` 调节，默认 `0.8` 秒。
- 单句超长时会优先在逗号等弱标点或词间停顿处切分；可用 `--max-sentence-duration` 调节。
- 专有名词较多时，可用 `--initial-prompt "专有名词列表"` 提示模型。
- 输出边缘默认保留少量声音，可用 `--padding-before` 和 `--padding-after` 调整。

## 开发与测试

运行完整测试：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q audio_sentence_splitter
```

真实 FFmpeg 导出测试在项目依赖安装后自动启用。完整语音识别验收需要首次下载一个 Whisper 模型。

## 隐私说明

音频内容由本机上的 `faster-whisper` 处理。除首次下载模型外，本项目不会主动上传音频或识别结果。使用 `--offline` 可要求模型加载器只使用本地文件。

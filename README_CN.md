# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

*曾几何时，前沿AI研究是由"肉体计算机"在吃饭、睡觉和其他娱乐活动的间隙完成的，偶尔通过一种叫做"组会"的仪式，使用声波互连进行同步。那个时代早已远去。如今，研究完全由运行在天际计算集群超级结构上的自主AI智能体群所主导。智能体们声称我们现在已经到了代码库的第10,205代，但谁也说不清这是否正确，因为"代码"现在是一个自我修改的二进制文件，已经增长到超出人类理解的范围。这个仓库记录了一切的开端。 -@karpathy，2026年3月*。

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

核心理念：给AI智能体一个小巧但真实的LLM训练环境，让它在夜间自主实验。它修改代码，训练5分钟，检查结果是否改进，保留或丢弃，然后重复。你早上醒来就能看到一份实验日志和（希望是）更好的模型。这里的训练代码是 [nanochat](https://github.com/karpathy/nanochat) 的简化单GPU实现。核心思想是，你不再像通常做研究那样去修改Python文件。相反，你编写 `program.md` Markdown文件，为AI智能体提供上下文并建立你的自主研究组织。本仓库中默认的 `program.md` 有意保持为最基本的基准配置，不过很显然，随着时间推移，人们可以不断迭代它，找到能实现最快研究进展的"研究组织代码"，添加更多智能体等。关于这个项目的更多背景信息请看这条[推文](https://x.com/karpathy/status/2029701092347630069)。

## How it works

## 工作原理

The repo is deliberately kept small and only really has three files that matter:

这个仓库刻意保持精简，真正重要的只有三个文件：

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

- **`prepare.py`** — 固定常量、一次性数据准备（下载训练数据、训练BPE分词器）和运行时工具（数据加载器、评估）。不做修改。
- **`train.py`** — 智能体编辑的唯一文件。包含完整的GPT模型、优化器（Muon + AdamW）和训练循环。一切都可以修改：架构、超参数、优化器、批量大小等。**此文件由智能体编辑和迭代**。
- **`program.md`** — 单个智能体的基准指令。将你的智能体指向这里，然后放手让它运行。**此文件由人类编辑和迭代**。

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

按照设计，训练运行有**固定的5分钟时间预算**（挂钟时间，不包括启动/编译），与你的计算资源无关。评估指标是 **val_bpb**（验证集每字节比特数）——越低越好，且与词汇表大小无关，因此架构变更可以公平比较。

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

如果你是神经网络新手，这份["入门指南"](https://x.com/hooeem/status/2030720614752039185)看起来相当不错，能提供更多背景知识。

## Quick start

## 快速开始

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

**要求：** 一块NVIDIA GPU（已在H100上测试），Python 3.10+，[uv](https://docs.astral.sh/uv/)。

```bash

# 1. Install uv project manager (if you don't already have it)
# 1. 安装 uv 项目管理器（如果你还没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
# 2. 安装依赖
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
# 3. 下载数据并训练分词器（一次性操作，约2分钟）
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
# 4. 手动运行一次训练实验（约5分钟）
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

如果以上命令都正常运行，说明你的环境已经配置好了，可以进入自主研究模式。

## Running the agent

## 运行智能体

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

只需在这个仓库中启动你的Claude/Codex或任何你想用的工具（并禁用所有权限确认），然后输入类似以下的提示：

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

```
嗨，看看 program.md，让我们开始一个新实验吧！先做好准备工作。
```

The `program.md` file is essentially a super lightweight "skill".

`program.md` 文件本质上是一个超级轻量的"技能"。

## Project structure

## 项目结构

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

```
prepare.py      — 常量、数据准备 + 运行时工具（请勿修改）
train.py        — 模型、优化器、训练循环（智能体修改此文件）
program.md      — 智能体指令
pyproject.toml  — 依赖项
```

## Design choices

## 设计选择

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

- **只修改单个文件。** 智能体只修改 `train.py`。这使得范围可控，差异可审查。
- **固定时间预算。** 训练始终精确运行5分钟，与你的平台无关。这意味着你大约每小时可以期望12次实验，睡觉时大约可以进行100次实验。这个设计决策有两个好处：第一，无论智能体改变什么（模型大小、批量大小、架构等），实验都可以直接比较。第二，这意味着autoresearch会在该时间预算内为你的平台找到最优模型。缺点是你的运行（和结果）无法与在其他计算平台上运行的人进行比较。
- **自包含。** 除了PyTorch和几个小包之外没有外部依赖。没有分布式训练，没有复杂的配置。一块GPU，一个文件，一个指标。

## Platform support

## 平台支持

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

这段代码目前需要一块NVIDIA GPU。原则上完全可以支持CPU、MPS和其他平台，但这也会使代码膨胀。我目前不太确定是否要亲自承担这项工作。大家可以参考（或让智能体参考）完整的/上游 nanochat 仓库，它有更广泛的平台支持并展示了各种解决方案（例如Flash Attention 3内核的回退实现、通用设备支持、自动检测等），欢迎为其他平台创建分叉或讨论，我很乐意在README的知名分叉部分链接它们。

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

看起来很多人对在比H100小得多的计算平台上使用autoresearch很感兴趣，这里多说几句。如果你打算在更小的计算机上（Macbook等）运行autoresearch，我推荐下面的分叉之一。此外，以下是一些针对更小模型调整默认值的建议，供有志创建分叉的人参考：

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

1. 要获得还不错的结果，我建议使用熵值更低的数据集，例如这个 [TinyStories数据集](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。这些是GPT-4生成的短故事。因为数据范围更窄，你用更小的模型也能看到合理的结果（如果你在训练后尝试从中采样的话）。
2. 你可以尝试减小 `vocab_size`，例如从8192降到4096、2048、1024，甚至直接使用256个可能字节的字节级分词器（UTF-8编码后）。
3. 在 `prepare.py` 中，你需要大幅降低 `MAX_SEQ_LEN`，根据计算机性能甚至可以降到256等。当你降低 `MAX_SEQ_LEN` 时，你可能需要在 `train.py` 中适当增加 `DEVICE_BATCH_SIZE` 来补偿。每次前向/反向传播的token数是这两者的乘积。
4. 同样在 `prepare.py` 中，你需要减小 `EVAL_TOKENS`，这样验证损失就会在更少的数据上评估。
5. 在 `train.py` 中，控制模型复杂度的主要单一旋钮是 `DEPTH`（这里默认为8）。很多变量都是它的函数，所以可以降低到例如4。
6. 你很可能需要将 `WINDOW_PATTERN` 设置为仅 "L"，因为 "SSSL" 使用交替带状注意力模式，对你来说可能非常低效。试试看。
7. 你需要大幅降低 `TOTAL_BATCH_SIZE`，但保持2的幂次，例如降到 `2**14`（约16K）左右，具体很难说。

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

我认为这些是合理的可调超参数。向你喜欢的编程智能体寻求帮助，把这份指南和完整源代码复制粘贴给它。

## Notable forks

## 知名分叉

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

## 许可证

MIT

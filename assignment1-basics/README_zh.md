# CS336 2025 春季 作业 1：Basics（基础）

> 中文翻译版，原文见 [README.md](./README.md)。翻译仅作学习辅助，一切以英文原版和作业 PDF 手册为准。

作业的完整说明请见官方作业手册：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果你发现作业手册或代码中有任何问题，欢迎提 GitHub issue，或者直接开 PR 修复。

## 环境搭建

### 虚拟环境

课程统一使用 `uv` 管理环境，以保证可复现、可移植、易用。
请先安装 `uv`（推荐方式见 [这里](https://github.com/astral-sh/uv#installation)），也可以直接 `pip install uv` 或 `brew install uv`。
推荐顺手读一下 uv 的项目管理文档（[链接](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)），不会后悔。

装好之后，仓库里任意代码都可以这样运行：

```sh
uv run <python_file_path>
```

`uv` 会在需要时自动解析依赖并激活环境。

### 运行单元测试

```sh
uv run pytest
```

**一开始所有测试都会失败**，并抛出 `NotImplementedError`——这是正常的。
你需要先补全 [./tests/adapters.py](./tests/adapters.py) 中的函数，把你自己的实现接入测试。

### 下载数据

下载 TinyStories 数据以及 OpenWebText 的一个子集：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 目录结构速览

```
assignment1-basics/
├── cs336_basics/          # 你需要在这里实现 BPE tokenizer、Transformer、优化器等
├── tests/
│   ├── adapters.py        # 把你的实现接入测试的「适配层」，先改这里
│   └── _snapshots/        # 参考输出快照（snapshot testing）
├── cs336_assignment1_basics.pdf   # 作业手册（必读）
├── make_submission.sh     # 打包提交
└── pyproject.toml
```

## 学习提示

- 本作业的核心是**从零手写**语言模型的各个基础组件：BPE 分词器、Transformer 模型结构、优化器、训练循环等。
- 官方测试采用 **snapshot testing**：把参考输出存在 `tests/_snapshots/` 下（`.npz` 文件），你的实现输出要与之对齐，所以函数签名、张量形状、dtype 都要严格按手册要求来。
- 建议顺序：先通读 PDF 手册 → 打开 `tests/adapters.py` 看清接口 → 再逐个实现 `cs336_basics/` 下的模块 → 反复 `uv run pytest` 验证。

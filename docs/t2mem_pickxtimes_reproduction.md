# T²Mem：PickXtimes 单任务复现

本实现比较 A（自己训练的 π0.5 Stage 1）、B（从 A 的同一个 checkpoint 初始化的 Full T²Mem）、C（加载 B 的同一个 checkpoint，推理时关闭所有 writes）。训练和评估只允许 `PickXtimes`。实验标记为 **single-task reproduction: PickXtimes**，不能将论文 16-task 结果作为本实验保证；论文 Table 1 的 π0.5 / Stage 1 / T²Mem 参考值分别为 42.89% / 34.00% / 91.33%。

**运行状态：用户 A100 日志中，旧版全套为 80 passed / 1 failed；修订后的四种等价性用例均已通过（4 passed，58.86s）。** 原失败是 memory-free 与原生 Gemma 的数值一致性，最高精度诊断支持其与矩阵计算精度相关。修订将精度限定在该测试内、保留原容差，四种padding/remat组合及逐层对照已由用户服务器验证。其余80项此前已通过；这些结果来自两轮运行，尚无新版全套一次运行的记录。完整模型训练和 simulator rollout 仍为 **NOT RUN — requires Linux A100**。Mac 上只编辑和进行静态检查，没有安装 ML 依赖、下载数据或 checkpoint、执行前向/反向/训练/仿真，也没有 commit/push。以下安装、下载和 GPU 命令均供用户在 Linux 服务器手动执行。

## 1. 同步与环境安装

用户先在 Mac 检查 diff 并自行 commit/push；服务器进入仓库后执行：

```bash
git pull
export CUDA_VISIBLE_DEVICES=6
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
nvidia-smi
```

GPU 6 是物理编号；进程内唯一设备通常为逻辑 GPU 0。训练器检查单 JAX process、单可见 GPU，完全不启动 FSDP。JAX 与 simulator 使用两个 Python 环境，防止原仓库不同的 PyTorch/Transformers 版本互相覆盖。

安装采用用户目录和仓库内的虚拟环境，**不需要 sudo**。服务器需要已有可用的 NVIDIA 驱动，以及能访问的 git 和网络。进入仓库后执行：

```bash
bash scripts/t2mem/setup_server.sh
export PATH="$HOME/.local/bin:$PATH"
```

`setup_server.sh` 拒绝在 Mac 运行。若缺少 uv，使用官方 installer 将它安装到`$HOME/.local/bin`，不修改 shell profile；因此脚本运行后需要上述 PATH 命令。使用 curl 或 wget 下载 installer，无需先安装系统 Python。uv 选择 Python 3.11，缺少兼容解释器时按需下载到用户目录；已有集群 Python 可用`T2MEM_PYTHON=/path/to/python3.11 bash scripts/t2mem/setup_server.sh`指定。脚本初始化 simulator submodule，执行`uv sync --python 3.11 --group dev`和 editable install。参考[uv 安装](https://docs.astral.sh/uv/getting-started/installation/)及[Python 管理](https://docs.astral.sh/uv/guides/install-python/)。

训练环境采用仓库已有版本（JAX CUDA12 0.5.3、Flax 0.10.2、Orbax 0.11.13）。`jax[cuda12]`通过 Python 包安装 CUDA/cuDNN 用户态依赖；此路径无需安装系统 CUDA Toolkit，但不能安装或替代服务器的 NVIDIA kernel driver。已有 driver 必须兼容 CUDA12，实际是否可用以之后的`jax.devices()`检查为准。不要因为`nvidia-smi`显示的 CUDA 版本就重新安装 Toolkit；该显示也不证明 Vulkan 可用。参考[JAX GPU 安装](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)。这些依赖尚未在目标服务器验证。无需为本实现安装 FlashAttention。过期的`sandbox2/flash_attn_jax`workspace member已移除，因为该目录不在 checkout 中。

PickXtimes 数据下载使用 Hugging Face HTTP API，π0.5-base 使用仓库的 GCS downloader，这两个命令不要求 git-lfs。若其他 checkout/submodule 报缺少 git-lfs，可从已有集群 module 或用户环境提供；不要执行需要系统权限的安装。若某个依赖需要从源码编译，编译器也应从集群 module 或用户环境提供。

仓库的`.gitmodules`使用GitHub SSH URL，fresh server可能没有SSH key。setup先执行`git submodule init`，再把此checkout的`submodule.third_party/robomme_benchmark.url`设为公开HTTPS URL；不修改global Git配置或`.gitmodules`。

模拟器环境`.venv-robomme`安装 ManiSkill fork 的固定 commit`07be6fbc66350ddca200abfb0a11b692f078f7fd`，与当前 benchmark gitlink`856bc3a189d4172f3f47dbee4424d585f8d78db3`声明的来源一致。该 benchmark 要求`opencv-python>=4.11.0.86`，安装脚本使用4.11.0.86，修正原先4.10.0.84的冲突；benchmark还会安装其指定的PyTorch2.9.1/torchvision0.24.1，与policy环境分开。参考[固定 benchmark 的依赖声明](https://github.com/RoboMME/robomme_benchmark/blob/856bc3a189d4172f3f47dbee4424d585f8d78db3/pyproject.toml)。其CUDA wheel也依赖兼容的host driver，不能仅凭policy环境可用就认定simulator已可用。

另外安装moviepy、imageio、imageio-ffmpeg以及editable benchmark/client。常见Linux平台的imageio-ffmpeg wheel自带ffmpeg binary，因此保存视频无需apt安装ffmpeg；已有用户binary也可用`IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg`指定。`ffmpeg-python`本身不是ffmpeg binary。参考[imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)。默认evaluation不保存视频，但模型仍需要simulator渲染两路相机图像。

T²Mem 的 evaluation 在原入口`examples/robomme/eval.py --t2mem`提前分流，不导入 oracle、Gemini、Qwen 或 subgoal predictor。原有基线的完整 simulator/VLM 安装说明仍见`examples/robomme/readme.md`和其`requirements.txt`；运行旧外部 VLM baseline 时再安装那些额外依赖。本地 simulator submodule 为空，不能据此确认远端 benchmark 的全部安装及 Vulkan 条件。

无 sudo 的 Vulkan 排查：如果仅缺 loader/tools，且已有 conda，可在用户目录建独立环境：

```bash
conda create -y -p "$HOME/.local/share/t2mem/vulkan" -c conda-forge libvulkan-loader vulkan-tools
conda run -p "$HOME/.local/share/t2mem/vulkan" vulkaninfo --summary
```

也可使用已有 mamba/micromamba 的等价命令。两个包来自 conda-forge 的[libvulkan-loader](https://anaconda.org/conda-forge/libvulkan-loader)和[vulkan-tools](https://anaconda.org/conda-forge/vulkan-tools)。如果需要让虚拟环境中的 simulator 使用该 loader，仅在 simulator 终端附加用户 library path：

```bash
LD_LIBRARY_PATH="$HOME/.local/share/t2mem/vulkan/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  bash scripts/t2mem/eval_pickxtimes.sh --save-dir runs/evaluation/pickxtimes_A
```

Vulkan loader 不等于 NVIDIA Vulkan ICD/graphics driver。`vulkaninfo`应能看到 NVIDIA GPU；只看到 CPU/Mesa 或仍报找不到 ICD 时，需要服务器已有的匹配 NVIDIA 渲染库和 manifest。已有正确 manifest 可通过`VK_DRIVER_FILES`（modern loader）或`VK_ICD_FILENAMES`（legacy loader）指向其路径。缺少`libGLX_nvidia.so.0`/`libEGL_nvidia.so.0`或 kernel driver 时，安装 loader、Mesa 或 Python 包不能代替匹配驱动，应使用集群提供的 driver/module 或联系管理员补齐。参考[ManiSkill Vulkan 要求](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)和[Khronos driver 查找规则](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md)。此用户环境方案未在目标 A100 验证。

安装脚本在`runs/environment/`保存两个环境的`pip freeze`和policy/benchmark Git revisions；复现实验时应归档这些文件，记录固定ManiSkill commit及最终解析的Python包版本。

服务器安装后检查：

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -c 'import jax; print(jax.devices())'
```

应显示单个 CUDA GPU。安装阶段的失败、驱动兼容问题或 unavailable CUDA backend 均需要服务器日志修复，不能用静态检查替代。

## 2. 官方 checkpoint 和 PickXtimes 数据

```bash
export CUDA_VISIBLE_DEVICES=6
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
uv run python scripts/download_pi05_base.py
uv run python scripts/t2mem/download_pickxtimes.py --output-dir data/robomme_pickxtimes_h5
uv run python scripts/tarxz_h5.py decompress --input_dir data/robomme_pickxtimes_h5 --jobs 1
bash scripts/t2mem/prepare_data.sh --official-train-root data/robomme_pickxtimes_h5 --output-dir data/t2mem
```

下载脚本只匹配官方 [Yinpei/robomme_data_h5](https://huggingface.co/datasets/Yinpei/robomme_data_h5/tree/main) 的 `*PickXtimes*` 文件。当前官方 archive 为`record_dataset_PickXtimes.h5.tar.xz`，数据准备接受解压后的`record_dataset_PickXtimes.h5`，并兼容历史名称`data_PickXtimes.h5`。同一root同时存在两个名称时拒绝歧义，避免混用版本。若release目录布局变化，显式将`--official-train-root`指向包含该文件的目录。不要把benchmark test recordings、生成的test episodes或validation文件放到此目录。

输出：

```text
data/t2mem/manifest.json
data/t2mem/assets/robomme/norm_stats.json
```

准备流程只打开明确命名的PickXtimes HDF5，使用其`episode_N/timestep_N`的真实连续时间线和`info/is_video_demo`分段，不依赖旧pickle中不存在的task-name字段。language instruction兼容官方HDF5的variable-length UTF-8字符串。不读取subgoal、抓取计数或success/task-progress来形成memory。显式held-out文件路径及HDF5 split metadata被拒绝；文件名本身不能证明数据来源，因此manifest记录操作者提供的官方training-release来源声明。

π0.5-base从GCS下载，`HF_ENDPOINT`只影响HF数据下载。GCS后台传输异常现在会向调用者抛出，失败的`.partial`文件不会升级成已完成cache；不要把传输失败当作checkpoint下载成功。本轮新增的文件名兼容及下载失败/retry回归测试尚未在服务器执行，可单独运行：

```bash
PYTHONPATH=src uv run pytest -q tests/test_t2mem_training.py tests/test_t2mem_download.py
```

没有重复存储每个 sample 的历史图像，也不下载单独的 vision-encoder checkpoint。训练期间按 episode 懒加载原始 front/wrist RGB 和 8 维 robot state。旧数据 builder 只保留 execution pickle，且 action horizon=20/end padding 无 mask，不能用于本实现的完整时序 H64 训练。

normalization 仅由该任务 training execution frames 计算：前 7 维关节动作减去当前关节状态，最后 1 维 gripper 保持原有约定；OpenPI quantile normalization 后 pad 到 32 维。未来 action chunk 保留 64 帧，越过 episode 末尾、缺失动作或 padded dimensions 的 mask=False，demo 帧没有 action loss。A→B 初始化检查保存与当前 normalization JSON 的 SHA256 一致。已有 manifest 不会被静默覆盖；需要重新准备时使用一个新的 output directory，并同步修改 YAML 的两个 data 路径。

`model.robot_action_dim=8`定义真实动作通道。A/B/C在sampling时的initial noise及每次Euler flow update都把其余24个padded通道保持为0，与训练输入的masked domain一致；它们不参与action loss，也不传给simulator。

## 3. 配置与算法对应

| 配置 | 用途 |
|---|---|
| `configs/t2mem/t2mem_paper.yaml` | 论文 Table 2/3 的 memory 参数、H64、stride16、完整梯度路径；缩减为 PickXtimes |
| `configs/t2mem/t2mem_a100_80gb.yaml` | 单卡、BF16 backbone、FP32 slow master/fast weights、rematerialization、gradient accumulation |
| `configs/t2mem/t2mem_smoke.yaml` | 一次 Stage 1 更新和一个 A/B cycle；**显式 approximation：保留序列末 8 个 observations，丢弃更早 history** |
| `configs/t2mem/t2mem_ablation.yaml` | no-write 的配置示意；C 不使用它重新训练，而使用 B checkpoint + `--ablation no-online-write` |

论文指定的部分：16 interface tokens，width1024；18 allocated / 17 effective memory layers；独立 Q/K/V；16 个 nonlinear fast-weight heads；64→256→64 exact GeLU MLP，含 biases；learned W0；FP32 fast state；Q/K/V RMS normalization、Q/K interleaved RoPE；channel-wise tanh gate 初值0.01；read-before-write。Eq. (5)–(8) 建立 observation-grounded read/write 和融合，Eq. (10)–(15) 实现 residual/displacement/alignment adaptive beta、JVP curvature calibration、learned positive multiplier。只有 beta 和 calibration stop-gradient，完整 fast-state recurrence 和 inner gradient 默认保留二阶梯度。beta=1 消融仍保留 curvature calibration。

Eq. (16) 的 Flow Matching objective 使用原有 OpenPI action/time embedding 与 AE，每个 chunk 按有效 action 元素数量归一化。按 episode 从 W0 重置，时间递归通过 `jax.lax.scan` 和 rematerialization 组织；late action loss 可回传到 earlier writes。冻结 slow 参数仅限制 optimizer 参数集合，Stage 2A 的 AE 仍将梯度传给 memory，Stage 2B 的 fast reads/writes 仍运行。

`model.temporal_remat_block_size=16`将时间维组织为嵌套scan：外层保存block边界fast states，backward重新计算完整block内的写入轨迹；每帧也进行activation rematerialization。梯度穿过所有block边界，没有detach或截短BPTT。这是保留算法语义的工程调整。可更改block size来平衡重计算和保存边界states；短序列自动缩小到不超过长度的可整除block size，不增加虚拟observations。`remat=false`仍走完整可微scan，只关闭重计算。

工程调整与未公开细节：

- 单任务替代论文的 16-task 训练；batch accumulation 替代 32×H200；没有验证其训练时间或 A100 OOM 边界。
- 复用 OpenPI 原生 AE 8-head attention 的扁平 rank-16 Q/KV/O interface adapters，总 adapter 参数为 2,211,840，论文列出的约 1.622M 的具体布局没有公开。不能声称参数树完全相同。
- Interface 的 base attention/FFN 复用 AE 权重，memory norm 和 adapters 独立；这是与现有 OpenPI 权重兼容的实现选择。
- 按当前结构静态计算，新增slow参数共68,525,074：read/write projections56,678,400，learnedW0 9,529,344，adapters2,211,840，interface norms36,864，interface tokens16,384，gates18,432，rho18及state projection33,792。W0的FP32 episode state仍为约36.35MiB。这些是结构计算，尚非服务器实际参数树测量。
- 所有 A/B/C 使用同一 input transform，生成 instruction-only 和 native π0.5 state+instruction 两种 tokenization。memory-free 路径使用 native π0.5 prompt；T²Mem 路径使用 VL-only prompt，并把 continuous proprioception 置于 suffix，确保 memory interface 不能访问 proprioception。normalization、相机、action convention、H64/E32、输入字典完全一致；论文没有公开该状态路由/adapter 的逐参数实现。
- `max_token_len=200`、AdamW beta/epsilon、outer gradient clipping、常量 LR、示范时间线 offset=0 和每个 execution frame 都监督是兼容仓库的显式选择。Table 3 没有给出总 alternating cycles，因此正式训练必须显式指定 `--cycles`。
- 默认 full episode prefix、`action_supervision_stride=1`、`detach_fast_state=false`。`max_sequence_length` 只有配合 `truncate_prefix=true` 才能开启；稀疏 supervision、截掉 prefix、detach fast state 都在 config、日志及 checkpoint 标记为 approximation。原论文 H64/E32、memory 容量不会被 A100 配置静默缩小。

## 4. 服务器 tests 和训练 smoke

在服务器运行以下测试。2026-10-12 用户提供的旧版全套日志为`80 passed, 1 failed`；随后最高精度单项诊断为`1 passed`，同步修订后padding/remat四种组合为`4 passed`（58.86s）。修订测试自行限定最高矩阵精度，不需要为正式训练全局设置该变量。其余80项此前已通过；以下命令可获取新版全套的一次运行记录：

```bash
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=src uv run pytest -q tests/test_t2mem*.py
```

测试涵盖 memory math/shape、reset/no leakage/no-write、界面 attention mask、earlier-write meta-gradient、stage ownership、PickXtimes-only filtering、chronological loader/padded-action masks、完整 checkpoint/optimizer/RNG/sampler resume、stateful write cadence。小型测试不构造 3B backbone；它们不能替代实际的 π0.5 compilation 与 training smoke。

分别运行 A 和 B 的 smoke：

```bash
CUDA_VISIBLE_DEVICES=6 uv run python scripts/t2mem/train.py --config configs/t2mem/t2mem_smoke.yaml --stage stage1 --checkpoint-dir runs/ckpts/t2mem_smoke_stage1
CUDA_VISIBLE_DEVICES=6 uv run python scripts/t2mem/train.py --config configs/t2mem/t2mem_smoke.yaml --stage alternating --init-checkpoint runs/ckpts/t2mem_smoke_stage1/1 --checkpoint-dir runs/ckpts/t2mem_smoke_stage2 --cycles 1
```

或运行 `bash scripts/t2mem/smoke_test.sh`，按顺序执行 tests 和上述两个训练。已有 run directory 非空时拒绝覆盖，请换路径或使用 resume。smoke 使用完整网络和真实官方 samples，但短 prefix 与小 batch 是 approximation，不作为 A/B/C 正式性能报告。

## 5. Experiment A：完整 Stage 1

```bash
bash scripts/t2mem/train_stage1.sh --checkpoint-dir runs/ckpts/pickxtimes_stage1
```

默认20,000 optimizer updates，LR2e-5，microbatch1×accumulation64=effective batch64 frames。base VLM 和 vision encoder frozen，VLM LoRA、完整 AE、action/time projections trainable；不运行 memory，新增 continuous state_proj 也 frozen。Stage 1 以单 execution frame 和带 mask 的 H64 future actions 训练。

保存 A checkpoint `runs/ckpts/pickxtimes_stage1/20000`，并用它初始化 B 和评估 A。不要用官方已发布的 RoboMME baseline替代自己训练的 A。若额外评估官方 baseline，请单独列为外部参考，其训练任务/预算与本实验不同。

## 6. Experiment B：完整 alternating Stage 2

```bash
bash scripts/t2mem/train_stage2.sh --init-checkpoint runs/ckpts/pickxtimes_stage1/20000 --checkpoint-dir runs/ckpts/pickxtimes_t2mem --cycles 4
```

`4` 是操作者选择的运行预算示例，不能声称论文给出了4 cycles；可调整。每个 cycle 的 Stage 2A / 2B 各500 updates，总计此例4000 updates；microbatch1×accumulation32=effective batch32 sequences。memory LR7.5e-5、gate LR5e-4、policy LR3e-5。

初始化要求 A 已完成其配置预算；B 从 A 的同一个 checkpoint 载入 policy，memory 使用 fresh learned-W0 initialization。Stage 2A 仅训练 interface/tokens/adapters、read/write projections、W0、rho 和 gates；Stage 2B 仅训练 VLM LoRA、AE、action/time/proprio projections。两个 optimizer states 独立持久化，不因 phase 切换清零。每个 effective batch 仅 PickXtimes，每个 sequence 独立 W0、严格按真实 observations 排序。PickXtimes 不需要 video demo；如果 data 中存在 demo prefix，仍遵守有 writes、无动作监督的通用规则。

入口也支持单独 phase，便于服务器诊断：

```bash
CUDA_VISIBLE_DEVICES=6 uv run python scripts/t2mem/train.py --stage stage2a --init-checkpoint runs/ckpts/pickxtimes_stage1/20000 --checkpoint-dir runs/ckpts/pickxtimes_phase2a --updates 500
CUDA_VISIBLE_DEVICES=6 uv run python scripts/t2mem/train.py --stage stage2b --init-checkpoint runs/ckpts/pickxtimes_phase2a/500 --checkpoint-dir runs/ckpts/pickxtimes_phase2b --updates 500
```

Stage 2B 不能直接从 Stage 1 启动。正式 B 的 provenance 由 alternating run记录，比较脚本检查 B 初始化来源确实为评估 A 的 checkpoint。

## 7. Checkpoint、resume 和日志

```bash
CUDA_VISIBLE_DEVICES=6 uv run python scripts/t2mem/train.py --stage resume --resume runs/ckpts/pickxtimes_t2mem
```

`--resume` 既可以指定完整 step directory，也可以指定 run root（取最后一个 metadata 已完整提交的 checkpoint）。恢复所有模型参数、阶段、cycle、phase/global step、每阶段 optimizer/scheduler counts、JAX RNG 和 host episode sampler RNG/counters。默认采用保存的 resolved config。A/B 是独立 run，因此 B 的 global step 从0起，A的来源/20,000步保留在 source metadata。中断时只恢复最后完整 snapshot，不声称恢复尚未保存的 microbatch。

恢复时使用该snapshot的`assets/robomme/norm_stats.json`，避免原始data-preparation目录被移动或改写后导致模型与动作normalization不一致。

如明确增加alternating总预算，可在resume命令加`--cycles 6`。已完成4cycles的run将从第5cycle的Stage2A继续，保持optimizer/RNG/source及step计数；不能把预算缩小到已完成或正在进行的cycle以下。未传`--cycles`时严格恢复原有预算。

```text
runs/ckpts/<run>/resolved_config.json
runs/ckpts/<run>/metrics.jsonl
runs/ckpts/<run>/<global_step>/params/             # OpenPI compatible Orbax {'params': ...}
runs/ckpts/<run>/<global_step>/train_state/        # separate per-stage optimizers + RNG
runs/ckpts/<run>/<global_step>/metadata.json       # config/provenance/phase/sampler/approximation
runs/ckpts/<run>/<global_step>/assets/robomme/norm_stats.json
```

每条训练日志记录loss、gradient norm、effective batch、task sampling counts、sequence length mean/max、valid supervision frames、scheduled writes、approximation标签。参数 ownership 及各组 scalar count 在启动时输出。写 checkpoint 前同步 device→CPU；不会在下一次训练同时保留异步 GPU snapshot，但需要容纳模型/optimizer snapshot 的 CPU RAM/磁盘。所有 checkpoint 当前保留，完整 runs 的磁盘开销较大，可在确认 final checkpoint 后手动归档旧 snapshots。

## 8. 单 A100 的 policy/simulator 资源

同一时刻只启动一个 A/B/C policy server 和一个 simulator client。停止训练后再评估，不并行三个模型。

`serve.sh` 默认 GPU6、JAX `PREALLOCATE=false`、memory fraction0.70；training wrapper默认0.90。0.70是留给 simulator/headroom 的工程起点，**不是测量结果或保证的 GPU 上限**。尤其关闭 preallocation 后，这个 fraction 不能代替严格显存隔离。JAX的模型/attention cache 与 ManiSkill的render/physics footprint仍需实际监测。

simulator 的 `gui_render=False` 只关闭GUI，不能等同于CPU simulation。默认`--sim-mode gpu`使用同一物理GPU6/逻辑GPU0；可尝试`--sim-mode cpu-render`，设置仓库README推荐的`SAPIEN_RENDER_DEVICE=cpu`及`MUJOCO_GL=osmesa`。本地缺失simulator源码，CPU/Vulkan支持尚未确认；不隐藏CUDA来制造成功，unsupported renderer应直接报错。服务器需要检查render输出、吞吐与`nvidia-smi`。

## 9. A/B/C evaluation：相同50 episodes × 3 seeds

每个模型在独立结果目录保存完整 per-episode records、difficulty、seed、checkpoint/provenance/config。三个模型均使用同一个官方test split、seeds7/17/27、H64/E32、stride16。每模型150 rollouts，50 episodes的难度数应为26 Easy/12 Medium/12 Hard；汇总器明确区分基础设施错误与有效rollout失败。

先运行低成本smoke，两个终端：

```bash
# 终端1：GPU policy环境，保持运行
bash scripts/t2mem/serve.sh runs/ckpts/t2mem_smoke_stage2/2 --ablation full --port 8000
# 终端2：simulator环境，仅两个episodes，一个seed
CUDA_VISIBLE_DEVICES=6 .venv-robomme/bin/python examples/robomme/eval.py --t2mem --tasks PickXtimes --episodes-per-task 2 --seeds 7 --port 8000 --save-dir runs/evaluation/pickxtimes_smoke
```

正式A，先关闭其他server：

```bash
# 终端1
bash scripts/t2mem/serve.sh runs/ckpts/pickxtimes_stage1/20000 --ablation memory-free --port 8000
# 终端2
CUDA_VISIBLE_DEVICES=6 .venv-robomme/bin/python examples/robomme/eval.py --t2mem --tasks PickXtimes --episodes-per-task 50 --seeds 7,17,27 --port 8000 --save-dir runs/evaluation/pickxtimes_A
```

正式B，关闭A server后：

```bash
# 终端1：本例4 cycles的step4000
bash scripts/t2mem/serve.sh runs/ckpts/pickxtimes_t2mem/4000 --ablation full --port 8000
# 终端2
CUDA_VISIBLE_DEVICES=6 .venv-robomme/bin/python examples/robomme/eval.py --t2mem --tasks PickXtimes --episodes-per-task 50 --seeds 7,17,27 --port 8000 --save-dir runs/evaluation/pickxtimes_B
```

正式C，关闭B server后，**加载B的同一个4000 checkpoint，不重新训练**：

```bash
# 终端1
bash scripts/t2mem/serve.sh runs/ckpts/pickxtimes_t2mem/4000 --ablation no-online-write --port 8000
# 终端2
CUDA_VISIBLE_DEVICES=6 .venv-robomme/bin/python examples/robomme/eval.py --t2mem --tasks PickXtimes --episodes-per-task 50 --seeds 7,17,27 --port 8000 --save-dir runs/evaluation/pickxtimes_C
```

No-write保留learned W0、reads、全部slow参数及正常sampling RNG cadence，禁止demo/execution所有writes。服务器每个connection维护独立episode state；start/reset使用W0，end/close销毁。每次replan在固定pre-write state上完成所有flow denoising，然后最多commit一次当前observation write；执行32动作，第16步发送memory-only observation，第32步重新规划。环境step与diffusionstep完全分离，不发送全部history，不重复写相同observation。

汇总：

```bash
.venv-robomme/bin/python scripts/t2mem/compare_results.py --baseline runs/evaluation/pickxtimes_A --full runs/evaluation/pickxtimes_B --no-write runs/evaluation/pickxtimes_C --output-dir runs/evaluation/pickxtimes_comparison
```

比较表包含π0.5 Fine-tuned、Full T²Mem、T²Mem No-Write的Easy/Medium/Hard/Overall。比较脚本校验相同episode/seed集合、A来源/B初始化和B/C相同checkpoint；不将中断run伪报成完整150-rollout结果。`episodes.json`/`manifest.json`/`metrics.json`保留原始证据。再用相同命令和save-dir可恢复未完成的evaluation。

上述evaluation命令也可使用`bash scripts/t2mem/eval_pickxtimes.sh --port 8000 --save-dir <对应结果目录>`。wrapper默认采用setup创建的`.venv-robomme/bin/python`，设置`ROBOMME_PYTHON`可选择其他simulator环境。汇总时还校验实际normalization SHA256、token/action维度、已完成的training budget及B/C的alternating来源；只有50×3、每seed的26/12/12难度数及stride16全部匹配才标记为论文评估协议。

## 10. OOM与最可能的运行时问题

完整长episode的二阶meta-gradient计算图，即使remat、BF16、microbatch1也可能超过80GB；本实现不能在未实测时声称已解决。先检查microbatch=1、num_devices=1、唯一GPU6、remat=true以及simulator/其他server已停止。减少accumulation次数不会减少单sequence的peak memory，只会改变effective batch。各长度按下一2的幂padding/bucketing，首次遇到新bucket会JIT compile；长episode和每帧监督可能引起compile耗时与cache增长。

先尝试调整`model.temporal_remat_block_size`，它保留跨时间完整梯度，不构成approximation。增加block size可能减少边界state存储，同时增加block重计算开销；真实最优值与OOM边界仍须A100测量。

需要调试时复制smoke YAML，显式设置`max_sequence_length`和`truncate_prefix=true`；它从截断窗口的W0开始，改变长程历史，必须继续标记approximation。增加write_stride会改变写入算法；sparse监督改变objective；`memory.detach_fast_state=true`切断跨时间meta-gradient，只适合显式近似诊断，不能称为论文完整训练。正式结果应使用full-prefix配置并记录最终config。

优先反馈以下服务器日志：首次模型init/compile时间、真实tree shape与missing checkpoint path、loss/gradient finite状态、Stage2A/2B ownership counts、sequence/write counts、peak GPU/CPU memory、Vulkan renderer错误、WebSocket异常以及失败episode记录。高风险点是pinned NNX bridge/Orbax API兼容、scan中的高阶AD/编译规模、OpenPI checkpoint树路径差异、GPU共享、simulator安装/渲染。失败应修复后复测，不能把静态通过描述为训练通过。

## 11. 实际静态检查

系统`python3`为3.9.6，完整`python3 -m compileall -q src scripts examples`在已有的Python3.10+ `match`语法文件失败；仓库`pyproject.toml`要求Python3.11–3.12。预装的`python3.11`进行完整compileall通过，没有安装Python或任何ML库：

```bash
python3.11 -m compileall -q src scripts examples packages/openpi-client/src tests
git diff --check
for script in scripts/t2mem/*.sh; do bash -n "$script"; done
```

新增文件也进行了AST/语法和接口静态审查。本地没有执行JAX tests。用户服务器旧版全套80项通过、1项数值一致性失败；修订后的四种等价性用例随后全部通过（4 passed，58.86s）。3B模型checkpoint加载、实际GPUtraining/inference、simulatorrollout和success rates仍为**NOT RUN — requires Linux A100**，等待用户按上面的服务器命令验证。

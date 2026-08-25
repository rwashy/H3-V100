# H3_V100

简体中文 | [English](README.md)

`H3_V100` 是面向 NVIDIA Tesla V100（SM70）的 MiniMax H3 ComfyUI
优化节点。v1.4.0 将已经完成实机验证的精度、显存和权重策略固定为稳定配置，
并精简工作流接口。

节点只修改流经它的克隆 `MODEL`，不会改写 ComfyUI 或其他自定义节点文件。
它会严格识别经过验证的 50-block MiniMax H3 结构，其他模型会被拒绝。

## 稳定版内置策略

- ComfyUI Dynamic VBAR 独占压缩权重驻留；不建立第二套 INT8 GPU staging。
- H3 进入采样前卸载已经失活的前阶段 CUDA Dynamic 模型，释放文本编码阶段
  占用，同时保留当前采样所需模型。
- 主残差流、文本前路径和音频 query 安全路径保持 FP32；QKV、主 Attention
  和大型 MLP linear 使用经过验证的 FP16 计算岛。
- Attention 固定使用 `/16` 预缩放，并在无 bias 的 output projection 后以
  FP32 恢复。
- scaled FP16 SwiGLU 固定开启：value branch `/16`、fc2 输入 `/8`，最终以
  FP32 恢复组合比例。
- MLP 根据真实 driver-free VRAM 自适应分块。每次 MLP 内只展开一次 fc1/fc2
  权重并供全部 activation chunks 复用，调用结束立即释放。
- 超长序列启用有界 QKV、Q/K Norm+RoPE 和 output projection 分块。

## 节点参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `mixed_precision` | 开 | 启用已验证的 V100 FP16/FP32 精度拆分。 |
| `attention_backend` | `flash_attn` | 选择精确 Flash 或显式 `sol_attn`。 |
| `sol_tau` | `1.0` | Sol 稀疏阈值，仅在 Sol 模式显示。 |
| `sol_start_percent` | `0.2` | Sol 调度窗口起点。 |
| `sol_end_percent` | `0.8` | Sol 调度窗口终点。 |

## 启动参数迁移（重要）

当前验证过的 V100 16 GiB 配置使用 ComfyUI 默认 DynamicVRAM：

```text
不要添加 --disable-dynamic-vram
不要添加 --lowvram
不要添加 --fast fp16_accumulation
```

如果启动命令仍含 `--disable-dynamic-vram` 或 `--lowvram`，请将它们删除，
完整重启 ComfyUI，并重新加载扩散模型。节点会检查传入的模型是否为 Dynamic
ModelPatcher；不是则直接报错，不会静默退回未经验证的路径。

## 安装与升级

1. 停止 ComfyUI。
2. 用完整的 v1.4.0 `H3_V100` 文件夹替换旧版本；不要只覆盖单个 `.py`。
3. 确认目录内包含
   `comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd`。
4. 按上节清理启动参数并重启 ComfyUI。

不要与其他会修改同一 H3 Attention、MLP、QKV 或 output projection 的节点
叠加。TE-Speed 如需使用，应放在 H3 V100 之前。

## 运行环境

当前 v1.4.0 验证环境是**单张 V100**，没有将模型或 VAE 分配到第二张显卡：

| 组件 | 当前验证配置 |
|---|---|
| GPU | 1 × NVIDIA Tesla V100 16 GiB（SM70），所有 CUDA 工作均在 `cuda:0` |
| 模型布置 | 使用 ComfyUI 默认 DynamicVRAM 自动管理，没有手动分卡 |
| H3 扩散模型 | MiniMax H3 INT8/混合精度权重，Dynamic VBAR 按需驻留于 `cuda:0` |
| 文本编码器 | MiniMax H3 文本编码器在 CPU 完成编码；进入 H3 阶段后释放失活的 Dynamic 模型 |
| 视频/音频 VAE | 由 ComfyUI 在同一张 `cuda:0` 与 CPU offload 之间管理 |
| LoRA | MiniMax H3 FL2V Turbo 8-step BF16 |
| 采样 | 8 步，24,792 packed tokens |
| 软件 | Windows x64、CPython 3.12、PyTorch 2.8.0、CUDA 12.8 |

运行环境需要包含当前 MiniMax H3 实现和 DynamicVRAM 的 ComfyUI。

不需要额外 pip 包，也不要为安装本节点替换便携版中已经正常工作的 Torch。
随包 `.pyd` 与 Python、Torch、CUDA ABI 相关；其他 Python 版本和 Linux 不在
当前预编译包支持范围内。

## v1.4.0 验证结果

同一 8-step、24,792-token 工作流的本机结果：

| 路线 | 状态 | 首轮 | 显示平均 | 完整任务 |
|---|---|---:|---:|---:|
| Flash | 冷启动 | 63.85 秒 | 54.70 秒/轮 | 563.77 秒 |
| Flash | 热启动、稳定配置 | 54.66 秒 | 55.41 秒/轮 | 490.84 秒 |
| Sol | 热启动、稳定配置 | 54.7 秒 | 约 54 秒/轮 | 485.69 秒 |

三次均完成视频和音频输出，未观察到画面或声音异常。不同随机数、温度和频率
状态会影响秒数。
这些数据用于确认运行稳定性和冷/热启动差距已明显缩小，不构成跨设备性能保证。

## 精确与近似路线

`flash_attn` 是精确 Attention 路线。`sol_attn` 是显式选择的稀疏近似路线，
只在满足序列长度、Block 位置和扩散阶段条件时运行；Flash 模式不会隐式进入
Sol。对质量敏感或建立基线时优先使用 Flash。

## 已知边界

分辨率、时长、参考输入、音频、其他常驻模型、CUDA 碎片和 ComfyUI 版本都会
改变显存边界。建议从 864×480 或 960×544、5–10 秒开始，再逐项增加。
极长序列会通过更小的 MLP/QKV 分块换取可运行性，速度可能显著下降。

![MiniMax H3 分辨率与时长运行区域](assets/h3-resolution-duration-vram-regions.png)

| 档位 | 分辨率与时长示例 | 建议 |
|---|---|---|
| 推荐档 | 0.2–0.5 MP、5–10 秒；0.2–0.3 MP 可延长至 15 秒 | 清晰度、时长和生成时间较均衡。 |
| 扩展档 | 0.4–0.6 MP、10–15 秒；0.7–0.9 MP、5–8 秒 | 通常会增加分块，耗时明显上升。 |
| 极端档 | 约 0.9 MP、15 秒，或大多数超过 1.0 MP 的组合 | 建议先用较短时长确认显存和耗时。 |
| 不推荐 | 1.5–2.0 MP 配合较长时长 | OOM 风险和生成时间可能不成比例。 |

## 参考项目与致谢

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)：模型和执行框架。
- [Sol-Attn 项目页](https://nvlabs.github.io/Sana/Sol-Attn/)与
  [论文](https://arxiv.org/abs/2607.24027)：训练无关的在线块稀疏
  Attention 方法参考。本节点针对 H3 与 V100/SM70 实现独立适配；Sol 是
  显式近似路线，其他模型和硬件上的结果不代表本节点性能。
- [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch)：
  H3 V100 混合精度拆分方案来源。本项目将修改源文件的方案改造成工作流范围
  节点，并增加音频安全和自适应内存保护。
- [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100)：
  SM70 Flash Attention 实现来源。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) 4.2.0：原生内核构建所用
  CUTLASS/CuTe 头文件。
- PyTorch SDPA：正确性、性能比较和安全回退参考。

完整署名和许可证说明见 `NOTICE.md` 和 `licenses/`。本项目整体按
GPL-3.0-only 分发，BSD-3-Clause 组件保留原声明。

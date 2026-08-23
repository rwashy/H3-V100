# H3_V100

简体中文 | [English](README.md)

`H3_V100` 是用于 NVIDIA Tesla V100（SM70）运行 MiniMax H3 的 ComfyUI
自定义节点。它提供：

1. 经过实测的混合精度，以及精确 Flash/显式 Sol Attention 路线；
2. 面向长序列的自适应显存保护（显存分块）。

节点不会修改 ComfyUI 或其他节点文件，只对流经节点的克隆 `MODEL` 应用
补丁，并严格检查预期的 50-block MiniMax H3 结构。

## 核心功能

节点公开一个精度开关、一个 Attention 后端选择和三项 Sol 参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `mixed_precision` | 开 | FP16 QKV、主要 Attention 和已验证的 FP16 MLP linear；Q/K Norm、RoPE、SwiGLU、Attention output projection、残差及音频 query 重算保持 FP32。 |
| `attention_backend` | `flash_attn` | 选择精确 Flash 或显式 `sol_attn`。 |
| `sol_tau` | `1.0` | Sol 稀疏阈值。 |
| `sol_start_percent` | `0.2` | Sol 扩散调度起点。 |
| `sol_end_percent` | `0.8` | Sol 扩散调度终点。 |

Sol 参数在 Flash 模式下自动隐藏。自适应内存始终开启，独立于公开选项。
关闭混合精度后，MLP/QKV/output
projection分块、缓存压力清理和动态权重预取保护仍然生效。若要完全绕过本
节点，请在工作流中旁路或删除它。

`flash_attn` 是精确 Attention 路线；`sol_attn` 是用户显式选择的稀疏近似
路线，只在满足序列长度、Block 位置和扩散阶段条件时运行。Flash 模式不会
在后台进入 Sol。

## 推荐分辨率与时长档位

下图将 0.2-2.0 MP 的全部分辨率预设与 5-15 秒逐项配对。格内数字为估算的
packed sequence 长度（千 tokens）；颜色表示“不进行 MLP 分块”大致需要的
物理显存区域，而不是总显存需求或成功保证。实时显存压力可能更早触发分块。

![MiniMax H3 分辨率与时长运行区域](assets/h3-resolution-duration-vram-regions.png)

| 档位 | 分辨率与时长示例 | 建议 |
|---|---|---|
| 推荐档 | 0.2-0.5 MP、5-10 秒；0.2-0.3 MP 可延长至 15 秒 | 清晰度、时长和生成时间最均衡。 |
| 扩展档 | 0.4-0.6 MP、10-15 秒；0.7-0.9 MP、5-8 秒 | 通常开始明显分块，耗时大幅增加。 |
| 极端档 | 约 0.9 MP、15 秒，或大多数超过 1.0 MP 的组合 | 大量 4K/8K 分块，应先用短片验证。 |
| 不推荐 | 1.5-2.0 MP 配合较长时长 | 仅作为理论矩阵范围，耗时和 OOM 风险不成比例。 |

16 GiB V100 实测参考：约 32K tokens 以内可能不分块。32K 以上开始分块，以防止
显存溢出，耗时可能急剧增加。建议从 864x480或 960x544、5-10秒开始。
1280x736、15秒等极端组合仍属于实验范围。


## 运行环境与依赖

- NVIDIA Tesla V100，计算能力 7.0（SM70）。
- 随包原生扩展支持 Windows x64、CPython 3.12。
- 与 ComfyUI 兼容的 CUDA 版 PyTorch；开发验证环境为 PyTorch 2.8.0、
  CUDA 12.8。
- 包含本节点所识别 MiniMax H3 实现的 ComfyUI 版本。
- 与 PyTorch CUDA 运行时兼容的 NVIDIA 驱动。

不需要额外 pip 依赖。PyTorch 由 ComfyUI 提供，不要为了安装本节点而替换
便携版中已经正常工作的 Torch。当前预编译版本不支持其他 Python 版本、Linux
或不同 Torch/CUDA ABI 环境。

## 公开基准测试配置

本仓库中的性能与边界数据采用以下可复现配置。

| 组件 | 基准配置 |
|---|---|
| GPU | 2 × NVIDIA Tesla V100-SXM2 16 GiB |
| 系统内存 | 128 GiB RAM |
| H3 扩散模型 | MiniMax H3 Ref2VA INT8 ConvRot，放置于第二张 V100 |
| 文本编码器 | Qwen3-VL 32B MiniMax H3 INT8 ConvRot，放置于 CPU |
| 视频 VAE | MiniMax H3 Video VAE FP16，放置于第一张 V100 |
| 音频 VAE | MiniMax H3 Audio VAE FP32，放置于第一张 V100 |
| 加速 LoRA | MiniMax H3 FL2V Turbo 8-step BF16，强度 0.75 |
| 采样 | 采样器运行 8 步，不跳步 |
| 全局 FP16 accumulation | 关闭 |

注意：文档中的耗时属于**双 V100 工作流结果**，不代表单卡吞吐量。

### 已发布 v1.1.2 实测性能

在固定种子、864×480、5 秒、18,376 tokens、采样器运行 8 步且不跳步的热启动
对比中，不使用本节点的
原始路径耗时 **738 秒**。开启混合精度后，耗时降至 **380.11 秒**，吞吐速度提升
**94.2%**（1.94 倍），耗时减少 **48.5%**；在混合精度基础上继续开启 Flash
Attention，又减少 **39.94 秒**，最终耗时为 **340.17 秒**。相对仅开启混合精度，
吞吐速度再提升 **11.7%**，耗时再减少 **10.5%**。总体而言，相较不使用节点的
原始路径，吞吐速度提升 **117.0%**（2.17 倍），总耗时减少 **53.9%**。

测试过程中 GPU 曾出现过热降频，因此上述耗时和提升幅度属于偏保守的实测结果；
在散热充分、核心频率稳定的条件下，实际优化效率可能更高。

以上数据保留自已经发布的 v1.1.2 基线，不作为 v1.3.0 的性能结论。新版
Flash/Sol、17K/25K、冷启动/热启动结果将在候选版测试矩阵完成后补充。

### v1.3.0 候选版单 V100 测试

本次热启动测试与上面的双 V100 已发布基准不是同一环境，必须分开理解：

| 项目 | 配置 |
|---|---|
| 工作流 | MiniMax H3 文生视频 |
| 输出 | 608x352、15秒 |
| 采样 | 8步 |
| GPU | 单张 V100 16 GiB；未手动分配组件 |
| 加载节点 | 默认 CLIP、扩散模型和 VAE 加载节点；使用 CPU 与 GPU 0 |
| 启动参数 | `--lowvram --disable-dynamic-vram` |
| 启动状态 | 热启动；暂不统计冷启动模型加载优化 |
| 温度状态 | GPU 因散热问题存在降频 |

| 路线 | 最终采样平均 | 估算8步采样时间 | 完整任务时间 |
|---|---:|---:|---:|
| 不使用 H3 V100 节点 | 214.71 秒/步 | 1,717.68 秒 | 1,774.00 秒（29:34） |
| H3 V100 `flash_attn` | 62.21 秒/步 | 497.68 秒 | 550.68 秒 |
| H3 V100 `sol_attn` | 62.51 秒/步 | 500.08 秒 | 552.33 秒 |

在本次记录中，Flash 相对“不使用节点”的兼容性参考将采样耗时降低71.0%，
对应3.45倍采样吞吐；Sol 将采样耗时降低70.9%，对应3.43倍采样吞吐。
Sol 的采样时间比 Flash 高约0.5%，完整任务时间高约0.3%。该差异很小且只对应
当前工作负载，不能据此宣称某一后端普遍更快。

“不使用节点”并不是 ComfyUI 无限制原生路线的最高速度：为了让三次测试使用
相同进程配置并保证 V100 16GB 可运行，测试保留了两个必需启动参数，而它们会
关闭 ComfyUI 的部分动态显存优化。因此该行只能作为“相同启动参数下”的兼容性
参考，不能当作完全公平的原生性能上限。本组每条路线仅记录一次且 GPU 存在
热降频，仍需重复热启动测试。三次非采样阶段分别约为56.32、53.00和52.25秒，
彼此接近，说明当前主要差异确实发生在去噪采样阶段，而不是 VAE 解码或其他
工作流开销。

## 安装和使用

1. 将 `H3_V100` 文件夹复制到 `ComfyUI/custom_nodes/`。
2. 需要 Flash Attention 时，确认目录内存在
   `comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd`。
3. 重启 ComfyUI。
4. 将 `H3_V100` 接在 H3 模型加载节点之后、采样器之前。

不要与其他独立修改同一 H3 Attention、MLP、QKV 或 output projection 方法的
扩展叠加。评估 TE-Speed 时，将它放在 H3 V100 之前，并保留下述两个启动
参数；缓存/跳步不属于本节点默认精确路线。

## 启动参数

当前验证过的 V100 16 GiB 路线应同时加入以下两个 ComfyUI 启动参数：

```text
--disable-dynamic-vram --lowvram
```

这是当前 V100 16 GiB 验证环境的兼容要求，不是性能开关。
`--disable-dynamic-vram` 避免动态权重预取与 H3 激活峰值同时扩张，
`--lowvram` 控制 GPU 常驻权重规模，为 AIMDO 权重流式加载保留空间。
PyTorch reserved cache 不等于 AIMDO 可直接使用的 driver-free VRAM；即使
allocator 内显示数 GiB 可复用缓存，下一段64 MiB直接权重拷贝仍可能 OOM。
两个参数不会关闭节点自己的 MLP/QKV 自动分块。更大显存和不同权重加载后端
尚未纳入当前验证范围。它们作用于整个 ComfyUI 进程，不能由节点内部开启。

V100 不建议启用 ComfyUI 全局参数 `--fast fp16_accumulation`。它与本节点的
混合精度不是同一功能。V100 验证测试中，该参数对 QKV 没有实质收益、使部分
MLP 略慢，并带来可测数值变化。不启用它不会关闭本节点的 FP16、混合精度或
Flash Attention。

## 边界

超长序列：在双 Tesla V100-SXM2 16 GiB、128 GiB 系统内存的公开测试配置下，
1280×736、15 秒（102,623 tokens）完成 8 步采样并成功生成视频，总耗时为
1 小时 51 分 58 秒。这是已完整验证的极限可运行档，不是日常推荐档。


## 参考项目与致谢

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)：模型和执行框架。
- [Sol-Attn 项目页](https://nvlabs.github.io/Sana/Sol-Attn/)与
  [论文](https://arxiv.org/abs/2607.24027)：训练无关的在线块稀疏
  Attention 方法参考。本节点针对 H3 与 V100/SM70 实现独立适配；Sol 是
  显式近似路线，论文在其他模型和硬件上的结果不代表本节点性能。
- [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch)：
  H3 V100混合精度拆分方案来源。本项目将修改源文件的方案改造成工作流范围
  节点，并增加音频安全和自适应内存保护。
- [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100)：
  SM70 Flash Attention实现来源。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) 4.2.0：原生内核构建所用
  CUTLASS/CuTe头文件。
- PyTorch SDPA：正确性、性能比较和安全回退参考。

完整署名和许可证说明见 `NOTICE.md` 和 `licenses/`。

## 许可证

由于 H3混合精度衍生部分采用 GPL-3.0-only，本项目整体以 GPL-3.0-only
分发。BSD-3-Clause组件保留原有声明。详情见 `NOTICE.md` 和 `licenses/`。

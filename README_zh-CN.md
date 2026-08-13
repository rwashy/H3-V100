# H3_V100

简体中文 | [English](README.md)

`H3_V100` 是用于 NVIDIA Tesla V100（SM70）运行 MiniMax H3 的 ComfyUI
自定义节点。它提供：

1. 经过实测的混合精度与 SM70 Flash Attention 加速；
2. 面向长序列的自适应显存保护（显存分块）。

节点不会修改 ComfyUI 或其他节点文件，只对流经节点的克隆 `MODEL` 应用
补丁，并严格检查预期的 50-block MiniMax H3 结构。

## 核心功能

节点只提供两个设置开关：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `mixed_precision` | 开 | FP16 QKV、主要 Attention 和已验证的 FP16 MLP linear；Q/K Norm、RoPE、SwiGLU、Attention output projection、残差及音频 query 重算保持 FP32。 |
| `flash_attention` | 开 | 启用 SM70 forward Attention 候选；不支持或更慢的形状自动回退。 |

自适应内存始终开启，独立于两个开关。关闭混合精度后，MLP/QKV/output
projection分块、缓存压力清理和动态权重预取保护仍然生效。若要完全绕过本
节点，请在工作流中旁路或删除它。

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
便携版中已经正常工作的 Torch。其他 Python 版本、Linux 或不同 ABI 环境需按
`native/BUILD.md` 重新编译 `.pyd`。

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

### 实测性能提升

在固定种子、864×480、5 秒、18,376 tokens、采样器运行 8 步且不跳步的热启动
对比中，不使用本节点的
原始路径耗时 **738 秒**。开启混合精度后，耗时降至 **380.11 秒**，吞吐速度提升
**94.2%**（1.94 倍），耗时减少 **48.5%**；在混合精度基础上继续开启 Flash
Attention，又减少 **39.94 秒**，最终耗时为 **340.17 秒**。相对仅开启混合精度，
吞吐速度再提升 **11.7%**，耗时再减少 **10.5%**。总体而言，相较不使用节点的
原始路径，吞吐速度提升 **117.0%**（2.17 倍），总耗时减少 **53.9%**。

测试过程中 GPU 曾出现过热降频，因此上述耗时和提升幅度属于偏保守的实测结果；
在散热充分、核心频率稳定的条件下，实际优化效率可能更高。

## 安装和使用

1. 将 `H3_V100` 文件夹复制到 `ComfyUI/custom_nodes/`。
2. 需要 Flash Attention 时，确认目录内存在
   `comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd`。
3. 重启 ComfyUI。
4. 将 `H3_V100` 接在 H3 模型加载节点之后、采样器之前。

不要与其他修改 MiniMax H3 Attention、MLP、QKV、output projection 或动态
显存行为的扩展叠加。

## 启动参数

V100 不建议启用 ComfyUI 全局参数 `--fast fp16_accumulation`。它与本节点的
混合精度不是同一功能。V100 本地测试中，该参数对 QKV 没有实质收益、使部分
MLP 略慢，并带来可测数值变化。不启用它不会关闭本节点的 FP16、混合精度或
Flash Attention。

## 边界

超长序列：在双 Tesla V100-SXM2 16 GiB、128 GiB 系统内存的公开测试配置下，
1280×736、15 秒（102,623 tokens）完成 8 步采样并成功生成视频，总耗时为
1 小时 51 分 58 秒。这是已完整验证的极限可运行档，不是日常推荐档。


## 参考项目与致谢

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)：模型和执行框架。
- [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch)：
  H3 V100混合精度拆分方案来源。本项目将修改源文件的方案改造成工作流范围
  节点，并增加音频安全和自适应内存保护。
- [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100)：
  SM70 Flash Attention实现来源。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) 4.2.0：原生内核构建所用
  CUTLASS/CuTe头文件。
- PyTorch SDPA：正确性、性能比较和安全回退参考。

完整署名和重编译说明见 `NOTICE.md`、`licenses/` 和 `native/BUILD.md`。

## 许可证

由于 H3混合精度衍生部分采用 GPL-3.0-only，本项目整体以 GPL-3.0-only
分发。BSD-3-Clause组件保留原有声明。详情见 `LICENSE`、`NOTICE.md` 和
`licenses/`。

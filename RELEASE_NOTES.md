# H3 V100 v1.4.0 更新说明

## 主要更新

- 适配 ComfyUI 默认 DynamicVRAM，由 ComfyUI VBAR 管理压缩权重驻留。
- H3 开始采样前自动释放已经失活的前阶段 CUDA Dynamic 模型，为单卡采样腾出
  显存。
- MLP 在一次调用内只准备一次 fc1/fc2 权重，并供全部 activation chunks
  复用，明显缩小冷启动与热启动的耗时差距。
- 固定启用经过验证的 scaled FP16 SwiGLU、自适应 MLP 分块和超长序列投影
  分块。
- 保留精确 `flash_attn` 与显式 `sol_attn` 两条路线；Flash 不会隐式进入 Sol。

## 启动参数

请从 ComfyUI 启动命令中删除：

```text
--disable-dynamic-vram
--lowvram
```

V100 不建议启用全局 `--fast fp16_accumulation`。修改启动命令后需要完整重启
ComfyUI，并重新加载扩散模型。

## 验证环境与结果

当前测试使用单张 NVIDIA Tesla V100 16 GiB。所有 CUDA 工作均在 `cuda:0`；
H3 与视频/音频 VAE 由 ComfyUI DynamicVRAM 和 CPU offload 管理，文本编码器
在 CPU 运行，没有进行多卡模型分配。

| 路线 | 状态 | 首轮 | 显示平均 | 完整任务 |
|---|---|---:|---:|---:|
| Flash | 冷启动 | 63.85 秒 | 54.70 秒/轮 | 563.77 秒 |
| Flash | 热启动 | 54.66 秒 | 55.41 秒/轮 | 490.84 秒 |
| Sol | 热启动 | 54.7 秒 | 约 54 秒/轮 | 485.69 秒 |

以上运行均正常生成画面和声音；耗时用于说明当前环境下的运行状态，不作为跨设备
性能保证。

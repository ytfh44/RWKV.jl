# RWKV.jl

RWKV模型的Julia实现。

## 介绍

RWKV (Receptance Weighted Key Value) 是一种创新的语言模型架构，它结合了RNN的高效推理和Transformer的出色性能。RWKV.jl旨在提供这一架构的Julia实现。

本实现基于[RWKV-7](https://github.com/BlinkDL/RWKV-LM)的结构，使用纯Julia代码从零开始构建。

## 特性

- 纯Julia实现，无需Python依赖（除了加载预训练模型时）
- 支持RWKV-7模型架构
- 提供从PyTorch权重文件加载预训练模型的功能
- 状态追踪，允许高效的增量推理

## 安装

```julia
using Pkg
Pkg.add(url="https://github.com/ytfh44/RWKV.jl")
```

## 使用

### 基本用法

```julia
using RWKV

# 创建模型配置
args = ModelArgs(
    n_layer = 24,        # 层数
    n_embd = 1024,       # 嵌入维度
    n_head = 16,         # 注意力头数
    head_size = 64,      # 每个头的大小
    vocab_size = 50277   # 词汇表大小
)

# 从预训练权重加载模型
model_file = "path/to/model.pth"
model = load_model_from_pytorch(model_file)

# 处理单个token
token = 1  # 假设这是您想要处理的token ID
logits, state = forward(model, token)

# 处理token序列
tokens = [1, 2, 3, 4]
logits, state = forward(model, tokens)

# 继续处理其他token，使用之前的状态
next_token = 5
logits, state = forward(model, next_token, state)
```

### 示例脚本

项目包含几个示例脚本，展示如何使用RWKV.jl：

1. `examples/minimal_example.jl` - 展示基本用法的最小示例
2. `test/runtests.jl` - 单元测试和更多示例

运行示例：

```bash
julia --project=. examples/minimal_example.jl
```

### 加载预训练模型

由于PyTorch模型格式与Julia不直接兼容，我们提供了从PyTorch权重文件加载模型的工具函数。您可以通过以下方式加载预训练模型：

```julia
using RWKV

# 方法1：直接从.pth文件加载（需要先转换为.npz）
model = load_model_from_pytorch("path/to/model.pth")

# 方法2：先将PyTorch权重转换为NPZ格式，再加载
npz_file = convert_pytorch_to_npz("path/to/model.pth")
model = load_model_from_pytorch(npz_file)
```

## 模型结构

RWKV-7模型包含以下核心组件：

1. **时间混合层 (Time Mixing)** - 处理序列中的时间依赖关系，类似于Transformer的注意力机制，但具有线性复杂度
2. **通道混合层 (Channel Mixing)** - 处理特征维度上的信息，类似于前馈网络
3. **层归一化 (Layer Normalization)** - 用于稳定训练过程
4. **嵌入层 (Embedding)** - 将输入token转换为向量表示

每个RWKV层由一个时间混合层和一个通道混合层组成，每层都有残差连接和层归一化。

## 开发

欢迎贡献！请通过提交Issue或Pull Request参与项目开发。

### 待办事项

- [ ] 优化性能
- [ ] 添加更多示例
- [ ] 实现训练功能
- [ ] 添加模型量化支持

## 许可

此项目采用MIT许可证。 
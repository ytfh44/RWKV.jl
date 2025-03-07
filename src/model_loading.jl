"""
模型加载模块，用于从PyTorch权重文件加载预训练的RWKV模型
"""
module ModelLoading

using NPZ
using BSON
using ..RWKV: ModelArgs, RWKVModel

export load_model_from_pytorch

"""
    load_model_from_pytorch(model_file::String)

从PyTorch权重文件加载RWKV模型。

## 参数
- `model_file::String`: PyTorch权重文件路径(.pth文件)

## 返回
- `model::RWKVModel`: 加载的RWKV模型
"""
function load_model_from_pytorch(model_file::String)
    if !endswith(model_file, ".pth") && !endswith(model_file, ".pt")
        error("模型文件必须是PyTorch格式(.pth或.pt)")
    end
    
    # 由于Julia没有直接的PyTorch权重加载机制，我们可以有几种选择:
    # 1. 使用Python PyCall调用torch.load
    # 2. 将PyTorch权重转换为NPZ格式，然后用NPZ.jl加载
    # 3. 将PyTorch权重转换为BSON格式，然后用BSON.jl加载
    
    # 这里我们假设已经将权重转换为NPZ格式
    # 可以通过以下Python代码完成转换:
    # ```python
    # import torch
    # import numpy as np
    # 
    # weights = torch.load("model.pth", map_location='cpu')
    # np.savez_compressed("model.npz", **{k: v.numpy() for k, v in weights.items()})
    # ```
    
    # 假设模型文件有对应的NPZ版本
    npz_file = replace(model_file, r"\.(pth|pt)$" => ".npz")
    
    if !isfile(npz_file)
        error("找不到对应的NPZ文件: $npz_file 
              请先使用Python将PyTorch权重转换为NPZ格式。")
    end
    
    # 加载NPZ文件
    weights = NPZ.npzread(npz_file)
    
    # 提取模型参数
    try
        # 从emb.weight的形状推断vocab_size和n_embd
        emb_weight = weights["emb.weight"]
        vocab_size, n_embd = size(emb_weight)
        
        # 计算n_layer
        layer_keys = [k for k in keys(weights) if startswith(k, "blocks.")]
        max_layer_idx = maximum([parse(Int, match(r"blocks\.(\d+)\.", k).captures[1]) for k in layer_keys if occursin(r"blocks\.\d+\.", k)])
        n_layer = max_layer_idx + 1
        
        # 计算n_head和head_size
        head_size_keys = [k for k in keys(weights) if endswith(k, ".time_r_k")]
        if length(head_size_keys) > 0
            head_size_array = weights[head_size_keys[1]]
            n_head_size = length(head_size_array)
            # 假设head_size是64，这是RWKV-7的典型值
            head_size = 64
            n_head = n_head_size ÷ head_size
        else
            # 如果找不到相关参数，使用默认值
            n_head = n_embd ÷ 64
            head_size = 64
        end
        
        # 创建ModelArgs
        args = ModelArgs(n_layer, n_embd, n_head, head_size, vocab_size)
        
        # 转换权重为Dict{String, Array{Float32}}
        params = Dict{String, Array{Float32}}()
        for (k, v) in weights
            params[k] = Float32.(v)
        end
        
        # 创建并返回RWKVModel
        return RWKVModel(args, params)
    catch e
        error("加载模型时出错: $e")
    end
end

"""
    convert_pytorch_to_npz(model_file::String)

将PyTorch模型文件转换为NPZ格式。
需要Python和PyTorch环境。

## 参数
- `model_file::String`: PyTorch权重文件路径(.pth或.pt文件)
"""
function convert_pytorch_to_npz(model_file::String)
    # 需要使用Python来完成转换
    # 这里我们使用系统命令调用Python脚本
    
    # 创建一个临时Python脚本
    python_script = """
import torch
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py input.pth output.npz")
    sys.exit(1)

model_file = sys.argv[1]
npz_file = sys.argv[2]

try:
    weights = torch.load(model_file, map_location='cpu')
    # 将所有张量转换为numpy数组
    numpy_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    np.savez_compressed(npz_file, **numpy_weights)
    print(f"成功将{model_file}转换为{npz_file}")
except Exception as e:
    print(f"转换失败: {e}")
    sys.exit(1)
"""
    
    # 写入临时脚本文件
    script_path = tempname() * ".py"
    open(script_path, "w") do io
        write(io, python_script)
    end
    
    # 确定输出NPZ文件名
    npz_file = replace(model_file, r"\.(pth|pt)$" => ".npz")
    
    # 调用Python脚本
    cmd = `python $script_path $model_file $npz_file`
    result = run(cmd)
    
    # 清理临时文件
    rm(script_path)
    
    if result.exitcode != 0
        error("转换失败，请确保已安装Python和PyTorch，并且可以访问模型文件")
    end
    
    return npz_file
end

end # module 
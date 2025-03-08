#!/usr/bin/env julia

# 添加当前项目到路径
using Pkg
Pkg.activate(dirname(dirname(abspath(@__FILE__))))

using LinearAlgebra
using Statistics
using Distributions

# 模型参数结构体
struct ModelArgs
    n_layer::Int
    n_embd::Int
    n_head::Int
    head_size::Int
    vocab_size::Int
end

# 简单的模型结构
struct SimpleModel
    args::ModelArgs
    params::Dict{String, Array{Float32}}
end

# 简单的前向传播函数
function forward(model::SimpleModel, token::Int)
    # 获取token的嵌入向量
    if token < 1 || token > model.args.vocab_size
        error("Token值 $token 超出了词汇表大小范围 (1-$(model.args.vocab_size))")
    end
    
    # 获取嵌入向量
    x = model.params["emb.weight"][token, :]
    
    # 简单的线性变换
    logits = model.params["head.weight"] * x
    
    return logits
end

# 创建一个随机初始化的测试模型
function create_test_model(n_layer=2, n_embd=4, n_head=2, head_size=2, vocab_size=128)
    # 创建测试参数
    params = Dict{String, Array{Float32}}()
    
    # 嵌入层
    params["emb.weight"] = rand(Float32, vocab_size, n_embd)
    
    # 输出层
    params["head.weight"] = rand(Float32, vocab_size, n_embd)
    
    # 创建模型
    args = ModelArgs(n_layer, n_embd, n_head, head_size, vocab_size)
    return SimpleModel(args, params)
end

# 简单的字符级分词器
function text_to_token(text)
    # 简单地将第一个字符转换为ASCII码
    if length(text) > 0
        return Int(text[1])
    else
        return 0
    end
end

# 将token转换回文本
function token_to_text(tokens)
    # 将ASCII码转换回字符
    return join([Char(t) for t in tokens])
end

# softmax函数
function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

# 采样函数
function sample(logits, temperature=1.0)
    # 温度采样实现
    logits = logits ./ temperature
    probs = softmax(logits)
    return rand(Categorical(probs))
end

# 生成函数
function generate(model, prompt; max_tokens=20, temperature=0.9)
    tokens = [text_to_token(prompt)]  # 文本编码
    
    # 确保起始token在有效范围内
    if tokens[1] < 1 || tokens[1] > model.args.vocab_size
        error("起始token值 $(tokens[1]) 超出了词汇表大小范围 (1-$(model.args.vocab_size))")
    end
    
    for _ in 1:max_tokens
        logits = forward(model, last(tokens))
        next_token = sample(logits, temperature)
        push!(tokens, next_token)
    end
    
    return token_to_text(tokens)  # 文本解码
end

function main()
    # 使用随机模型进行测试
    println("创建随机初始化的测试模型...")
    model = create_test_model(2, 4, 2, 2, 128)
    
    # 打印模型信息
    println("\n模型参数:")
    println("  层数: $(model.args.n_layer)")
    println("  嵌入维度: $(model.args.n_embd)")
    println("  注意力头数: $(model.args.n_head)")
    println("  头尺寸: $(model.args.head_size)")
    println("  词汇表大小: $(model.args.vocab_size)")
    
    # 简单推理测试
    test_text = "A"  # 使用简单的字符，确保ASCII码在词汇表范围内
    token = text_to_token(test_text)
    println("\n输入: $test_text (token: $token)")
    
    # 确保token在有效范围内
    if token >= 1 && token <= model.args.vocab_size
        logits = forward(model, token)
        println("预测top5: ", sortperm(logits, rev=true)[1:5])
        
        # 生成测试
        println("\n生成测试:")
        generated = generate(model, "A", max_tokens=20)
        println("生成结果: $generated")
    else
        println("错误: token值 $token 超出了词汇表大小范围 (1-$(model.args.vocab_size))")
    end
    
    println("\n要加载RWKV7模型，需要以下步骤:")
    println("1. 安装PyTorch: pip install torch")
    println("2. 运行转换脚本: python3 convert_model.py examples/assets/RWKV7-G1-0.1B-68%25trained-20250303-ctx4k.pth")
    println("3. 修改此示例代码，使用生成的.npz文件")
end

# 运行主函数
main() 
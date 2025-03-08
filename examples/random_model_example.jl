#!/usr/bin/env julia

# 添加当前项目到路径
using Pkg
Pkg.activate(dirname(dirname(abspath(@__FILE__))))

using RWKV
using NPZ
using LinearAlgebra
using Statistics
using Distributions

# 简单的字符级分词器
function text_to_token(text)
    # 简单地将第一个字符转换为ASCII码
    if length(text) > 0
        return Int(text[1])
    else
        return 0
    end
end

function token_to_text(tokens)
    # 将ASCII码转换回字符
    return join([Char(t) for t in tokens])
end

function sample(logits, temperature=1.0)
    # 温度采样实现
    logits = logits ./ temperature
    probs = softmax(logits)
    return rand(Categorical(probs))
end

function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

"""
创建一个随机初始化的测试模型
"""
function create_test_model(n_layer=2, n_embd=4, n_head=2, head_size=2, vocab_size=10)
    # 创建测试参数
    params = Dict{String, Array{Float32}}()
    
    # 嵌入层
    params["emb.weight"] = rand(Float32, vocab_size, n_embd)
    params["blocks.0.ln0.weight"] = ones(Float32, n_embd)
    params["blocks.0.ln0.bias"] = zeros(Float32, n_embd)
    
    # 对每一层创建参数
    for i in 0:n_layer-1
        # 层归一化参数
        params["blocks.$i.ln1.weight"] = ones(Float32, n_embd)
        params["blocks.$i.ln1.bias"] = zeros(Float32, n_embd)
        params["blocks.$i.ln2.weight"] = ones(Float32, n_embd)
        params["blocks.$i.ln2.bias"] = zeros(Float32, n_embd)
        
        # 时间混合层参数
        params["blocks.$i.att.time_mix_r"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_mix_w"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_mix_k"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_mix_v"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_mix_a"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_mix_g"] = rand(Float32, n_embd)
        params["blocks.$i.att.time_w_bias"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_r_k"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_w1.weight"] = rand(Float32, n_embd, n_embd)
        params["blocks.$i.att.time_w2.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_a1.weight"] = rand(Float32, n_embd, n_embd)
        params["blocks.$i.att.time_a2.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_a_bias"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_g1.weight"] = rand(Float32, n_embd, n_embd)
        params["blocks.$i.att.time_g2.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_v1.weight"] = rand(Float32, n_embd, n_embd)
        params["blocks.$i.att.time_v2.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_v_bias"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_k_k"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_k_a"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_r.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_k.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_v.weight"] = rand(Float32, n_head * head_size, n_embd)
        params["blocks.$i.att.time_o.weight"] = rand(Float32, n_embd, n_head * head_size)
        params["blocks.$i.att.time_ln_w"] = rand(Float32, n_head * head_size)
        params["blocks.$i.att.time_ln_b"] = rand(Float32, n_head * head_size)
        
        # 通道混合层参数
        params["blocks.$i.ffn.time_mix_k"] = rand(Float32, n_embd)
        params["blocks.$i.ffn.key.weight"] = rand(Float32, n_embd, n_embd)
        params["blocks.$i.ffn.value.weight"] = rand(Float32, n_embd, n_embd)
    end
    
    # 输出层参数
    params["ln_out.weight"] = ones(Float32, n_embd)
    params["ln_out.bias"] = zeros(Float32, n_embd)
    params["head.weight"] = rand(Float32, vocab_size, n_embd)
    
    # 创建模型
    args = ModelArgs(n_layer, n_embd, n_head, head_size, vocab_size)
    return RWKVModel(args, params)
end

"""
简单打印权重矩阵的形状，用于调试
"""
function print_weights_shapes(model)
    println("打印模型权重形状:")
    for (key, value) in model.params
        println("  $key: $(size(value))")
    end
end

# 添加生成函数
function generate(model, prompt; max_tokens=20, temperature=0.9)
    tokens = [text_to_token(prompt)]  # 需要实现文本编码
    for _ in 1:max_tokens
        logits = forward(model, last(tokens))
        next_token = sample(logits, temperature)
        push!(tokens, next_token)
    end
    return token_to_text(tokens)  # 需要实现文本解码
end

function main()
    # 使用随机模型进行测试
    println("创建随机初始化的测试模型...")
    # 使用更大的词汇表大小，确保能够处理ASCII字符
    model = create_test_model(2, 4, 2, 2, 128)
    
    # 打印权重形状，用于调试
    print_weights_shapes(model)
    
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
#!/usr/bin/env julia

# 添加当前项目到路径
using Pkg
Pkg.activate(dirname(dirname(abspath(@__FILE__))))

using RWKV
using NPZ
using LinearAlgebra
using Statistics
using Tokenizers  # 需要安装Tokenizers.jl

const tokenizer = Tokenizer.from_pretrained("gpt2")  # 示例使用GPT-2分词器

function text_to_token(text)
    return encode(tokenizer, text).ids[1]  # 简单取第一个token
end

function token_to_text(tokens)
    return decode(tokenizer, tokens)
end

function sample(logits, temperature=1.0)
    # 温度采样实现
    logits = logits ./ temperature
    probs = softmax(logits)
    return rand(Categorical(probs))
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

"""
示例：使用随机初始化的小模型
"""
function example_with_random_model()
    println("创建随机初始化的测试模型...")
    model = create_test_model()
    
    # 打印权重形状，用于调试
    print_weights_shapes(model)
    
    # 只进行简单的测试
    println("执行简单的测试...")
    token = 1
    println("  模型参数:")
    println("    层数: $(model.args.n_layer)")
    println("    嵌入维度: $(model.args.n_embd)")
    println("    注意力头数: $(model.args.n_head)")
    println("    头尺寸: $(model.args.head_size)")
    
    # 跳过实际的前向传播测试
    println("跳过实际的前向传播，因为需要更多的调试")
    
    return model
end

"""
示例：如果有预训练模型，从文件加载并使用
"""
function example_with_pretrained_model(model_file)
    if !isfile(model_file)
        println("找不到预训练模型文件: $model_file")
        return nothing
    end
    
    try
        println("从NPZ文件加载预训练模型...")
        model = load_model_from_pytorch(model_file)
        
        # 打印模型信息
        println("模型参数:")
        println("  层数: $(model.args.n_layer)")
        println("  嵌入维度: $(model.args.n_embd)")
        println("  注意力头数: $(model.args.n_head)")
        println("  头尺寸: $(model.args.head_size)")
        println("  词汇表大小: $(model.args.vocab_size)")
        
        # 添加推理示例
        println("\n运行推理测试...")
        test_token = 3  # 示例输入
        logits = forward(model, test_token)
        println("  输入token: $test_token")
        println("  输出logits: ", logits[1:min(5, end)])  # 显示前5个logits
        
        # 添加生成示例
        println("\n生成示例:")
        generated = generate(model, "The", max_tokens=20)
        println("生成结果: $generated")
        
        return model
    catch e
        println("加载预训练模型失败: $e")
        return nothing
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

function load_model_from_pytorch(npz_file)
    weights = npzread(npz_file)
    params = Dict{String, Array{Float32}}()
    
    # 转换参数名称以匹配Julia实现
    for (k, v) in weights
        new_key = replace(k, "weight" => "weight")
        new_key = replace(new_key, "_" => ".")
        params[new_key] = convert(Array{Float32}, v)
    end
    
    # 推断模型参数
    n_layer = count(startswith("blocks."), keys(params)) ÷ 20  # 根据参数数量估算层数
    n_embd = size(params["emb.weight"], 2)
    vocab_size = size(params["emb.weight"], 1)
    
    args = ModelArgs(n_layer, n_embd, 0, 0, vocab_size)  # head参数需要根据实际调整
    return RWKVModel(args, params, [LayerState(zeros(n_embd), nothing, zeros(0,0,0)) for _ in 1:n_layer])
end

function main()
    # 初始化模型
    model = example_with_pretrained_model("examples/assets/rwkv-4-pile-430m.npz")
    
    # 简单推理测试
    test_text = "Hello"
    token = text_to_token(test_text)
    logits = forward(model, token)
    println("输入: $test_text")
    println("预测top5: ", sortperm(logits, rev=true)[1:5])
    
    # 生成测试
    println("\n生成测试:")
    generated = generate(model, "The future of AI", max_tokens=20)
    println(generated)
end

# 运行主函数
main() 
module RWKV

using LinearAlgebra
using Statistics # 添加Statistics包，用于mean和var函数

# 禁用预编译以避免方法重载错误
__precompile__(false)

# 导出函数和类型
export ModelArgs, RWKVModel, forward

"""
    ModelArgs
    
RWKV模型的配置参数
"""
struct ModelArgs
    n_layer::Int
    n_embd::Int
    n_head::Int
    head_size::Int
    vocab_size::Int
end

"""
    RWKVModel

RWKV模型结构
"""
struct RWKVModel
    args::ModelArgs
    params::Dict{String, Array{Float32}}
end

"""
    layer_norm(x, w, b)

实现层归一化操作
"""
function layer_norm(x, w, b)
    x_mean = mean(x)
    x_var = var(x, corrected=false)  # 使用无偏估计
    x_normalized = (x .- x_mean) ./ sqrt(x_var .+ 1e-5)
    return x_normalized .* w .+ b
end

"""
    group_norm(x, w, b)

实现组归一化操作
"""
function group_norm(x, w, b)
    x_mean = mean(x, dims=1, keepdims=true)
    x_var = var(x, dims=1, keepdims=true)
    return ((x .- x_mean) ./ sqrt.(x_var .+ 64e-5))[:] .* w .+ b
end

"""
    sigmoid(x)

实现sigmoid激活函数
"""
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

"""
    time_mixing(x, v0, last_x, S, params)

实现RWKV的时间混合层
"""
function time_mixing(x, v0, last_x, S, params)
    mr, mw, mk, mv, ma, mg, w_bias, r_k, Ww1, Ww2, Wa1, Wa2, a_bias, Wg1, Wg2 = params[1:15]
    k_k, k_a, Wr, Wk, Wv, Wo, ln_w, ln_b = params[end-7:end]
    
    n_head = size(S, 1)
    head_size = size(S, 2)
    
    # 确保所有输入是Float32类型
    x = Float32.(x)
    last_x = Float32.(last_x)
    
    # 计算各种混合状态
    xr = x .+ mr .* (last_x .- x)
    xw = x .+ mw .* (last_x .- x)
    xk = x .+ mk .* (last_x .- x)
    xv = x .+ mv .* (last_x .- x)
    xa = x .+ ma .* (last_x .- x)
    xg = x .+ mg .* (last_x .- x)
    
    # 矩阵乘法，确保维度匹配
    r = Wr * xr
    w = exp.(-sigmoid.(tanh.(Ww1 * xw) * Ww2 .+ w_bias) ./ exp(0.5))
    k = Wk * xk
    v = Wv * xv
    
    if v0 === nothing
        v0 = v
    else
        Wv2, Wv1, v_bias = params[16:18]
        v .+= (v0 .- v) .* sigmoid.(Wv1 * xv * Wv2 .+ v_bias)
    end
    
    a = sigmoid.(Wa1 * xa * Wa2 .+ a_bias)
    g = sigmoid.(Wg1 * xg) * Wg2
    kk = k .* k_k
    k .+= k .* (a .- 1.0) .* k_a
    
    # 重塑张量形状
    r_reshaped = reshape(r, n_head, head_size, 1)
    w_reshaped = reshape(w, n_head, head_size, 1)
    k_reshaped = reshape(k, n_head, head_size, 1)
    v_reshaped = reshape(v, n_head, head_size, 1)
    kk_reshaped = reshape(kk, n_head, head_size, 1)
    a_reshaped = reshape(a, n_head, head_size, 1)
    r_k_reshaped = reshape(r_k, n_head, head_size, 1)
    
    # 对kk进行归一化
    for i in 1:n_head
        norm_val = maximum([norm(kk_reshaped[i,:,:]), 1e-12])
        kk_reshaped[i,:,:] ./= norm_val
    end
    
    # 更新状态S
    S = S .* permutedims(w_reshaped, (1, 3, 2)) .- 
        S * permutedims(kk_reshaped .* (kk_reshaped .* a_reshaped), (1, 3, 2)) .+ 
        v_reshaped .* permutedims(k_reshaped, (1, 3, 2))
    
    y = S * r_reshaped
    
    # 应用组归一化和添加项
    y = group_norm(y, ln_w, ln_b)
    y .+= reshape(sum(r_reshaped .* k_reshaped .* r_k_reshaped, dims=1), :, 1) .* reshape(v_reshaped, :)
    
    return Wo * (y .* g), v0, x, S
end

"""
    channel_mixing(x, last_x, mix, Wk, Wv)

实现RWKV的通道混合层
"""
function channel_mixing(x, last_x, mix, Wk, Wv)
    k = Wk * (x .+ mix .* (last_x .- x))
    v = Wv * (max.(k, 0.0) .^ 2)
    return v, x
end

"""
    get_params(model::RWKVModel, prefix::String)

获取给定前缀的参数列表
"""
function get_params(model::RWKVModel, prefix::String)
    return [model.params[key] for key in keys(model.params) if startswith(key, prefix)]
end

"""
    forward(model::RWKVModel, token::Int, state=nothing)

RWKV模型的前向传播函数
"""
function forward(model::RWKVModel, token::Int, state=nothing)
    args = model.args
    n_layer = args.n_layer
    n_embd = args.n_embd
    n_head = args.n_head
    head_size = args.head_size
    
    # 初始化状态如果没有提供
    if state === nothing
        state = (
            zeros(Float32, (n_layer, 2, n_embd)),
            zeros(Float32, (n_layer, n_head, head_size, head_size))
        )
    end
    
    # 嵌入层
    x = model.params["emb.weight"][token+1, :]  # Julia是1-indexed，而Python是0-indexed
    x = layer_norm(x, model.params["blocks.0.ln0.weight"], model.params["blocks.0.ln0.bias"])
    
    v0 = nothing
    for i in 0:n_layer-1
        # 时间混合层
        x_ = layer_norm(x, model.params["blocks.$i.ln1.weight"], model.params["blocks.$i.ln1.bias"])
        
        # 构建时间混合层的参数
        att_params = [
            model.params["blocks.$i.att.time_mix_r"],
            model.params["blocks.$i.att.time_mix_w"],
            model.params["blocks.$i.att.time_mix_k"],
            model.params["blocks.$i.att.time_mix_v"],
            model.params["blocks.$i.att.time_mix_a"],
            model.params["blocks.$i.att.time_mix_g"],
            model.params["blocks.$i.att.time_w_bias"],
            model.params["blocks.$i.att.time_r_k"],
            model.params["blocks.$i.att.time_w1.weight"],
            model.params["blocks.$i.att.time_w2.weight"],
            model.params["blocks.$i.att.time_a1.weight"],
            model.params["blocks.$i.att.time_a2.weight"],
            model.params["blocks.$i.att.time_a_bias"],
            model.params["blocks.$i.att.time_g1.weight"],
            model.params["blocks.$i.att.time_g2.weight"]
        ]
        
        # 如果有time_v1和time_v2参数
        if haskey(model.params, "blocks.$i.att.time_v1.weight")
            push!(att_params, model.params["blocks.$i.att.time_v2.weight"])
            push!(att_params, model.params["blocks.$i.att.time_v1.weight"])
            push!(att_params, model.params["blocks.$i.att.time_v_bias"])
        end
        
        push!(att_params, model.params["blocks.$i.att.time_k_k"])
        push!(att_params, model.params["blocks.$i.att.time_k_a"])
        push!(att_params, model.params["blocks.$i.att.time_r.weight"])
        push!(att_params, model.params["blocks.$i.att.time_k.weight"])
        push!(att_params, model.params["blocks.$i.att.time_v.weight"])
        push!(att_params, model.params["blocks.$i.att.time_o.weight"])
        push!(att_params, model.params["blocks.$i.att.time_ln_w"])
        push!(att_params, model.params["blocks.$i.att.time_ln_b"])
        
        dx, v0, state[1][i+1,1,:], state[2][i+1,:,:,:] = time_mixing(
            x_, v0, state[1][i+1,1,:], state[2][i+1,:,:,:], att_params
        )
        x = x .+ dx
        
        # 通道混合层
        x_ = layer_norm(x, model.params["blocks.$i.ln2.weight"], model.params["blocks.$i.ln2.bias"])
        dx, state[1][i+1,2,:] = channel_mixing(
            x_, 
            state[1][i+1,2,:], 
            model.params["blocks.$i.ffn.time_mix_k"],
            model.params["blocks.$i.ffn.key.weight"],
            model.params["blocks.$i.ffn.value.weight"]
        )
        x = x .+ dx
    end
    
    # 输出层
    x = layer_norm(x, model.params["ln_out.weight"], model.params["ln_out.bias"])
    logits = model.params["head.weight"] * x
    
    return logits, state
end

"""
    forward(model::RWKVModel, tokens::Vector{Int}, state=nothing)

处理一个token序列，返回最后一个token的logits和最终状态
"""
function forward(model::RWKVModel, tokens::Vector{Int}, state=nothing)
    for token in tokens
        logits, state = forward(model, token, state)
    end
    return logits, state
end

# 包含模型加载模块
include("model_loading.jl")
# 重新导出模型加载函数
export load_model_from_pytorch, convert_pytorch_to_npz

end # module 
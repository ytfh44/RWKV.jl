mutable struct LayerState
    x_prev::Vector{Float32}
    v0::Union{Vector{Float32}, Nothing}
    S::Matrix{Float32}  # 新增状态矩阵
end

struct RWKVModel
    args::ModelArgs
    params::Dict{String, Array{Float32}}
    state::Vector{LayerState}  # 新增状态存储
end

# 修改模型创建函数
function create_test_model(n_layer=2, n_embd=4, n_head=2, head_size=2, vocab_size=10)
    # ...原有参数创建代码...
    
    # 初始化状态（包含S矩阵）
    state = [LayerState(zeros(n_embd), nothing, zeros(n_head, head_size, head_size)) 
            for _ in 1:n_layer]
    
    return RWKVModel(args, params, state)
end

function forward(model::RWKVModel, token::Int)
    x = model.params["emb.weight"][token, :]
    
    for i in 0:(model.args.n_layer-1)
        # 层归一化
        ln_w = model.params["blocks.$i.ln1.weight"]
        ln_b = model.params["blocks.$i.ln1.bias"]
        x = layer_norm(x, ln_w, ln_b)
        
        # 时间混合（传入状态）
        x = time_mixing(model, i, x, model.state)
        
        # 通道混合（类似实现）
        x = channel_mixing(model, i, x)
    end
    
    # 输出层
    x = layer_norm(x, model.params["ln_out.weight"], model.params["ln_out.bias"])
    return model.params["head.weight"] * x
end

function time_mixing(model, layer_idx, x, state)
    prefix = "blocks.$layer_idx.att."
    p = model.params
    
    # 获取所有必要参数
    mr = p[prefix*"time_mix_r"]; mw = p[prefix*"time_mix_w"]; mk = p[prefix*"time_mix_k"]
    mv = p[prefix*"time_mix_v"]; ma = p[prefix*"time_mix_a"]; mg = p[prefix*"time_mix_g"]
    w_bias = p[prefix*"time_w_bias"]; r_k = p[prefix*"time_r_k"]
    Ww1 = p[prefix*"time_w1.weight"]; Ww2 = p[prefix*"time_w2.weight"]
    Wa1 = p[prefix*"time_a1.weight"]; Wa2 = p[prefix*"time_a2.weight"]; a_bias = p[prefix*"time_a_bias"]
    Wg1 = p[prefix*"time_g1.weight"]; Wg2 = p[prefix*"time_g2.weight"]
    Wv1 = p[prefix*"time_v1.weight"]; Wv2 = p[prefix*"time_v2.weight"]; v_bias = p[prefix*"time_v_bias"]
    k_k = p[prefix*"time_k_k"]; k_a = p[prefix*"time_k_a"]
    Wr = p[prefix*"time_r.weight"]; Wk = p[prefix*"time_k.weight"]
    Wv = p[prefix*"time_v.weight"]; Wo = p[prefix*"time_o.weight"]
    ln_w = p[prefix*"time_ln_w"]; ln_b = p[prefix*"time_ln_b"]
    
    # 与Python实现对齐的混合计算
    last_x = state[layer_idx+1].x_prev
    xr = x + mr .* (last_x - x)
    xw = x + mw .* (last_x - x)
    xk = x + mk .* (last_x - x)
    xv = x + mv .* (last_x - x)
    xa = x + ma .* (last_x - x)
    xg = x + mg .* (last_x - x)
    
    # 实现完整的时间混合逻辑
    r = Wr * xr
    w = exp.(-sigmoid.(tanh.(xw * Ww1) * Ww2 .+ w_bias) ./ exp(0.5))
    k = Wk * xk
    v = Wv * xv
    
    # 状态更新逻辑
    if state[layer_idx+1].v0 === nothing
        v0 = v
    else
        v += (state[layer_idx+1].v0 - v) .* sigmoid.(xv * Wv1 * Wv2 .+ v_bias)
    end
    
    # 实现S矩阵计算
    a = sigmoid.(xa * Wa1 * Wa2 .+ a_bias)
    g = sigmoid.(xg * Wg1) * Wg2
    kk = k .* k_k
    k += k .* (a .- 1) .* k_a
    
    # 重塑为多头格式
    head_size = model.args.head_size
    n_head = model.args.n_head
    r = reshape(r, (n_head, head_size, 1))
    w = reshape(w, (n_head, head_size, 1))
    k = reshape(k, (n_head, head_size, 1))
    v = reshape(v, (n_head, head_size, 1))
    kk = reshape(kk, (n_head, head_size, 1))
    a = reshape(a, (n_head, head_size, 1))
    r_k = reshape(r_k, (n_head, head_size, 1))
    
    # 更新状态矩阵S
    S = state[layer_idx+1].S
    S = S .* transpose(w) .- S * transpose(kk) .* (kk .* a) .+ v .* transpose(k)
    y = S * r
    
    # 组归一化实现
    y = group_norm(y, ln_w, ln_b)
    y += sum(r .* k .* r_k, dims=2) .* v
    output = Wo * (y .* g)
    
    # 更新状态
    state[layer_idx+1] = LayerState(x, v0, S)
    return output
end

function group_norm(x::Array{Float32}, w::Array{Float32}, b::Array{Float32})
    n_head = size(x, 1)
    x_reshaped = reshape(x, (n_head, :))
    μ = mean(x_reshaped, dims=2)
    σ = var(x_reshaped, dims=2) .+ 64e-5
    normalized = (x_reshaped .- μ) ./ sqrt.(σ)
    return normalized .* w' .+ b'
end 
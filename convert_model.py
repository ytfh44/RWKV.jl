#!/usr/bin/env python3
"""
将PyTorch模型转换为NPZ格式，以便在Julia中加载
"""

import torch
import numpy as np
import os
import sys
import argparse

def convert_model(input_file, output_file):
    """
    将PyTorch模型转换为NPZ格式
    """
    print(f"加载模型: {input_file}")
    state_dict = torch.load(input_file, map_location=torch.device("cpu"))
    
    # 如果state_dict包含model_state_dict键，则使用它
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # 打印一些键以便调试
    print("模型包含以下键的子集:")
    for i, k in enumerate(list(state_dict.keys())[:5]):
        print(f"  {k}")
    
    # 转换为NumPy数组
    numpy_dict = {}
    for k, v in state_dict.items():
        numpy_dict[k] = v.detach().cpu().numpy()
    
    # 保存为NPZ格式
    print(f"保存为NPZ格式: {output_file}")
    np.savez(output_file, **numpy_dict)
    print("转换完成!")

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为NPZ格式")
    parser.add_argument("input_file", help="输入的PyTorch模型文件(.pth)")
    parser.add_argument("--output_file", help="输出的NPZ文件名")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return 1
    
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".npz"
    
    convert_model(input_file, output_file)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
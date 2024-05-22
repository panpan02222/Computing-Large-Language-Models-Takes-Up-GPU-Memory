#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   calc_mode_params.py
@Time    :   2024/05/21 13:35:35
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :  计算大模型训练时占用显存大小 
'''

# from method import calculate_modelmem

def calculate_modelmem(typesize:str, params:int):
    """
    calculate the model memory usage
    """
    if typesize == 'float32':
        typesize = 4
    elif typesize == 'float16':
        typesize = 2
    elif typesize == 'bfloat16':
        typesize = 2
    elif typesize == 'int8':
        typesize = 1
    else:
        raise ValueError('typesize should be float32 or float16')
    modelmem = params * typesize
    return modelmem

def calculate_model_train_usage_gpu_memory():
    pass

typesize = input("Enter the size of a single data type in bytes: ")

params = input("Enter the number of parameters in the model: ")

modelmem = calculate_modelmem(typesize, int(params))

print("The model requires approximately", modelmem, "bytes of GPU memory.")

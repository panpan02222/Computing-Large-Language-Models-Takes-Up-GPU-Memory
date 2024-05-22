#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   method.py
@Time    :   2024/05/21 13:35:28
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''

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
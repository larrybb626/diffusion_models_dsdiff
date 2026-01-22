# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：JY_Network.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/7/18 17:11 
"""
class JunyangFramework:
    def __init__(self,password="Junyang is the best!"):
        self.password = password
        if self.password != "Junyang is the best!":
            raise ValueError("Wrong password!")
    def get_model(self, model):
        return model
    def get_config(self, config):
        return config

#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 模型推理包
'''

from .inference import TurtleBotInference
from .turtlebot_inference import TurtleBotDirectInference

__all__ = ['TurtleBotInference', 'TurtleBotDirectInference']

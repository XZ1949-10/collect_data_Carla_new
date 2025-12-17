#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 推理启动脚本
'''

import argparse
import sys
import rospy

from inference import TurtleBotInference


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='TurtleBot 模型推理')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--net-structure', type=int, default=2,
                        help='网络结构类型 (默认: 2)')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID，-1 表示使用 CPU (默认: 0)')
    
    # TurtleBot 参数
    parser.add_argument('--turtlebot', '-t', type=str, default='burger',
                        choices=['burger', 'waffle', 'waffle_pi', 
                                 'turtlebot1', 'turtlebot2', 'kobuki'],
                        help='TurtleBot 型号 (默认: burger)')
    
    # 控制参数
    parser.add_argument('--rate', '-r', type=int, default=10,
                        help='控制频率 Hz (默认: 10)')
    parser.add_argument('--joystick', '-j', type=str, default='xbox',
                        choices=['xbox', 'ps4'],
                        help='手柄类型 (默认: xbox)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                        help='运行时长（秒），不指定则无限运行')
    
    # ROS 话题
    parser.add_argument('--image-topic', type=str, default=None,
                        help='图像话题')
    parser.add_argument('--odom-topic', type=str, default=None,
                        help='里程计话题')
    parser.add_argument('--cmd-vel-topic', type=str, default=None,
                        help='速度命令话题')
    parser.add_argument('--joy-topic', type=str, default=None,
                        help='手柄话题')
    
    args = parser.parse_args()
    
    # 初始化 ROS 节点
    rospy.init_node('turtlebot_inference', anonymous=True)
    
    # 创建推理器
    inference = TurtleBotInference(
        model_path=args.model,
        turtlebot_model=args.turtlebot,
        gpu_id=args.gpu,
        joystick_type=args.joystick,
        control_rate=args.rate,
        image_topic=args.image_topic,
        odom_topic=args.odom_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        joy_topic=args.joy_topic,
        net_structure=args.net_structure,
    )
    
    try:
        # 加载模型
        inference.load_model()
        
        # 设置 ROS 接口
        inference.setup_ros()
        
        # 运行推理
        inference.run(duration=args.duration)
        
        # 打印统计
        inference.print_statistics()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        inference.cleanup()


if __name__ == '__main__':
    main()

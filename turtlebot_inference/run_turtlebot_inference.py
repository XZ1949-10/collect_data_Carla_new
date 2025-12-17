#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 推理启动脚本 (直接预测 linear_vel 和 angular_vel)

使用方法:
    python run_turtlebot_inference.py --model ./model.pth --turtlebot-model burger
'''

import argparse
import rospy
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlebot_inference import TurtleBotDirectInference


def main():
    parser = argparse.ArgumentParser(description='TurtleBot 端到端推理 (直接预测速度)')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--net-structure', type=int, default=2,
                        choices=[1, 2, 3],
                        help='网络结构类型 (1=纯回归, 2=独立不确定性, 3=共享不确定性)')
    
    # TurtleBot 参数
    parser.add_argument('--turtlebot-model', type=str, default='burger',
                        choices=['burger', 'waffle', 'waffle_pi', 'turtlebot2', 'kobuki'],
                        help='TurtleBot 型号')
    
    # 控制参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID，-1 表示使用 CPU')
    parser.add_argument('--rate', type=int, default=10,
                        help='控制频率 (Hz)')
    parser.add_argument('--duration', type=float, default=None,
                        help='运行时长（秒），不指定则无限运行')
    parser.add_argument('--print-interval', type=int, default=10,
                        help='状态打印间隔（帧数）')
    
    # 手柄参数
    parser.add_argument('--joystick', '-j', type=str, default='xbox',
                        choices=['xbox', 'ps4'],
                        help='手柄类型')
    
    # ROS 话题参数
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
    
    try:
        # 创建推理器
        inference = TurtleBotDirectInference(
            model_path=args.model,
            turtlebot_model=args.turtlebot_model,
            gpu_id=args.gpu,
            joystick_type=args.joystick,
            control_rate=args.rate,
            image_topic=args.image_topic,
            odom_topic=args.odom_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            joy_topic=args.joy_topic,
            net_structure=args.net_structure
        )
        
        # 加载模型
        # 尝试导入 TurtleBot 网络 (FinalNet)
        try:
            from turtlebot_train.turtlebot_net import FinalNet
            inference.load_model(FinalNet)
        except ImportError:
            print("警告: 无法导入 FinalNet，尝试使用默认网络")
            inference.load_model()
        
        # 设置 ROS 接口
        inference.setup_ros()
        
        # 运行推理
        inference.run(
            duration=args.duration,
            print_interval=args.print_interval
        )
        
        # 打印统计
        inference.print_statistics()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"推理错误: {e}")
        raise
    finally:
        if 'inference' in locals():
            inference.cleanup()


if __name__ == '__main__':
    main()

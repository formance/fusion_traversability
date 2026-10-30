"""Publish the configured static transforms together in a single ROS node."""
import argparse
import math
from pathlib import Path
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


class StaticTransforms(Node):
    def __init__(self, config):
        super().__init__('fusion_static_transforms')
        items = yaml.safe_load(Path(config).read_text())['transforms']
        messages = []
        stamp = self.get_clock().now().to_msg()
        for item in items:
            msg = TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id, msg.child_frame_id = item['parent'], item['child']
            msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, item['xyz'])
            q = quaternion_from_rpy(*map(float, item['rpy']))
            msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
            messages.append(msg)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(messages)
        self.get_logger().info(f'Published {len(messages)} configured static transforms')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = StaticTransforms(args.config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

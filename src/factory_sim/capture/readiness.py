"""Wait for control and sensor discovery before starting the capture node."""
import json,time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
def main():
 rclpy.init();node=Node('capture_readiness');latest=[];node.create_subscription(String,'/robot/step_status',lambda m:latest.append(json.loads(m.data)),1);deadline=time.monotonic()+100
 try:
  while time.monotonic()<deadline:
   rclpy.spin_once(node,timeout_sec=.1)
   if latest and latest[-1].get('state')=='idle' and all(node.count_publishers(t)>0 for t in ['/camera/image_raw','/camera/camera_info','/evaluation/labels_raw','/lidar/points_raw']):print('Control and sensors discovered',flush=True);return
  raise RuntimeError('Control/sensor startup timeout: '+str(latest[-1] if latest else 'no control status'))
 finally:
  node.destroy_node()
  if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()

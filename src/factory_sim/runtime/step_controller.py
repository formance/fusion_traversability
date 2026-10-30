"""Four-key, odometry-closed-loop discrete motion controller."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import yaml
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

ROOT=Path(__file__).resolve().parents[3]
KEYS={'w':'forward','x':'backward','a':'left','d':'right'}

def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))

def clamp(value,limit):
    return max(-limit,min(limit,value))

@dataclass(frozen=True)
class Settings:
    distance_step_m: float=.2
    turn_step_deg: float=15.
    linear_speed_mps: float=.1
    angular_speed_radps: float=.2
    distance_tolerance_m: float=.005
    angle_tolerance_deg: float=.8
    odom_timeout_s: float=1.
    step_timeout_s: float=30.
    settle_time_s: float=.3

    @classmethod
    def load(cls,path):
        data=yaml.safe_load(Path(path).read_text())
        if not isinstance(data,dict):raise ValueError('Control config must be a mapping')
        unknown=set(data)-set(cls.__dataclass_fields__)
        if unknown:raise ValueError('Unknown control settings: '+', '.join(sorted(unknown)))
        settings=cls(**{k:float(v) for k,v in data.items()})
        if any(not math.isfinite(v) or v<=0 for v in asdict(settings).values()):
            raise ValueError('Control parameters must be finite and positive')
        if settings.linear_speed_mps>1.5 or settings.angular_speed_radps>.5:
            raise ValueError('Configured speeds exceed robot limits: 1.5 m/s, 0.5 rad/s')
        if settings.distance_tolerance_m>=settings.distance_step_m/2:
            raise ValueError('Distance tolerance must be less than half the step distance')
        if settings.angle_tolerance_deg>=settings.turn_step_deg/2:
            raise ValueError('Angle tolerance must be less than half the turn step')
        return settings

class StepController(Node):
    def __init__(self,config):
        super().__init__('robot_step_controller')
        self.config_path=Path(config)
        self.settings=Settings.load(self.config_path)
        self.pose=None;self.velocity=(0.,0.);self.last_odom=None;self.last_stamp=None
        self.previous_yaw=None;self.yaw_continuous=0.
        self.goal=None;self.started=0.;self.settled_since=None
        self.completed=0;self.accepted=0;self.last_step=None
        self.state='waiting_for_odometry';self.message=''
        self.publisher=self.create_publisher(Twist,'/cmd_vel',1)
        self.status_publisher=self.create_publisher(String,'/robot/step_status',1)
        self.create_subscription(Odometry,'/odom',self.odometry,qos_profile_sensor_data)
        self.create_subscription(String,'/robot/step_command',self.command,1)
        self.wall_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.05,self.update,clock=self.wall_clock)
        self.create_timer(.2,self.publish_status,clock=self.wall_clock)
        self.get_logger().info('Step control: W forward, X backward, A left, D right. Config: '+str(self.config_path))

    def zero(self):
        self.publisher.publish(Twist())

    def odometry(self,msg):
        if msg.header.frame_id!='odom' or msg.child_frame_id!='base_link':
            self.message='Expected odom -> base_link odometry';return
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        values=[p.x,p.y,q.x,q.y,q.z,q.w,msg.twist.twist.linear.x,msg.twist.twist.angular.z]
        if not all(math.isfinite(v) for v in values):return
        norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
        if norm<1e-12:return
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y)/norm,1-2*(q.y*q.y+q.z*q.z)/norm)
        stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        if self.last_stamp is not None and stamp<self.last_stamp:
            self.abort('Simulation clock reset')
            self.previous_yaw=None
        if self.previous_yaw is None:self.yaw_continuous=yaw
        else:self.yaw_continuous+=wrap(yaw-self.previous_yaw)
        self.previous_yaw=yaw;self.last_stamp=stamp
        self.pose=(float(p.x),float(p.y),self.yaw_continuous)
        self.velocity=(msg.twist.twist.linear.x,msg.twist.twist.angular.z)
        self.last_odom=time.monotonic()
        if self.goal is None and self.state=='waiting_for_odometry':self.state='idle';self.message=''

    def command(self,msg):
        key=msg.data.lower()
        if key not in KEYS:return
        # No queue: repeated keys while moving cannot schedule extra steps.
        if self.goal is not None:return
        try:settings=Settings.load(self.config_path)
        except Exception as error:
            self.message='Configuration error: '+str(error);self.zero();return
        self.settings=settings
        now=time.monotonic()
        if self.pose is None or self.last_odom is None or now-self.last_odom>settings.odom_timeout_s:
            self.message='Waiting for fresh odometry';self.zero();return
        if abs(self.velocity[0])>.02 or abs(self.velocity[1])>.04:
            self.message='Wait until the robot stops';self.zero();return
        self.accepted+=1
        self.goal={'id':self.accepted,'key':key,'direction':KEYS[key],'start':self.pose,
                   'distance':settings.distance_step_m*(1 if key=='w' else -1) if key in ('w','x') else 0.,
                   'angle':math.radians(settings.turn_step_deg)*(1 if key=='a' else -1) if key in ('a','d') else 0.}
        self.started=now;self.settled_since=None;self.state='moving';self.message=''
        self.get_logger().info('Accepted step '+json.dumps(self.goal))
        self.publish_status()

    def errors(self):
        x,y,yaw=self.pose;sx,sy,syaw=self.goal['start']
        distance=(x-sx)*math.cos(syaw)+(y-sy)*math.sin(syaw)
        angle=yaw-syaw
        if self.goal['key'] in ('w','x'):
            return self.goal['distance']-distance,wrap(syaw-yaw),distance,angle
        return 0.,self.goal['angle']-angle,distance,angle

    def abort(self,message):
        if self.goal:
            self.last_step={'id':self.goal['id'],'key':self.goal['key'],'result':'aborted','message':message}
            self.get_logger().warning(message)
        self.goal=None;self.settled_since=None;self.state='waiting_for_odometry';self.message=message
        self.zero()

    def update(self):
        now=time.monotonic()
        if self.last_odom is None or now-self.last_odom>self.settings.odom_timeout_s:
            if self.goal:self.abort('Odometry timeout; stopped')
            else:self.state='waiting_for_odometry';self.zero()
            return
        if self.goal is None:
            self.state='idle';self.zero();return
        if now-self.started>self.settings.step_timeout_s:
            self.abort('Step timeout; stopped (target may be blocked)');return
        distance_error,angle_error,distance,angle=self.errors()
        distance_ok=abs(distance_error)<=self.settings.distance_tolerance_m
        angle_ok=abs(angle_error)<=math.radians(self.settings.angle_tolerance_deg)
        twist=Twist()
        if not distance_ok:twist.linear.x=clamp(distance_error*1.3,self.settings.linear_speed_mps)
        if not angle_ok:twist.angular.z=clamp(angle_error*1.8,self.settings.angular_speed_radps)
        self.publisher.publish(twist)
        stopped=abs(self.velocity[0])<.015 and abs(self.velocity[1])<.025
        if distance_ok and angle_ok and stopped:
            self.state='settling'
            if self.settled_since is None:self.settled_since=now
            if now-self.settled_since>=self.settings.settle_time_s:
                self.completed+=1
                self.last_step={'id':self.goal['id'],'key':self.goal['key'],'result':'completed',
                    'target_distance_m':self.goal['distance'],'actual_distance_m':distance,
                    'target_angle_deg':math.degrees(self.goal['angle']),'actual_angle_deg':math.degrees(angle),
                    'elapsed_s':now-self.started}
                self.get_logger().info('Completed step '+json.dumps(self.last_step))
                self.goal=None;self.state='idle';self.message='';self.zero();self.publish_status()
        else:self.state='moving';self.settled_since=None

    def publish_status(self):
        data={'state':self.state,'accepted':self.accepted,'completed':self.completed,
              'settings':asdict(self.settings),'message':self.message,'last_step':self.last_step}
        if self.goal:
            distance_error,angle_error,_,_=self.errors()
            data.update(active_key=self.goal['key'],remaining_distance_m=distance_error,
                        remaining_angle_deg=math.degrees(angle_error))
        self.status_publisher.publish(String(data=json.dumps(data)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'config/control.yaml'))
    args,ros_args=parser.parse_known_args();rclpy.init(args=ros_args)
    node=StepController(args.config)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.zero();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()

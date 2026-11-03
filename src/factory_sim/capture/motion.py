import json, math, os, re, signal, subprocess, time
from collections import OrderedDict
from pathlib import Path
import cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge

ROOT=Path(__file__).resolve().parents[3]
STATE=Path(os.environ.get('FUSION_STATE_DIR',str(Path.home()/'.local/state/fusion_traversability')))
CONTEXT=json.loads(Path(os.environ['FUSION_CAPTURE_CONTEXT']).read_text())
RUN=Path(CONTEXT['run']); OUT=Path(CONTEXT['output'])
PLAN=json.loads(Path(CONTEXT['route']).read_text())
GEOM=json.loads((RUN/'gt_camera_geometry.json').read_text())
CLASSES=json.loads((ROOT/'src/perception/config/classes.json').read_text())
PALETTE=CLASSES['palette']
OUT.mkdir(parents=True,exist_ok=True)
result={'route':CONTEXT['route'],'mode':'physical driving, Gazebo world pose verified','position_tolerance_m':0.12,'heading_tolerance_deg':3.0,'points':[],'steps':[],'complete':False}
if (OUT/'result.json').exists():
 previous=json.loads((OUT/'result.json').read_text()); result=previous
 result.setdefault('interruptions',[]).append(previous.pop('error','resumed'))
def ns(m):return m.header.stamp.sec*1000000000+m.header.stamp.nanosec
def wrap(v):return math.atan2(math.sin(v),math.cos(v))
def blk(s,start):
 a=s.index('{',start);level=1;b=a+1
 while level:
  level+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[a+1:b-1]
def world():
 s=subprocess.check_output(['gz','topic','-e','-t','/world/factory/dynamic_pose/info','-n','1'],text=True,timeout=12)
 h=blk(s,s.index('header'));t=blk(h,h.index('stamp'))
 def value(b,k):
  m=re.search(r'\b'+k+r':\s*([-+0-9.eE]+)',b);return float(m.group(1)) if m else 0.
 match=re.search(r'pose\s*\{\s*name:\s*"factory_robot"',s)
 if not match:raise RuntimeError('Gazebo robot world pose is missing')
 b=blk(s,match.start());p=blk(b,b.index('position'));q=blk(b,b.index('orientation'));qx,qy,qz,qw=(value(q,k) for k in ('x','y','z','w'))
 return {'timestamp_ns':int(value(t,'sec'))*1000000000+int(value(t,'nsec')),'x':value(p,'x'),'y':value(p,'y'),'z':value(p,'z'),'yaw':math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))}
def configure(distance=1,turn=15):
 text=f'distance_step_m: {max(distance,.05):.6f}\nturn_step_deg: {max(turn,3.):.6f}\nlinear_speed_mps: 1.0\nangular_speed_radps: 0.4\ndistance_tolerance_m: 0.025\nangle_tolerance_deg: 1.0\nodom_timeout_s: 2.0\nstep_timeout_s: 75.0\nsettle_time_s: 0.5\n'
 p=RUN/'control.yaml';tmp=p.with_suffix('.tmp');tmp.write_text(text);tmp.replace(p)
def checkpoint():
 # Detailed movement diagnostics belong to the runtime state directory.
 p=RUN/'capture_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(p)
 saved={k:result[k] for k in ('complete','capture_complete','requested_capture_complete','phase','error','points') if k in result}
 p=OUT/'result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(saved,ensure_ascii=False,indent=2));tmp.replace(p)
class PatrolMotion(Node):
 def camera(self,m):
  if self.mapx is not None:return
  self.info=m;k=np.array(m.k).reshape(3,3);yy,xx=np.indices((m.height,m.width),dtype=np.float32);f=GEOM['gt_focal_px'];self.mapx=(xx-k[0,2])*f/k[0,0]+GEOM['gt_cx'];self.mapy=(yy-k[1,2])*f/k[1,1]+GEOM['gt_cy'];self.valid=(self.mapx>=1)&(self.mapx<GEOM['gt_image_wh'][0]-2)&(self.mapy>=1)&(self.mapy<GEOM['gt_image_wh'][1]-2)
 def check(self):
  if json.loads((STATE/'session.json').read_text()).get('state')!='running':raise RuntimeError('Simulation supervisor has stopped')
 def wait(self,fn,seconds):
  end=time.monotonic()+seconds
  while time.monotonic()<end:
   rclpy.spin_once(self,timeout_sec=.05);self.check()
   if fn():return True
  return False
 def pause(self,seconds):self.wait(lambda:False,seconds)
 def command(self,key,distance=1,turn=15):
  before_pose=world();configure(distance,turn);before=self.status['accepted'];self.pub.publish(String(data=key))
  if not self.wait(lambda:self.status and self.status['accepted']>before,12):raise RuntimeError('Step command not accepted: '+key)
  ident=self.status['accepted']
  if not self.wait(lambda:self.status and self.status.get('last_step') and self.status['last_step']['id']==ident,90):raise RuntimeError('Step timeout '+key)
  step=dict(self.status['last_step']);result['steps'].append(step);checkpoint()
  if step['result']!='completed':raise RuntimeError('Step controller aborted: '+str(step))
  after=world();actual=math.hypot(after['x']-before_pose['x'],after['y']-before_pose['y']);rotation=abs(wrap(after['yaw']-before_pose['yaw']))
  if key in ('w','x') and (actual<max(0,distance-.18) or actual>distance+.22 or rotation>math.radians(8)):raise RuntimeError(f'World motion mismatch / obstruction: commanded {distance:.2f}m, observed {actual:.2f}m, rotation {math.degrees(rotation):.1f}deg')
  if key in ('a','d') and (abs(rotation-math.radians(turn))>math.radians(max(10,turn*.18)) or actual>.32):raise RuntimeError(f'World turn mismatch: commanded {turn:.1f}deg, observed {math.degrees(rotation):.1f}deg; shift {actual:.2f}m')
  return after
 def turn(self,target):
  for i in range(5):
   p=world();delta=wrap(target-p['yaw'])
   if abs(delta)<=math.radians(2):return p
   self.command('a' if delta>0 else 'd',turn=abs(math.degrees(delta)))
  raise RuntimeError('Unable to achieve world heading')
 def move(self,x,y):
  for i in range(80):
   p=world();dx=x-p['x'];dy=y-p['y'];distance=math.hypot(dx,dy)
   if distance<=.10:return p
   p=self.turn(math.atan2(dy,dx));distance=math.hypot(x-p['x'],y-p['y'])
   if distance>.1:self.command('w',distance=min(.8,distance))
  raise RuntimeError('Unable to reach target position')

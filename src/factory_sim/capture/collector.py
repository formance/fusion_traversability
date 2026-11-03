"""Physically drive the route; save aligned images and labels without persisting XYZ."""
from factory_sim.capture.motion import *
import urllib.request, shutil
from perception.io.array_wire import pack_arrays,unpack_arrays
from perception.io.label_io import save_ground_truth
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer,TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
from perception.geometry.transforms import transform_matrix
from perception.geometry.preprocess import effective_ground_evaluation_mask,front_ground_region_mask,near_ground_blind_mask
SENSORS=json.loads((ROOT/'src/factory_sim/config/sensors.json').read_text());CONFIG=json.loads((ROOT/'src/perception/config/inference.json').read_text())
result.update(mode='synchronized capture only; raw point clouds not saved',phase='capture',capture_complete=False)
if result.get('error'):result.setdefault('interruptions',[]).append(result.pop('error'))
class Capture(PatrolMotion):
 def __init__(self):
  Node.__init__(self,'verified_factory_capture');self.bridge=CvBridge();self.rgb=OrderedDict();self.gt=OrderedDict();self.clouds=OrderedDict();self.mapx=None;self.info=None;self.status=None;self.odom=None;self.inference=None;self.geometry_saved=False
  self.buffer=Buffer(cache_time=Duration(seconds=10));self.listener=TransformListener(self.buffer,self)
  q=QoSProfile(depth=3,reliability=ReliabilityPolicy.RELIABLE);sensor=QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT);self.pub=self.create_publisher(String,'/robot/step_command',1)
  self.create_subscription(Image,'/camera/image_raw',lambda m:self.raw_image('rgb',m),sensor);self.create_subscription(Image,'/evaluation/labels_raw',lambda m:self.raw_image('gt',m),sensor);self.create_subscription(PointCloud2,'/lidar/points_raw',self.raw_cloud,sensor);self.create_subscription(CameraInfo,'/camera/camera_info',self.camera,sensor);self.create_subscription(Odometry,'/odom',lambda m:setattr(self,'odom',m),qos_profile_sensor_data);self.create_subscription(String,'/robot/step_status',lambda m:setattr(self,'status',json.loads(m.data)),q)
 def raw_image(self,key,m):
  array=self.bridge.imgmsg_to_cv2(m,desired_encoding='bgr8' if key=='rgb' else 'passthrough')
  if key=='gt' and array.ndim==3:array=array[...,0 if m.encoding=='bgr8' else 2]
  cache=getattr(self,key);cache[ns(m)]=(array.copy(),m.header)
  while len(cache)>2:cache.popitem(last=False)
 def raw_cloud(self,m):
  self.clouds[ns(m)]=m
  while len(self.clouds)>2:self.clouds.popitem(last=False)
 def sample(self,point):
  x,y,yaw=point['x_m'],point['y_m'],math.radians(point['yaw_deg'])
  for _ in range(5):
   self.turn(yaw);p=world()
   if math.hypot(p['x']-x,p['y']-y)<=.105:break
   self.move(x,y)
  self.pause(1.2);a=world();self.pause(.6);b=world()
  in_pose=lambda p:math.hypot(p['x']-x,p['y']-y)<=.12 and abs(wrap(p['yaw']-yaw))<=math.radians(3)
  stable=lambda p,q:math.hypot(p['x']-q['x'],p['y']-q['y'])<.015 and abs(wrap(p['yaw']-q['yaw']))<math.radians(.6)
  if not(in_pose(a) and in_pose(b) and stable(a,b)):raise RuntimeError('Capture world pose is not stable: '+point['id'])
  selected={}
  def ready():
   for t in sorted(set(self.rgb)&set(self.gt)):
    if t<=b['timestamp_ns']+50_000_000 or not self.clouds:continue
    ct=min(self.clouds,key=lambda k:abs(k-t))
    if abs(ct-t)>50_000_000 or ct<b['timestamp_ns']:continue
    selected.update(t=t,ct=ct,rgb=self.rgb[t][0].copy(),header=self.rgb[t][1],gt=self.gt[t][0].copy(),cloud=self.clouds[ct]);return True
   return False
  if not self.wait(ready,35):raise RuntimeError('No exact RGB/GT match with lidar within 50 ms: '+point['id'])
  after=world();t=selected['t'];ct=selected['ct']
  if not(b['timestamp_ns']<=min(t,ct)<=max(t,ct)<=after['timestamp_ns'] and in_pose(after) and stable(b,after)):raise RuntimeError('World pose does not bracket both sensor timestamps')
  header=selected['header'];cloud=selected['cloud'];it=Time.from_msg(header.stamp);lt=Time.from_msg(cloud.header.stamp)
  if not self.wait(lambda:self.buffer.can_transform_full(header.frame_id,it,cloud.header.frame_id,lt,'odom'),3):raise RuntimeError('Time-specific motion compensation TF did not arrive')
  base=self.buffer.lookup_transform('base_link',cloud.header.frame_id,lt,timeout=Duration(seconds=.2)).transform
  camera=self.buffer.lookup_transform_full(header.frame_id,it,cloud.header.frame_id,lt,'odom',timeout=Duration(seconds=.2)).transform
  camera_base=self.buffer.lookup_transform('base_link',header.frame_id,it,timeout=Duration(seconds=.2)).transform
  if not self.geometry_saved:
   evaluation=effective_ground_evaluation_mask(self.info.k,self.info.width,self.info.height,camera_base,SENSORS['lidar'],SENSORS['detection_region']);front=front_ground_region_mask(self.info.k,self.info.width,self.info.height,camera_base,SENSORS['detection_region']);blind=near_ground_blind_mask(self.info.k,self.info.width,self.info.height,camera_base,SENSORS['lidar'],CONFIG['ground_prior']);save_masks(OUT,evaluation,front,blind,self.valid);self.geometry_saved=True
  rgb=selected['rgb'];gt=cv2.remap(selected['gt'],self.mapx,self.mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);xyz=point_cloud2.read_points_numpy(cloud,field_names=('x','y','z'),skip_nans=False).reshape(-1,3).astype(np.float32)
  meta={'request_id':'factory-v2-'+point['id'],'rgb_timestamp_ns':t,'lidar_timestamp_ns':ct,'base_from_lidar':transform_matrix(base),'camera_from_lidar':transform_matrix(camera),'ego_boxes':SENSORS['ego_filter_boxes'],'intrinsic':list(self.info.k),'maximum':CONFIG['lidar_encoding']['max_depth_m'],'dilation':CONFIG['lidar_encoding']['dilation_pixels'],'roi':[0,0,self.info.width,self.info.height],'aligned_depth_range':[.05,120.],'need_lidar':True,'need_aligned':False,'need_points':False}
  meta['need_lidar']=True
  request=urllib.request.Request(CONTEXT['projection_endpoint']+'/infer_points',data=pack_arrays(meta,{'rgb':cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),'xyz':xyz}),headers={'Content-Type':'application/octet-stream'})
  http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with http.open(request,timeout=20) as response: projected,arrays=unpack_arrays(response.read())
  assert projected['request_id']==meta['request_id'] and projected['rgb_timestamp_ns']==t and projected['lidar_timestamp_ns']==ct,'Projection input pairing mismatch'
  folder=OUT/'samples'/('point_'+point['id']);temporary=folder.with_name('.'+folder.name+'.tmp');temporary.mkdir(parents=True,exist_ok=False)
  valid=self.valid&(gt>=1)&(gt<=11);labels=np.full(gt.shape,255,np.uint8);labels[valid]=(gt[valid]-1).astype(np.uint8)
  save_ground_truth(temporary/'ground_truth.png',labels,CLASSES)
  images={'rgb.png':rgb,'lidar_encoding.png':cv2.cvtColor(arrays['lidar'],cv2.COLOR_RGB2BGR)}
  for filename,array in images.items():
   if not cv2.imwrite(str(temporary/filename),array):raise RuntimeError('Could not save '+filename)
  temporary.rename(folder)
  rec={'id':point['id'],'target_pose':{k:point[k] for k in ('x_m','y_m','yaw_deg')},'actual_pose':after,'image_timestamp_ns':t,'lidar_timestamp_ns':ct,'sensor_delta_ms':abs(t-ct)/1e6,'pose_verified':True,'position_error_max_m':max(math.hypot(p['x']-x,p['y']-y) for p in (b,after)),'heading_error_max_deg':max(abs(math.degrees(wrap(p['yaw']-yaw))) for p in (b,after)),'sample_dir':str(folder.relative_to(OUT))}
  result['points'].append(rec);checkpoint();print('CAPTURED POINT',point['id'],flush=True)
def save_masks(output,evaluation,front,blind,valid):
 mask=evaluation & ~blind
 if not cv2.imwrite(str(output/'evaluation_mask.png'),mask.astype(np.uint8)*255):raise RuntimeError('Could not save evaluation mask')
 if not cv2.imwrite(str(output/'front_blind_mask.png'),blind.astype(np.uint8)*255):raise RuntimeError('Could not save front blind mask')

rclpy.init();node=Capture()
try:
 if not node.wait(lambda:node.status and node.status.get('state')=='idle' and node.info and node.rgb and node.gt and node.clouds and node.pub.get_subscription_count()>0,100):raise RuntimeError('Capture sensors not ready: '+str({'status':node.status,'camera_info':bool(node.info),'rgb':len(node.rgb),'gt':len(node.gt),'lidar':len(node.clouds),'control_subscribers':node.pub.get_subscription_count()}))
 for point in PLAN['points'][len(result['points']):CONTEXT.get('limit') or len(PLAN['points'])]:
  print('MOVING TO',point['id'],flush=True);node.move(point['x_m'],point['y_m']);node.sample(point)
 result['capture_complete']=len(result['points'])==len(PLAN['points']);result['requested_capture_complete']=True;result['complete']=result['capture_complete'];result['phase']='capture_complete' if result['complete'] else 'limited_capture_complete'
except KeyboardInterrupt:
 result.update(phase='interrupted',error='Capture interrupted by session shutdown')
 print('CAPTURE STOP: session shutdown; completed samples retained',flush=True)
except Exception as e:result['error']=str(e);print('CAPTURE STOP:',e,flush=True);raise
finally:
 checkpoint();node.destroy_node()
 if rclpy.ok():rclpy.shutdown()

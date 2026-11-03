"""WSL ROS adapter: sensor alignment, HTTP Docker inference, semantic publishers."""
import argparse
import ctypes
from collections import deque
import json
from pathlib import Path
import threading
import time
import urllib.request
import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
RESULT_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from perception.geometry.preprocess import transform_points, near_ground_blind_mask, apply_ground_prior, effective_ground_evaluation_mask, front_ground_region_mask, render_detection_overlay
from perception.io.save_results import save_bundle
from perception.taxonomy import build_lookup, map_labels
from perception.geometry.alignment import common_view_roi
from perception.io.array_wire import pack_arrays, unpack_arrays

from perception.geometry.transforms import transform_matrix

ROOT = Path(__file__).resolve().parents[3]

def nanoseconds(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec

class SemanticNode(Node):
    def __init__(self, args):
        super().__init__('perception', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.config = json.loads((ROOT/'src/perception/config/inference.json').read_text())
        self.args, self.bridge = args, CvBridge()
        self.buffer = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.buffer, self)
        self.pair_queue_size = int(self.config['synchronization'].get('input_pair_queue_size', 4))
        if self.pair_queue_size < 2:
            raise ValueError('input_pair_queue_size must be at least 2')
        self.images, self.clouds = deque(maxlen=self.pair_queue_size), deque(maxlen=self.pair_queue_size)
        self.condition = threading.Condition()
        self.pending, self.info, self.epoch = None, None, 0
        self.last_image_stamp = None
        self.crop_cache = None
        self.ground_mask_cache = None
        self.evaluation_mask_cache = None
        self.front_region_cache = None
        self.sensor_config = json.loads((ROOT/'src/factory_sim/config/sensors.json').read_text())
        self.stop = threading.Event()
        self.stats = {'state':'waiting_for_inputs','encoding':'EXPERIMENTAL projected log-depth TURBO','frames':0,'dropped':0}
        self.meta = json.loads((ROOT/'src/perception/config/classes.json').read_text())
        self.palette = np.asarray(self.meta['palette'], dtype=np.uint8)
        self.mapping = json.loads((ROOT/'src/perception/config/class_mapping.json').read_text())
        self.class_lookup = build_lookup(self.mapping)
        self.stats.update(taxonomy=self.meta['taxonomy'],published_classes=len(self.palette),unavailable_classes=self.meta['unavailable_from_source'])
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        output = self.config['output']
        topics = {'labels':output['labels_topic'], 'semantic':output['color_topic'],
                  'aligned':output['aligned_image_topic']}
        self.output_publishers = {name:self.create_publisher(Image, topic, RESULT_QOS) for name,topic in topics.items()}
        self.status_publisher = self.create_publisher(String, output['status_topic'], 1)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info, SENSOR_QOS)
        self.create_subscription(Image, '/camera/image_raw', self.image, SENSOR_QOS)
        self.create_subscription(PointCloud2, '/lidar/points_raw', self.cloud, SENSOR_QOS)
        self.create_timer(1., self.publish_status)
        self.worker = threading.Thread(target=self.process, name='perception_http_worker', daemon=True)
        self.worker.start()
        self.get_logger().warning('LiDAR encoding is experimental; official raw-point conversion is unpublished. No official accuracy claim.')

    def camera_info(self, message):
        self.info = message

    def image(self, message):
        stamp = nanoseconds(message.header.stamp)
        with self.condition:
            if self.last_image_stamp is not None and stamp < self.last_image_stamp - 500_000_000:
                self.images.clear(); self.clouds.clear(); self.pending = None; self.epoch += 1
            self.last_image_stamp = stamp
            self.images.append(message)
            self.pair()

    def cloud(self, message):
        with self.condition:
            self.clouds.append(message)
            self.pair()

    def pair(self):
        if not self.images or not self.clouds or self.info is None:
            return
        slop = self.config['synchronization']['slop_s'] * 1e9
        for image in sorted(self.images, key=lambda x:nanoseconds(x.header.stamp), reverse=True):
            stamp = nanoseconds(image.header.stamp)
            cloud = min(self.clouds, key=lambda x:abs(nanoseconds(x.header.stamp)-stamp))
            if abs(nanoseconds(cloud.header.stamp)-stamp) <= slop:
                if self.pending is not None:
                    self.stats['dropped'] += 1
                self.pending = (image, cloud, self.info, self.epoch)
                self.images = deque((x for x in self.images if nanoseconds(x.header.stamp)>stamp), maxlen=self.pair_queue_size)
                point_stamp = nanoseconds(cloud.header.stamp)
                self.clouds = deque((x for x in self.clouds if nanoseconds(x.header.stamp)>point_stamp), maxlen=self.pair_queue_size)
                self.condition.notify()
                return

    def publish_status(self):
        with self.condition:
            data = dict(self.stats)
        self.status_publisher.publish(String(data=json.dumps(data)))

    def publish_image(self, name, array, encoding, header):
        if self.output_publishers[name].get_subscription_count() == 0:
            return
        if name in ('semantic','aligned') and array.shape[1] > 960:
            array = cv2.resize(array,(960,round(960*array.shape[0]/array.shape[1])),interpolation=cv2.INTER_AREA)
        message = self.bridge.cv2_to_imgmsg(array, encoding=encoding)
        message.header = header
        self.output_publishers[name].publish(message)

    def process(self):
        next_start = 0.
        while not self.stop.is_set():
            with self.condition:
                self.condition.wait_for(lambda:self.pending is not None or self.stop.is_set(), timeout=.5)
            if self.stop.is_set():
                return
            delay = next_start - time.monotonic()
            if delay > 0 and self.stop.wait(delay):
                return
            with self.condition:
                job, self.pending = self.pending, None
            if job is None:
                continue
            image, cloud, info, epoch = job
            started = time.perf_counter()
            next_start = time.monotonic() + 1. / self.args.max_fps
            try:
                age = (self.get_clock().now().nanoseconds-nanoseconds(image.header.stamp))/1e9
                if age > self.config['synchronization']['max_input_age_s'] or age < -.1:
                    self.stats.update(state='discarded_stale_input', input_age_s=round(age,3))
                    continue
                if info.header.frame_id != image.header.frame_id or not image.header.frame_id or not cloud.header.frame_id:
                    raise ValueError('Image/CameraInfo frame mismatch or missing point-cloud frame')
                if (info.width,info.height)!=(image.width,image.height) or any(abs(x)>1e-8 for x in info.d):
                    raise ValueError('Expected matching zero-distortion RGB and CameraInfo')
                rgb = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
                xyz = point_cloud2.read_points_numpy(cloud,field_names=('x','y','z'),skip_nans=False).reshape(-1,3).astype(np.float32)
                if not len(xyz):
                    raise ValueError('Empty LiDAR scan; RGB-only fallback is disabled')
                cloud_time, image_time = Time.from_msg(cloud.header.stamp), Time.from_msg(image.header.stamp)
                base_tf = self.buffer.lookup_transform('base_link',cloud.header.frame_id,cloud_time,timeout=Duration(seconds=.2)).transform
                if self.config['synchronization']['motion_compensation']:
                    transform = self.buffer.lookup_transform_full(image.header.frame_id,image_time,cloud.header.frame_id,cloud_time,'odom',timeout=Duration(seconds=.2)).transform
                else:
                    transform = self.buffer.lookup_transform(image.header.frame_id,cloud.header.frame_id,cloud_time,timeout=Duration(seconds=.2)).transform
                encoding = self.config['lidar_encoding']
                rigid_tf = self.buffer.lookup_transform(image.header.frame_id,cloud.header.frame_id,image_time,timeout=Duration(seconds=.2)).transform
                settings=self.config['alignment']
                signature=(image.width,image.height,tuple(info.k),tuple(round(v,6) for v in [rigid_tf.translation.x,rigid_tf.translation.y,rigid_tf.translation.z,rigid_tf.rotation.x,rigid_tf.rotation.y,rigid_tf.rotation.z,rigid_tf.rotation.w]))
                if self.crop_cache is None or self.crop_cache[0]!=signature:
                    self.crop_cache=(signature,common_view_roi(info.k,image.width,image.height,rigid_tf,settings,self.sensor_config['lidar']))
                roi=self.crop_cache[1]
                need_aligned = self.args.save_every > 0 or self.output_publishers['aligned'].get_subscription_count() > 0
                need_lidar = self.args.save_every > 0
                request_id = str(nanoseconds(image.header.stamp))
                metadata={'request_id':request_id,'rgb_timestamp_ns':nanoseconds(image.header.stamp),'lidar_timestamp_ns':nanoseconds(cloud.header.stamp),
                    'base_from_lidar':transform_matrix(base_tf),'camera_from_lidar':transform_matrix(transform),
                    'ego_boxes':self.sensor_config['ego_filter_boxes'],'intrinsic':list(info.k),
                    'maximum':encoding['max_depth_m'],'dilation':encoding['dilation_pixels'],'roi':list(roi),
                    'aligned_depth_range':settings['camera_depth_range_m'] if settings.get('enabled',True) else [.05,120.],
                    'need_aligned':need_aligned,'need_points':False,'need_lidar':need_lidar}
                self.stats.update(state='inferencing_in_docker')
                request = urllib.request.Request(self.args.endpoint.rstrip('/')+'/infer_points',
                    data=pack_arrays(metadata,{'rgb':rgb,'xyz':xyz}),headers={'Content-Type':'application/octet-stream'})
                with self.http.open(request,timeout=self.args.timeout) as response:
                    result,arrays = unpack_arrays(response.read())
                if result.get('request_id')!=request_id or result.get('rgb_timestamp_ns')!=nanoseconds(image.header.stamp) or result.get('lidar_timestamp_ns')!=nanoseconds(cloud.header.stamp):
                    raise ValueError('Container returned a mismatched input timestamp')
                labels=arrays['labels'];lidar=arrays.get('lidar');aligned=arrays.get('aligned')
                raw_depth=np.where(arrays['observed']>0,1.,np.inf).astype(np.float32)
                if labels is None or labels.shape!=rgb.shape[:2] or labels.dtype!=np.uint8:
                    raise ValueError('Invalid semantic label map returned by Docker worker')
                labels = map_labels(labels,self.class_lookup)
                model_labels = labels.copy()
                camera_to_base = self.buffer.lookup_transform('base_link',image.header.frame_id,image_time,timeout=Duration(seconds=.2)).transform
                if self.evaluation_mask_cache is None or self.evaluation_mask_cache[0] != signature:
                    self.evaluation_mask_cache=(signature,effective_ground_evaluation_mask(info.k,image.width,image.height,camera_to_base,self.sensor_config['lidar'],self.sensor_config['detection_region']))
                evaluation_mask=self.evaluation_mask_cache[1]
                prior = self.config.get('ground_prior', {})
                if prior.get('enabled', False):
                    if self.ground_mask_cache is None or self.ground_mask_cache[0] != signature:
                        self.ground_mask_cache = (signature,near_ground_blind_mask(info.k,image.width,image.height,camera_to_base,self.sensor_config['lidar'],prior))
                    labels, prior_mask = apply_ground_prior(labels,self.ground_mask_cache[1],raw_depth,prior)
                else:
                    prior_mask = np.zeros(labels.shape, dtype=bool)
                prior_pixels = int(prior_mask.sum())
                prior_changed = int((labels != model_labels).sum())

                if self.stop.is_set() or epoch!=self.epoch:
                    continue
                result_age=(self.get_clock().now().nanoseconds-nanoseconds(image.header.stamp))/1e9
                if result_age>self.config['synchronization']['max_result_age_s'] or result_age<-.1:
                    self.stats.update(state='discarded_stale_result',result_age_s=round(result_age,3))
                    continue
                if self.front_region_cache is None or self.front_region_cache[0] != signature:
                    self.front_region_cache=(signature,front_ground_region_mask(info.k,image.width,image.height,camera_to_base,self.sensor_config['detection_region']))
                front_region=self.front_region_cache[1]
                # Preserve original pixels outside the nominal front ROI.
                color=render_detection_overlay(rgb,model_labels,self.palette,evaluation_mask,prior_mask & front_region)
                x,y,crop_width,crop_height=roi
                rgb,labels,color,model_labels,prior_mask,evaluation_mask=[np.ascontiguousarray(a[y:y+crop_height,x:x+crop_width]) for a in (rgb,labels,color,model_labels,prior_mask,evaluation_mask)]
                if lidar is not None:
                    lidar=np.ascontiguousarray(lidar[y:y+crop_height,x:x+crop_width])
                if aligned is not None:
                    aligned = np.ascontiguousarray(aligned[y:y+crop_height,x:x+crop_width])
                for name,array,mode in [('labels',model_labels,'mono8'),('semantic',color,'rgb8'),('aligned',aligned,'rgb8')]:
                    if array is None or self.output_publishers[name].get_subscription_count() == 0:
                        continue
                    self.publish_image(name,array,mode,image.header)
                # Return released NumPy/serialization buffers to the OS.
                ctypes.CDLL(None).malloc_trim(0)
                self.stats['frames'] += 1
                delta=abs(nanoseconds(image.header.stamp)-nanoseconds(cloud.header.stamp))/1e6
                self.stats.update(state='streaming',model_ms=round(result['model_ms'],1),
                    pipeline_ms=round(1000*(time.perf_counter()-started),1),input_pair_delta_ms=round(delta,2),
                    input_hw=result['input_hw'],output_roi_xywh=list(roi),output_hw=[crop_height,crop_width],aligned_points=result['aligned_points'],projection_device=result['projection_device'],projection_ms=round(result['projection_ms'],1),projected_pixels=result['projected_pixels'],encoded_pixels=result['encoded_pixels'],ground_prior_pixels=prior_pixels,ground_prior_changed_pixels=prior_changed,evaluation_pixels=int(evaluation_mask.sum()),ground_prior_usage='visualization_only',error='')
                if self.args.save_every>0 and self.stats['frames']%self.args.save_every==0:
                    save_bundle(ROOT/'outputs',request_id,rgb,model_labels,color,color,lidar,aligned,evaluation_mask=evaluation_mask)
            except Exception as error:
                self.stats.update(state='waiting_or_error',error=str(error)[:400])
                self.get_logger().warning(str(error)[:400])
                self.stop.wait(1.)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=2.)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--endpoint',default='http://127.0.0.1:8765')
    parser.add_argument('--max-fps',type=float,default=5.)
    parser.add_argument('--timeout',type=float,default=15.)
    parser.add_argument('--save-every',type=int,default=0)
    args,ros_args=parser.parse_known_args()
    if args.max_fps<=0 or args.timeout<=0 or args.save_every<0:
        parser.error('FPS/timeout must be positive and save-every must be nonnegative')
    cv2.setNumThreads(1)
    rclpy.init(args=ros_args)
    node=SemanticNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__':
    main()

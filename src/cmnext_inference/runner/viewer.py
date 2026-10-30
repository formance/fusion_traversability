"""Independent 2x2 viewer for synchronized RGB, aligned LiDAR and semantics."""
import argparse
from collections import deque
import signal
import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
RESULT_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
from .gpu_cloud_view import CloudState, GPUCloudWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QSurfaceFormat
from PyQt5.QtWidgets import QApplication, QLabel, QSizePolicy, QWidget, QGridLayout, QVBoxLayout

ROOT=Path(__file__).resolve().parents[3]

class Viewer(Node):
    def __init__(self,width):
        super().__init__('cmnext_viewer')
        self.bridge,self.width=CvBridge(),width
        camera=json.loads((ROOT/'src/factory_sim/config/sensors.json').read_text())['camera']
        self.source_aspect=camera['height']/camera['width']
        self.frames,self.bundle,self.status={},None,{'state':'waiting_for_inference'}
        self.control={'state':'waiting_for_controller'}
        self.rate_samples={name:deque(maxlen=256) for name in ('rgb','lidar','semantic')}
        self.raw_image=None;self.raw_image_stamp=None;self.raw_cloud_stamp=None
        self.raw_image_received=0.;self.raw_cloud_received=0.
        self.raw_cloud_view=CloudState()
        self.create_subscription(Image,'/camera/image_raw',self.raw_rgb,SENSOR_QOS)
        self.create_subscription(PointCloud2,'/lidar/points_raw',self.raw_points,SENSOR_QOS)
        self.command_publisher=self.create_publisher(String,'/robot/step_command',1)
        self.create_subscription(String,'/robot/step_status',self.control_state,1)
        self.received=0.; self.last_stamp=None
        self.header=Header(frame_id='camera_optical_frame')
        self.publisher=self.create_publisher(Image,'/cmnext/dashboard',1)
        self.create_timer(.2,self.publish_dashboard)
        for name,topic in [('aligned','/cmnext/lidar_aligned'),('semantic','/cmnext/semantic_color')]:
            self.create_subscription(Image,topic,lambda message,key=name:self.image(key,message),RESULT_QOS)
        self.create_subscription(String,'/cmnext/status',self.state,1)

    def output_rate(self,name):
        samples=self.rate_samples[name];now=time.monotonic()
        while samples and samples[0]<now-5:samples.popleft()
        if len(samples)<2:return '0.0' if samples else '-'
        return f'{(len(samples)-1)/(now-samples[0]):.1f}'

    def raw_rgb(self,message):
        self.rate_samples['rgb'].append(time.monotonic())
        image=self.bridge.imgmsg_to_cv2(message,desired_encoding='bgr8')
        self.raw_image=cv2.resize(image,(self.width,max(1,round(self.width*message.height/message.width))),interpolation=cv2.INTER_AREA)
        self.raw_image_stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        self.raw_image_received=time.monotonic()

    def raw_points(self,message):
        self.rate_samples['lidar'].append(time.monotonic())
        points=point_cloud2.read_points_numpy(message,field_names=('x','y','z'),skip_nans=False).reshape(-1,3)
        self.raw_cloud_view.set_points(points,message.width*message.height)
        self.raw_cloud_stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        self.raw_cloud_received=time.monotonic()

    def image(self,name,message):
        stamp=(message.header.stamp.sec,message.header.stamp.nanosec)
        if self.last_stamp is not None and stamp[0]<self.last_stamp[0]-1:
            self.frames.clear()
        self.last_stamp=stamp
        entry=self.frames.setdefault(stamp,{})
        image=self.bridge.imgmsg_to_cv2(message,desired_encoding='bgr8')
        entry[name]=cv2.resize(image,(self.width,max(1,round(self.width*message.height/message.width))),interpolation=cv2.INTER_AREA)
        if len(entry)==2:
            self.rate_samples['semantic'].append(time.monotonic())
            self.bundle=dict(entry); self.header=message.header; self.received=time.monotonic(); self.frames.pop(stamp,None)
        while len(self.frames)>4:
            self.frames.pop(next(iter(self.frames)))

    def state(self,message):
        try: self.status=json.loads(message.data)
        except (ValueError,TypeError): pass

    def control_state(self,message):
        try:self.control=json.loads(message.data)
        except (ValueError,TypeError):pass

    def send_step(self,key):
        self.command_publisher.publish(String(data=key))

    def publish_dashboard(self):
        if self.publisher.get_subscription_count() == 0:
            return
        message=self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.canvas(),cv2.COLOR_BGR2RGB),encoding='rgb8')
        message.header=self.header
        self.publisher.publish(message)

    @property
    def panel_height(self):
        image=getattr(self,'raw_image',None)
        aspect=self.source_aspect if image is None else image.shape[0]/image.shape[1]
        return max(1,round(self.width*aspect))

    def canvas(self):
        tile_width=self.width;tile_height=self.panel_height
        panels=[]
        raw=getattr(self,'raw_image',None)
        blank=lambda:np.zeros((tile_height,tile_width,3),np.uint8)
        raw_panel=blank() if raw is None else cv2.resize(raw,(tile_width,tile_height))
        cloud=getattr(self,'raw_cloud_view',None)
        cloud_panel=blank() if cloud is None else cloud.render(tile_width,tile_height)
        aligned=blank() if self.bundle is None else cv2.resize(self.bundle['aligned'],(tile_width,tile_height))
        semantic=blank() if self.bundle is None else cv2.resize(self.bundle['semantic'],(tile_width,tile_height))
        for panel,title in [(raw_panel,'1. Raw RGB (full frame)'),(cloud_panel,'2. Raw LiDAR 3D (drag / scroll)'),(aligned,'3. Matched LiDAR -> full RGB'),(semantic,'4. Detection ROI semantics | blind: amber')]:
            cv2.rectangle(panel,(0,0),(tile_width,25),(25,25,25),-1)
            cv2.putText(panel,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)
        footer=np.zeros((28,2*tile_width,3),np.uint8)
        stale=self.bundle is not None and time.monotonic()-self.received>3
        state='STALE' if stale else ('waiting' if self.bundle is None else 'live')
        line=f"Output (wall Hz): RGB {self.output_rate('rgb')} | LiDAR {self.output_rate('lidar')} | Semantics {self.output_rate('semantic')} | {state}"
        cv2.putText(footer,line,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,200,255),1,cv2.LINE_AA)
        return np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:]),footer])

class ResultWindow(QWidget):
    KEY_MAP={Qt.Key_W:'w',Qt.Key_A:'a',Qt.Key_D:'d',Qt.Key_X:'x'}

    def __init__(self,node,width):
        super().__init__();self.node=node;self.held=set();self.displayed={};self.next_footer=0.
        self.setWindowTitle('CMNeXt - GPU point cloud / RGB / Semantics')
        self.setFocusPolicy(Qt.StrongFocus)
        layout=QGridLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(2)
        self.panels={}
        for row,column,name,title in [(0,0,'rgb','1. Raw RGB (full frame)'),(1,0,'aligned','3. Matched LiDAR -> full RGB'),(1,1,'semantic','4. Detection ROI semantics | blind: amber')]:
            pane=QWidget();stack=QVBoxLayout(pane);stack.setContentsMargins(0,0,0,0);stack.setSpacing(0);stack.addWidget(QLabel(title));image=QLabel();image.setAlignment(Qt.AlignCenter);image.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored);image.setMinimumSize(200,120);stack.addWidget(image,1);layout.addWidget(pane,row,column);self.panels[name]=image
        pane=QWidget();stack=QVBoxLayout(pane);stack.setContentsMargins(0,0,0,0);stack.setSpacing(0);self.cloud_title=QLabel('2. Raw LiDAR 3D - GPU (drag / scroll)');stack.addWidget(self.cloud_title);self.cloud=GPUCloudWidget(node.raw_cloud_view);container=QWidget.createWindowContainer(self.cloud);container.setFocusPolicy(Qt.NoFocus);container.setMinimumSize(200,120);stack.addWidget(container,1);layout.addWidget(pane,0,1)
        self.footer=QLabel();layout.addWidget(self.footer,2,0,1,2);layout.setRowStretch(0,1);layout.setRowStretch(1,1);layout.setColumnStretch(0,1);layout.setColumnStretch(1,1)
        self.resize(2*width,min(950,2*node.panel_height+84))
        self.cloud.key_press_handler=self.keyPressEvent;self.cloud.key_release_handler=self.keyReleaseEvent

    def keyPressEvent(self,event):
        if event.isAutoRepeat():event.accept();return
        key=event.key()
        if key in (Qt.Key_Q,Qt.Key_Escape):self.close();event.accept();return
        if key in self.KEY_MAP:
            if key not in self.held:self.held.add(key);self.node.send_step(self.KEY_MAP[key])
            event.accept();return
        super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if not event.isAutoRepeat():self.held.discard(event.key())
        event.accept()

    def focusOutEvent(self,event):self.held.clear();super().focusOutEvent(event)

    def show_image(self,name,array):
        if array is None:return
        label=self.panels[name];size=(label.width(),label.height());previous=self.displayed.get(name)
        if previous is not None and previous[0] is array and previous[1]==size:return
        rgb=np.ascontiguousarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB));image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy();label.setPixmap(QPixmap.fromImage(image).scaled(label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation));self.displayed[name]=(array,size)

    def refresh(self):
        for _ in range(2):rclpy.spin_once(self.node,timeout_sec=0)
        self.show_image('rgb',self.node.raw_image)
        if self.node.bundle:
            self.show_image('aligned',self.node.bundle['aligned']);self.show_image('semantic',self.node.bundle['semantic'])
        if self.cloud.uploaded!=self.node.raw_cloud_view.generation:self.cloud.update()
        stale=self.node.bundle is not None and time.monotonic()-self.node.received>3
        state='STALE' if stale else ('waiting' if self.node.bundle is None else 'live')
        if time.monotonic()>=self.next_footer:
            self.footer.setText(f"Output (wall Hz): RGB {self.node.output_rate('rgb')} | LiDAR {self.node.output_rate('lidar')} | Semantics {self.node.output_rate('semantic')} | {state}");self.next_footer=time.monotonic()+.2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--width',type=int,default=512)
    args=parser.parse_args()
    if args.width<400:parser.error('Viewer width must be at least 400')
    cv2.setNumThreads(1)
    rclpy.init();node=Viewer(args.width)
    fmt=QSurfaceFormat();fmt.setVersion(3,3);fmt.setProfile(QSurfaceFormat.CoreProfile);fmt.setDepthBufferSize(24);fmt.setSwapInterval(0);QSurfaceFormat.setDefaultFormat(fmt)
    app=QApplication([]);window=ResultWindow(node,args.width)
    timer=QTimer();timer.timeout.connect(window.refresh);timer.start(50)
    signal.signal(signal.SIGINT,lambda *_:app.quit())
    signal.signal(signal.SIGTERM,lambda *_:app.quit())
    window.show();window.setFocus()
    try:app.exec_()
    finally:
        timer.stop();window.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()

"""Dependency-free orbitable 3D preview of unfiltered sensor-frame XYZ returns."""
import math
import cv2
import numpy as np

class RawCloudView:
    def __init__(self,max_render_points=60000):
        self.points=None;self.colors=None;self.total=0;self.finite=0;self.shown=0
        self.azimuth=math.radians(-55);self.elevation=math.radians(35);self.zoom=1.
        self.radius=None;self.generation=0;self.cache=None;self.max_render_points=max_render_points

    def set_points(self,points,total):
        finite=points[np.isfinite(points).all(axis=1)]
        self.total=int(total);self.finite=len(finite)
        # Decimation is only for drawing; the source PointCloud2 is unchanged.
        stride=max(1,math.ceil(len(finite)/self.max_render_points))
        self.points=np.ascontiguousarray(finite[::stride],dtype=np.float32);self.shown=len(self.points)
        if len(self.points):
            ranges=np.linalg.norm(self.points,axis=1)
            if self.radius is None:self.radius=max(2.,float(np.linalg.norm(finite,axis=1).max())*1.05)
            index=np.rint(255*np.clip(1-np.log1p(ranges)/np.log1p(120.),0,1)).astype(np.uint8)
            self.colors=cv2.applyColorMap(index[:,None],cv2.COLORMAP_TURBO)[:,0,:]
        self.generation+=1

    def rotate(self,dx,dy):
        self.azimuth+=math.radians(dx*.45)
        self.elevation=float(np.clip(self.elevation+math.radians(dy*.4),math.radians(-85),math.radians(85)))

    def scale(self,steps):
        self.zoom=float(np.clip(self.zoom*math.exp(steps*.15),.2,10.))

    def render(self,width,height):
        key=(self.generation,width,height,self.azimuth,self.elevation,self.zoom)
        if self.cache is not None and self.cache[0]==key:return self.cache[1].copy()
        canvas=np.full((height,width,3),18,np.uint8)
        if self.points is None or not len(self.points):
            cv2.putText(canvas,'Waiting for raw LiDAR',(12,height//2),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
            return canvas
        a,e=self.azimuth,self.elevation
        right=np.array([-math.sin(a),math.cos(a),0.],np.float32)
        up=np.array([-math.cos(a)*math.sin(e),-math.sin(a)*math.sin(e),math.cos(e)],np.float32)
        towards=np.array([math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)],np.float32)
        basis=np.stack([right,up,towards],axis=1)
        scale=min(width,height)/(2.15*self.radius)*self.zoom
        def project(points):
            values=points@basis
            uv=np.rint(np.column_stack([width/2+values[:,0]*scale,height/2-values[:,1]*scale])).astype(np.int32)
            return uv,values[:,2]
        uv,depth=project(self.points)
        inside=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=27)&(uv[:,1]<height-18)
        uv,depth,colors=uv[inside],depth[inside],self.colors[inside]
        flat=uv[:,1]*width+uv[:,0]
        zbuffer=np.full(width*height,-np.inf,np.float32);np.maximum.at(zbuffer,flat,depth)
        visible=depth>=zbuffer[flat]-1e-5
        canvas.reshape(-1,3)[flat[visible]]=colors[visible]
        endpoints=np.array([[0,0,0],[5,0,0],[0,5,0],[0,0,5]],np.float32)
        axis,_=project(endpoints)
        for i,(text,color) in enumerate([('X',(40,40,240)),('Y',(40,230,40)),('Z',(230,100,40))],1):
            cv2.line(canvas,tuple(axis[0]),tuple(axis[i]),color,1,cv2.LINE_AA)
            cv2.putText(canvas,text,tuple(axis[i]),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        cv2.putText(canvas,f'finite {self.finite}/{self.total} | drawn {self.shown} | lidar_link',(8,height-5),cv2.FONT_HERSHEY_SIMPLEX,.31,(210,210,210),1,cv2.LINE_AA)
        self.cache=(key,canvas.copy())
        return canvas

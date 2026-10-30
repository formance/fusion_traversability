"""OpenGL point preview. No CUDA/PyOpenGL dependency in the ROS environment."""
import ctypes as C
import math
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer, QMatrix4x4, QImage, QOpenGLWindow

class CloudState:
    def __init__(self,max_points=60000):
        self.points=None;self.generation=0;self.total=0;self.radius=35.;self.max_points=max_points
        self.azimuth=math.radians(-55);self.elevation=math.radians(35);self.zoom=1.;self.widget=None
    def set_points(self,points,total):
        self.total=total;stride=max(1,math.ceil(len(points)/self.max_points))
        self.points=np.ascontiguousarray(points[::stride],dtype=np.float32);self.generation+=1
    def rotate(self,dx,dy):
        self.azimuth+=math.radians(dx*.45);self.elevation=float(np.clip(self.elevation+math.radians(dy*.4),math.radians(-85),math.radians(85)))
    def scale(self,steps):self.zoom=float(np.clip(self.zoom*math.exp(steps*.15),.2,10.))
    def render(self,width,height):
        # Read-back is only used for an explicitly subscribed dashboard.
        if self.widget is None:return np.zeros((height,width,3),np.uint8)
        image=self.widget.grabFramebuffer().convertToFormat(QImage.Format_RGB888)
        if image.isNull():return np.zeros((height,width,3),np.uint8)
        bits=image.bits();bits.setsize(image.byteCount())
        rgb=np.frombuffer(bits,np.uint8).reshape(image.height(),image.bytesPerLine())[:,:image.width()*3].reshape(image.height(),image.width(),3)
        return cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(width,height))

class GPUCloudWidget(QOpenGLWindow):
    def __init__(self,state,parent=None):
        super().__init__(QOpenGLWindow.NoPartialUpdate,parent);self.state=state;state.widget=self;self.uploaded=-1;self.drag=None;self.count=0;self.failed=None
    def initializeGL(self):
        self.gl=C.CDLL('libGL.so.1')
        signatures={'glClearColor':(None,[C.c_float]*4),'glClear':(None,[C.c_uint]),'glEnable':(None,[C.c_uint]),'glDrawArrays':(None,[C.c_uint,C.c_int,C.c_int]),'glViewport':(None,[C.c_int]*4),'glGetString':(C.c_char_p,[C.c_uint]),'glGenTextures':(None,[C.c_int,C.POINTER(C.c_uint)]),'glBindTexture':(None,[C.c_uint,C.c_uint]),'glTexParameteri':(None,[C.c_uint,C.c_uint,C.c_int]),'glPixelStorei':(None,[C.c_uint,C.c_int]),'glTexImage2D':(None,[C.c_uint,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_void_p]),'glActiveTexture':(None,[C.c_uint]),'glGenVertexArrays':(None,[C.c_int,C.POINTER(C.c_uint)]),'glBindVertexArray':(None,[C.c_uint]),'glBufferSubData':(None,[C.c_uint,C.c_ssize_t,C.c_ssize_t,C.c_void_p])}
        for name,(restype,args) in signatures.items():f=getattr(self.gl,name);f.restype=restype;f.argtypes=args
        renderer=self.gl.glGetString(0x1F01).decode();print('POINT_CLOUD_GL_RENDERER:',renderer,flush=True)
        if 'llvmpipe' in renderer.lower() or 'softpipe' in renderer.lower():raise RuntimeError('Point preview requires hardware OpenGL')
        self.program=QOpenGLShaderProgram(self)
        vertex='''#version 330 core
in vec3 position; uniform mat4 mvp; out float colorIndex;
void main(){
 if(any(isnan(position)) || any(isinf(position))){gl_Position=vec4(2,2,2,1);colorIndex=0;return;}
 gl_Position=mvp*vec4(position,1);gl_PointSize=1.5;
 colorIndex=round(255.0*clamp(1.0-log(1.0+length(position))/log(121.0),0.0,1.0));
}'''
        fragment='''#version 330 core
in float colorIndex;uniform sampler2D lut;out vec4 fragColor;
void main(){fragColor=vec4(texture(lut,vec2((colorIndex+0.5)/256.0,0.5)).rgb,1.0);}'''
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex,vertex) or not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment,fragment) or not self.program.link():raise RuntimeError(self.program.log())
        self.vao=C.c_uint();self.gl.glGenVertexArrays(1,C.byref(self.vao));self.gl.glBindVertexArray(self.vao.value)
        self.buffers=[];self.buffer_index=0
        for _ in range(3):
            buffer=QOpenGLBuffer(QOpenGLBuffer.VertexBuffer);buffer.create();buffer.setUsagePattern(QOpenGLBuffer.StreamDraw);buffer.bind();buffer.allocate(self.state.max_points*12);buffer.release();self.buffers.append(buffer)
        self.vbo=self.buffers[0]
        self.texture=C.c_uint();self.gl.glGenTextures(1,C.byref(self.texture));self.gl.glBindTexture(0x0DE1,self.texture.value)
        self.gl.glTexParameteri(0x0DE1,0x2801,0x2600);self.gl.glTexParameteri(0x0DE1,0x2800,0x2600);self.gl.glPixelStorei(0x0CF5,1)
        lut=cv2.cvtColor(cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB).reshape(1,256,3)
        self.gl.glTexImage2D(0x0DE1,0,0x8051,256,1,0,0x1907,0x1401,lut.ctypes.data_as(C.c_void_p))
        self.gl.glEnable(0x0B71);self.gl.glEnable(0x8642);self.gl.glClearColor(18/255,18/255,18/255,1)
    def paintGL(self):
        if not hasattr(self,'vbo'):return
        gl=self.gl;gl.glClear(0x4000|0x0100);state=self.state
        if state.points is None:return
        gl.glBindVertexArray(self.vao.value)
        if self.uploaded!=state.generation:
            self.buffer_index=(self.buffer_index+1)%len(self.buffers);self.vbo=self.buffers[self.buffer_index];self.vbo.bind()
            gl.glBufferSubData(0x8892,0,state.points.nbytes,state.points.ctypes.data_as(C.c_void_p));self.count=len(state.points);self.uploaded=state.generation
        else:self.vbo.bind()
        a,e=state.azimuth,state.elevation;right=np.array([-math.sin(a),math.cos(a),0.]);up=np.array([-math.cos(a)*math.sin(e),-math.sin(a)*math.sin(e),math.cos(e)]);towards=np.array([math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)])
        scale=min(self.width(),self.height())/(2.15*state.radius)*state.zoom
        matrix=np.eye(4);matrix[0,:3]=right*2*scale/max(1,self.width());matrix[1,:3]=up*2*scale/max(1,self.height());matrix[2,:3]=-towards/240
        self.program.bind();self.program.setUniformValue('mvp',QMatrix4x4(*matrix.astype(float).ravel().tolist()));self.program.setUniformValue('lut',0)
        gl.glActiveTexture(0x84C0);gl.glBindTexture(0x0DE1,self.texture.value);self.program.enableAttributeArray('position');self.program.setAttributeBuffer('position',0x1406,0,3,12)
        gl.glDrawArrays(0x0000,0,self.count);self.program.disableAttributeArray('position');self.program.release();self.vbo.release()
    def resizeGL(self,w,h):self.gl.glViewport(0,0,w,h)
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:self.drag=event.pos();event.accept()
    def mouseMoveEvent(self,event):
        if self.drag is not None:
            delta=event.pos()-self.drag;self.state.rotate(delta.x(),delta.y());self.drag=event.pos();self.update();event.accept()
    def mouseReleaseEvent(self,event):self.drag=None
    def wheelEvent(self,event):self.state.scale(event.angleDelta().y()/120);self.update();event.accept()
    def keyPressEvent(self,event):
        handler=getattr(self,'key_press_handler',None)
        if handler is not None:handler(event)
        else:super().keyPressEvent(event)
    def keyReleaseEvent(self,event):
        handler=getattr(self,'key_release_handler',None)
        if handler is not None:handler(event)
        else:super().keyReleaseEvent(event)

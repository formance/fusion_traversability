"""CUDA point projection service for capture; no neural model is loaded."""
from http.server import BaseHTTPRequestHandler,HTTPServer
import json,time
import torch
from perception.io.array_wire import pack_arrays, unpack_arrays, MAX_BODY
from perception.geometry.gpu_projection import GPUProjection

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body=json.dumps({'ready':True,'operation':'lidar_projection_only','device':'cuda'}).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        try:
            if self.path!='/infer_points':raise ValueError('Only point projection is supported')
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=MAX_BODY:raise ValueError('Invalid request size')
            meta,arrays=unpack_arrays(self.rfile.read(length));start=time.perf_counter()
            _,lidar,outputs,stats=self.server.projection.process(meta,arrays)
            body=pack_arrays({'request_id':meta['request_id'],'rgb_timestamp_ns':meta['rgb_timestamp_ns'],'lidar_timestamp_ns':meta['lidar_timestamp_ns'],'projection_ms':1000*(time.perf_counter()-start),**stats},{'lidar':lidar.cpu().numpy()})
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except Exception as error:
            self.send_error(400,str(error))

def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA projection requires a GPU')
    torch.set_num_threads(1);server=HTTPServer(('0.0.0.0',8765),Handler);server.projection=GPUProjection('cuda')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()

"""Local HTTP worker for a GPU Docker container. No ROS dependency."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import time
from PIL import Image
from .model_adapter import ModelAdapter
from .array_wire import pack_arrays, unpack_arrays, MAX_BODY
from .gpu_projection import GPUProjection

MAX_BYTES = 24 * 1024 * 1024

class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/ready':
            self.send_json(200, {'ready': True, 'model': 'CMNeXt-B2', 'modalities': ['img', 'lidar'], 'classes': 25, 'dataset': 'DELIVER',
                'execution': self.server.model.execution})
        else:
            self.send_json(404, {'error': 'Unknown endpoint'})

    def do_POST(self):
        if self.path == '/infer_points':
            return self.infer_points()
        if self.path != '/infer':
            self.send_json(404, {'error': 'Unknown endpoint'})
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size <= 0 or size > MAX_BYTES:
                self.send_json(413, {'error': 'Invalid payload size'})
                return
            payload = json.loads(self.rfile.read(size))
            images = []
            for name in ('rgb_png', 'lidar_png'):
                image = Image.open(io.BytesIO(base64.b64decode(payload[name], validate=True)))
                if image.width * image.height > 8_000_000:
                    raise ValueError('Input image is too large')
                images.append(image.convert('RGB'))
            with self.server.model_lock:
                started = time.perf_counter()
                labels, input_hw = self.server.model.predict(*images)
                elapsed = 1000 * (time.perf_counter() - started)
            output = io.BytesIO()
            Image.fromarray(labels, mode='L').save(output, format='PNG')
            self.send_json(200, {'request_id': payload.get('request_id'),
                'labels_png': base64.b64encode(output.getvalue()).decode(),
                'model_ms': elapsed, 'input_hw': input_hw,
                'encoding_status': 'experimental_projected_depth_not_official_encoding'})
        except (KeyError, ValueError, TypeError) as error:
            self.send_json(400, {'error': str(error)})
        except Exception as error:
            print('Inference request failed:', repr(error), flush=True)
            self.send_json(500, {'error': str(error)})

    def log_message(self, fmt, *args):
        print('%s %s' % (self.log_date_time_string(), fmt % args), flush=True)

    def infer_points(self):
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=MAX_BODY:raise ValueError('Invalid binary payload length')
            metadata,arrays=unpack_arrays(self.rfile.read(size))
            with self.server.model_lock:
                started=time.perf_counter()
                rgb,lidar,outputs,stats=self.server.projection.process(metadata,arrays)
                projected=time.perf_counter()
                labels,input_hw=self.server.model.predict_tensors(rgb,lidar)
                model_done=time.perf_counter()
                response_arrays={name:value.cpu().numpy() for name,value in outputs.items()}
                response_arrays['labels']=labels
            body=pack_arrays({'request_id':metadata['request_id'],'rgb_timestamp_ns':metadata['rgb_timestamp_ns'],
                'lidar_timestamp_ns':metadata['lidar_timestamp_ns'],'input_hw':input_hw,
                'projection_ms':1000*(projected-started),'model_ms':1000*(model_done-projected),**stats},response_arrays)
            self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            self.send_json(400,{'error':str(error)})
        except Exception as error:
            print('GPU projection failed:',repr(error),flush=True);self.send_json(500,{'error':str(error)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/app/models/cmnext/cmnext_b2_deliver_rgbl.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--short-side', type=int, default=376)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    model = ModelAdapter(args.checkpoint, args.device, args.short_side, args.amp)
    server = WorkerServer(('0.0.0.0', args.port), Handler)
    server.model, server.model_lock = model, threading.Lock()
    server.projection = GPUProjection(model.device)
    model.execution['point_projection'] = str(model.device)
    print('CMNeXt RGB-L worker ready; LiDAR encoding is an experimental engineering adapter.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()

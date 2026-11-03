"""Prepare the current factory and collect synchronized pose-verified route samples."""
import argparse,copy,fcntl,hashlib,json,math,os,re,signal,subprocess,time,urllib.request
from pathlib import Path
import xml.etree.ElementTree as E
import yaml
from factory_sim.runtime.session import Session, ROOT, STATE
from factory_sim.runtime.resource_monitor import ResourceMonitor

def prepare(args):
    rate=float(args.sensor_rate)
    if not math.isfinite(rate) or not 1 <= rate <= 10:
        raise ValueError('Capture sensor rate must be finite and in [1, 10] Hz')
    route=Path(args.route).expanduser().resolve();plan=json.loads(route.read_text())
    ids=[p['id'] for p in plan['points']]
    if not ids or len(set(ids))!=len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',str(i)) for i in ids):raise ValueError('Route point IDs must be nonempty and unique')
    for p in plan['points']:
        if not all(math.isfinite(float(p[k])) for k in ('x_m','y_m','yaw_deg')):raise ValueError('Invalid route pose')
    stamp=time.strftime('%Y%m%d_%H%M%S');output=Path(args.output).expanduser().resolve() if args.output else ROOT/'data'/('patrol_'+stamp)
    if output.exists():raise ValueError('Refusing to overwrite an existing dataset: '+str(output))
    output.mkdir(parents=True);run=STATE/'captures'/('patrol_'+stamp);run.mkdir(parents=True,exist_ok=False)
    sensors=json.loads((ROOT/'src/factory_sim/config/sensors.json').read_text());camera=sensors['camera'];height=max(1,camera['height']//2);k=camera['simulation_intrinsics'];width=math.ceil(camera['width']*height/camera['height']*k['fy']/k['fx']);f=width/(2*math.tan(camera['horizontal_fov_rad']/2))
    geometry={'gt_image_wh':[width,height],'gt_focal_px':f,'gt_cx':width/2,'gt_cy':height/2,'method':'Nearest-neighbor ray remap from lower-resolution GT sensor to RGB. Boundary quantization remains.'};(run/'gt_camera_geometry.json').write_text(json.dumps(geometry,indent=2))
    tree=E.parse(ROOT/'src/factory_sim/worlds/factory.sdf');world=tree.getroot().find('world');robot=E.parse(ROOT/'src/factory_sim/models/factory_robot/model.sdf').getroot().find('model');robot=copy.deepcopy(robot);robot.set('name','factory_robot');include=next(i for i in world.findall('include') if i.findtext('name')=='factory_robot');pose=include.findtext('pose');E.SubElement(robot,'pose').text=pose;idx=list(world).index(include);world.remove(include);world.insert(idx,robot)
    link=next(l for l in robot.findall('link') if l.find("sensor[@type='camera']") is not None);gt=copy.deepcopy(link.find("sensor[@type='camera']"));gt.set('name','ground_truth_camera');gt.set('type','segmentation');gt.find('topic').text='/evaluation/semantic';cam=gt.find('camera');cam.find('image/width').text=str(width);cam.find('image/height').text=str(height)
    for tag in ('lens','camera_info_topic'):
        for child in list(cam.findall(tag)):cam.remove(child)
    E.SubElement(cam,'segmentation_type').text='semantic';link.append(gt)
    # Parked sampling needs fresh, synchronized frames, not full-rate rendering
    # throughout every drive and turn. Keep sensor geometry and noise intact.
    for sensor in robot.findall('.//sensor'):
        if sensor.get('type') in ('camera','segmentation','gpu_lidar'):
            sensor.find('update_rate').text=str(rate)
    E.indent(tree,space='  ');tree.write(run/'evaluation.sdf',encoding='unicode',xml_declaration=True)
    bridge=yaml.safe_load((ROOT/'src/factory_sim/config/bridge_camera.yaml').read_text());bridge.append({'ros_topic_name':'/evaluation/labels_raw','gz_topic_name':'/evaluation/semantic/labels_map','ros_type_name':'sensor_msgs/msg/Image','gz_type_name':'gz.msgs.Image','direction':'GZ_TO_ROS','publisher_queue':1,'subscriber_queue':1});(run/'bridge_camera_gt.yaml').write_text(yaml.safe_dump(bridge))
    launch=(ROOT/'src/factory_sim/launch/simulation.launch.py').read_text().replace("ROOT = Path(__file__).resolve().parents[3]",'ROOT = Path('+repr(str(ROOT))+')').replace("world = ROOT / 'src/factory_sim/worlds/factory.sdf'","world = Path(__file__).parent/'evaluation.sdf'")
    launch=launch.replace("params = {'config_file':str(ROOT/f'src/factory_sim/config/bridge_{kind}.yaml'),'use_sim_time':True}","config=Path(__file__).parent/'bridge_camera_gt.yaml' if kind=='camera' else ROOT/f'src/factory_sim/config/bridge_{kind}.yaml'\n        params = {'config_file':str(config),'use_sim_time':True}");(run/'evaluation.launch.py').write_text(launch)
    (run/'control.yaml').write_text('distance_step_m: 1.0\nturn_step_deg: 15.0\nlinear_speed_mps: 1.0\nangular_speed_radps: 0.4\ndistance_tolerance_m: 0.025\nangle_tolerance_deg: 1.0\nodom_timeout_s: 2.0\nstep_timeout_s: 75.0\nsettle_time_s: 0.5\n')
    compose_path=ROOT/'config/docker/compose.inference.yaml'
    compose=yaml.safe_load(compose_path.read_text());service=compose['services']['inference'];service['volumes']=[v for v in service['volumes'] if '/models/' not in v.get('target','')]
    for v in service['volumes']:v['source']=str((compose_path.parent/v['source']).resolve())
    service['command']=['python3','-m','perception.services.projection_capture_server'];service['ports']=['127.0.0.1:8766:8765'];service.pop('build',None);service['mem_limit']='2g';service['memswap_limit']='2g';compose['name']='fusion-capture-projection';compose['services']={'projection':service};(run/'compose.yaml').write_text(yaml.safe_dump(compose))
    config=json.loads((ROOT/'src/perception/config/inference.json').read_text())
    classes=json.loads((ROOT/'src/perception/config/classes.json').read_text())
    mapping=json.loads((ROOT/'src/perception/config/class_mapping.json').read_text())
    metadata={'format_version':5,'ground_truth_storage':'indexed PNG without legend; palette is display-only; read IDs with load_ground_truth or PIL without RGB/L conversion','ground_truth_ids':'0..10; 255 ignored','classes':classes['classes'],'prediction_source_to_target':mapping['source_to_target'],'raw_lidar_saved':False,'image_width':camera['width'],'image_height':camera['height'],'evaluation':'evaluation_mask.png > 0 AND ground_truth.png != 255; blind region excluded','lidar_encoding':config['lidar_encoding'],'model_input':config['model_input'],'pose_tolerance':{'position_m':0.12,'heading_deg':3,'rgb_lidar_delta_ms':50},'gt_alignment':'Lower-resolution semantic sensor nearest-neighbor remapped to RGB; boundaries retain sampling error.'}
    metadata['capture_sensor_rate_hz']=rate
    metadata['front_blind_mask_file']='front_blind_mask.png'
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    context={'run':str(run),'output':str(output),'route':str(route),'limit':args.limit,'projection_endpoint':'http://127.0.0.1:8766'};(run/'context.json').write_text(json.dumps(context,indent=2));return run,output

def record_interruption(output, reason):
    """Keep committed samples and persist the supervisor's actual stop reason."""
    path=output/'result.json'
    data=json.loads(path.read_text()) if path.exists() else {'points':[]}
    if data.get('requested_capture_complete'):return
    data.update(complete=False,capture_complete=False,requested_capture_complete=False,
                phase='interrupted',error=reason or 'Capture interrupted')
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temporary.replace(path)

class CaptureSession(Session):
    def run(self):
        # Bound Gazebo's transport backlog as well as ROS queue depths. These
        # queues contain full-resolution images, so the transport default of
        # 1000 messages is inappropriate for a resource-limited capture host.
        os.environ.setdefault('GZ_TRANSPORT_RCVHWM','8')
        os.environ.setdefault('GZ_TRANSPORT_SNDHWM','8')
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,self.signal_stop)
        self.resources=ResourceMonitor(self.args.resource_config,STATE,self.log_dir);self.resources.start();self.clean_stale_shm('before');self.write_state('starting')
        self.compose=['docker','compose','-f',str(self.capture_run/'compose.yaml')];self.compose_attempted=True
        child=self.start('compose',self.compose+['up','-d','--no-build','--pull','never','projection'])
        if not self.wait_startup(child):return
        end=time.monotonic()+90
        while time.monotonic()<end:
            if not self.tick():return
            try:
                with self.http.open('http://127.0.0.1:8766/ready',timeout=2) as response:
                    d=json.load(response)
                if d.get('operation')=='lidar_projection_only':break
            except (OSError,ValueError):pass
        else:raise RuntimeError('Projection-only GPU service not ready')
        self.start('gazebo',['ros2','launch',str(self.capture_run/'evaluation.launch.py'),'headless:='+str(self.args.headless).lower()]);self.start('step_controller',['/usr/bin/python3','-m','factory_sim.runtime.step_controller','--config',str(self.capture_run/'control.yaml')]);self.write_state('running')
        startup=self.start('readiness',['/usr/bin/python3','-m','factory_sim.capture.readiness'])
        while startup.poll() is None:
            if not self.tick(.2):return
            if self.exited(['gazebo','step_controller']):raise RuntimeError('Simulation exited during readiness')
        if startup.returncode:
            self.tail('readiness');raise RuntimeError('Control or sensors did not become ready')
        if not self.args.headless:
            self.start('gui_roof',['/usr/bin/python3','-m','factory_sim.runtime.gui_roof'])
        context=json.loads((self.capture_run/'context.json').read_text())
        total=len(json.loads(Path(context['route']).read_text())['points'])
        total=min(total,context['limit']) if context.get('limit') else total
        print(f'控制器与传感器已就绪，开始采集 {total} 个点位。',flush=True)
        self.start('capture',['/usr/bin/python3','-m','factory_sim.capture.collector'])
        completed=0
        with (self.log_dir/'capture.log').open() as progress:
            while self.tick():
                for line in progress.readlines():
                    line=line.strip()
                    if line.startswith('MOVING TO '):
                        print(f'[{completed}/{total}] 正在前往采集点 {line.split()[-1]}，到点后停车并校验位姿。',flush=True)
                    elif line.startswith('CAPTURED POINT '):
                        completed+=1
                        saved=json.loads((Path(context['output'])/'result.json').read_text())['points'][-1]
                        print(f'[{completed}/{total}] 点位 {saved["id"]} 已保存 | 位置误差 {saved["position_error_max_m"]:.3f} m | 朝向误差 {saved["heading_error_max_deg"]:.2f}° | 图像/雷达时间差 {saved["sensor_delta_ms"]:.1f} ms',flush=True)
                    elif line.startswith('CAPTURE STOP:'):
                        print('采集停止：'+line.split(':',1)[1].strip(),flush=True)
                ended=self.exited(['capture','gazebo','step_controller'])
                if ended:
                    if ended[0]=='capture' and ended[1]==0:
                        self.reason='capture complete'
                        print(f'采集完成：{completed}/{total} 个点位。',flush=True)
                        return
                    self.tail(ended[0]);raise RuntimeError(str(ended)+'; see capture logs')

def main():
    parser=argparse.ArgumentParser(description='Drive the patrol route and save paired images, labels and LiDAR encoding; no model inference or raw point-cloud files.')
    parser.add_argument('--route',default=str(ROOT/'config/routes/patrol_plan.json'));parser.add_argument('--output');parser.add_argument('--gui',action='store_true');parser.add_argument('--limit',type=int);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resource-config',default=str(ROOT/'config/resources.json'))
    parser.add_argument('--sensor-rate',type=float,default=2.,help='Capture-only RGB, GT and LiDAR rate in Hz (default: 2; range: 1..10); geometry is unchanged')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if not math.isfinite(args.sensor_rate) or not 1 <= args.sensor_rate <= 10:parser.error('--sensor-rate must be finite and in [1, 10] Hz')
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'session.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A simulation session is already running; close it before capture')
        run,output=prepare(args);print('Data directory:',output,flush=True)
        if args.prepare_only:return
        os.environ['FUSION_CAPTURE_CONTEXT']=str(run/'context.json');args.headless=not args.gui;session=CaptureSession(args);session.capture_run=run;session.inference_service="projection"
        try:session.run()
        except Exception as error:session.reason=str(error);raise
        finally:
            try:session.cleanup()
            finally:record_interruption(output,session.reason)
        resultdata=json.loads((output/'result.json').read_text())
        if not resultdata.get('requested_capture_complete'):
            raise SystemExit(f'采集已停止，已保留 {len(resultdata["points"])} 个完整点位。原因：{resultdata["error"]}。数据：{output}')
        print('CAPTURE SAVED:',output,flush=True)

if __name__=='__main__':main()

"""Own the simulation, local ROS workers and project inference container."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import termios
import time
import tty
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(os.environ.get('FUSION_STATE_DIR', str(Path.home()/'.local/state/fusion_traversability')))

class Session:
    def __init__(self, args):
        self.args = args
        self.children = {}
        self.files = []
        self.stop = False
        self.reason = ''
        self.compose_attempted = False
        self.compose = ['docker', 'compose', '-f', str(ROOT/'config/docker/compose.inference.yaml')]
        self.endpoint = 'http://127.0.0.1:' + os.environ.get('CMNEXT_PORT', '8765')
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.terminal = None
        self.terminal_settings = None
        self.resources = None
        self.log_dir = STATE/'logs'/('session_'+time.strftime('%Y%m%d_%H%M%S'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_state(self, state):
        data = {'state':state, 'supervisor_pid':os.getpid(), 'reason':self.reason,
                'logs':str(self.log_dir), 'processes':{name:p.pid for name,p in self.children.items()},
                'endpoint':self.endpoint, 'exit_keys':'Q / Esc / Ctrl+C in launch terminal; Q / Esc in independent viewer'}
        temporary = STATE/'session.json.tmp'
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(STATE/'session.json')

    def signal_stop(self, number, frame):
        self.reason = 'signal '+str(number)
        self.stop = True

    def open_keyboard(self):
        try:
            self.terminal = open('/dev/tty', 'rb', buffering=0)
            self.terminal_settings = termios.tcgetattr(self.terminal.fileno())
            tty.setcbreak(self.terminal.fileno())
        except (OSError, termios.error):
            if self.terminal:
                self.terminal.close()
            self.terminal = None
            print('No interactive terminal: use viewer Q/Esc or send Ctrl+C to stop.', flush=True)

    def tick(self, seconds=.1):
        if self.resources and self.resources.reason:
            self.reason = 'resource_guard: '+self.resources.reason
            self.stop = True
        if self.terminal:
            ready,_,_ = select.select([self.terminal], [], [], seconds)
            if ready:
                value = os.read(self.terminal.fileno(), 1)
                if value in (b'q', b'Q', b'\x1b', b'\x03'):
                    self.reason = 'keyboard exit'
                    self.stop = True
        else:
            time.sleep(seconds)
        return not self.stop

    def start(self, name, command):
        stream = (self.log_dir/(name+'.log')).open('w')
        self.files.append(stream)
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        print(f'Started {name} (PID {child.pid}).', flush=True)
        return child

    def exited(self, names):
        for name in names:
            child = self.children.get(name)
            if child and child.poll() is not None:
                return name, child.returncode
        return None

    def tail(self, name):
        f = self.log_dir/(name+'.log')
        if f.exists():
            print('\n'.join(f.read_text(errors='replace').splitlines()[-18:]), flush=True)

    def wait_startup(self, child):
        deadline = time.monotonic()+90
        while child.poll() is None:
            if not self.tick():
                return False
            if time.monotonic()>deadline:
                raise RuntimeError('Docker startup timed out; see compose.log')
        if child.returncode:
            self.tail('compose')
            raise RuntimeError('Docker Compose startup failed')
        return not self.stop

    def wait_ready(self):
        deadline = time.monotonic()+self.args.startup_timeout
        next_container_check = 0.
        print('Waiting for GPU model and strict checkpoint loading...', flush=True)
        while not self.stop:
            try:
                with self.http.open(self.endpoint+'/ready', timeout=1) as response:
                    if json.load(response).get('ready'):
                        print('Docker StitchFusion model is ready.', flush=True)
                        return True
            except (OSError, ValueError):
                pass
            ended = self.exited(['gazebo','step_controller'])
            if ended:
                self.tail(ended[0])
                raise RuntimeError(f'{ended[0]} exited during startup ({ended[1]})')
            if time.monotonic()>=next_container_check:
                result = subprocess.run(self.compose+['ps','--all','--format','json'], capture_output=True, text=True, timeout=10)
                # Compose versions may emit a JSON object per line or one JSON array.
                records = []
                try:
                    for line in result.stdout.splitlines():
                        entry=json.loads(line)
                        records.extend(entry if isinstance(entry,list) else [entry])
                except ValueError:
                    pass
                if any(x.get('State') in ('exited','dead') for x in records):
                    logs=subprocess.run(self.compose+['logs','--no-color','--tail','25','inference'],capture_output=True,text=True,timeout=10)
                    print(logs.stdout, flush=True)
                    raise RuntimeError('Inference container exited; see Docker logs')
                next_container_check = time.monotonic()+3
            if time.monotonic()>deadline:
                raise RuntimeError('Model readiness timed out; see Docker logs')
            self.tick(.2)
        return False

    def clean_stale_shm(self, phase):
        if os.environ.get('RMW_IMPLEMENTATION')!='rmw_fastrtps_cpp':return
        command=Path('/opt/ros/jazzy/bin/fastdds')
        if not command.exists():return
        try:
            result=subprocess.run([str(command),'shm','clean'],capture_output=True,text=True,timeout=10)
            (self.log_dir/('shm_cleanup_'+phase+'.log')).write_text(result.stdout+result.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            (self.log_dir/('shm_cleanup_'+phase+'.log')).write_text(str(error))

    def run(self):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, self.signal_stop)
        self.open_keyboard()
        from .resource_monitor import ResourceMonitor
        self.resources=ResourceMonitor(self.args.resource_config,STATE,self.log_dir)
        self.resources.start()
        self.clean_stale_shm('before')
        self.write_state('starting')
        print('Q / Esc / Ctrl+C in THIS terminal exits the entire session.', flush=True)
        print('Focus the result window: W/X move one step, A/D turn one step (hold does not repeat).',flush=True)
        print('Q / Esc or closing the independent result window also exits the entire session.', flush=True)
        print('Logs: '+str(self.log_dir), flush=True)
        subprocess.run(['docker','image','inspect','fusion_traversability_stitchfusion:latest'],
                       check=True, stdout=subprocess.DEVNULL, timeout=15)
        self.compose_attempted = True
        compose = self.start('compose', self.compose+['up','-d','--no-build','--pull','never','inference'])
        if not self.wait_startup(compose):
            return
        command = ['ros2','launch',str(ROOT/'src/factory_sim/launch/simulation.launch.py')]
        command += [name+':='+str(getattr(self.args,name)).lower() for name in ('headless','paused')]
        self.start('gazebo', command)
        if not self.args.headless:
            self.start('gui_roof',['/usr/bin/python3','-m','factory_sim.runtime.gui_roof'])
        self.start('step_controller',['/usr/bin/python3','-m','factory_sim.runtime.step_controller','--config',self.args.control_config])
        if not self.wait_ready():
            return
        command = ['/usr/bin/python3','-m','perception.ros.semantic_node',
                   '--endpoint',self.endpoint,'--max-fps',str(self.args.max_fps),
                   '--save-every',str(self.args.save_every)]
        self.start('ros_inference',command)
        names = ['gazebo','ros_inference','step_controller']
        if not self.args.no_viewer:
            self.start('viewer',['/usr/bin/python3','-m','perception.visualization.viewer','--width',str(self.args.viewer_width)])
            names.append('viewer')
        self.write_state('running')
        next_check = 0.
        while self.tick(.2):
            ended = self.exited(names)
            if ended:
                self.reason = f'{ended[0]} exited ({ended[1]})'
                if ended[1]:
                    self.tail(ended[0])
                    raise RuntimeError(self.reason)
                break
            if time.monotonic()>next_check:
                try:
                    with self.http.open(self.endpoint+'/ready',timeout=2) as response:
                        json.load(response)
                except (OSError,ValueError) as error:
                    raise RuntimeError('Model service stopped responding: '+str(error))
                next_check = time.monotonic()+10

    def cleanup(self):
        print('\nStopping session: '+(self.reason or 'shutdown'), flush=True)
        self.write_state('stopping')
        # Signal only process groups created by this supervisor; never pkill other projects.
        groups = [child.pid for child in self.children.values()]
        for signum, duration in ((signal.SIGINT,5),(signal.SIGTERM,3),(signal.SIGKILL,1)):
            active=[]
            for group in groups:
                try:
                    os.killpg(group,signum)
                    active.append(group)
                except ProcessLookupError:
                    pass
            if not active:
                break
            deadline=time.monotonic()+duration
            while time.monotonic()<deadline:
                remaining=[]
                for group in active:
                    try:
                        os.killpg(group,0)
                        remaining.append(group)
                    except ProcessLookupError:
                        pass
                for child in self.children.values():
                    child.poll()  # reap terminated leaders
                if not remaining:
                    break
                active=remaining
                time.sleep(.1)
            groups=active
        for child in self.children.values():
            try: child.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        if self.compose_attempted:
            try:
                logs=subprocess.run(self.compose+['logs','--no-color','--tail','200',getattr(self,'inference_service','inference')],capture_output=True,text=True,timeout=10)
                (self.log_dir/'inference_container.log').write_text(logs.stdout+logs.stderr)
                result=subprocess.run(self.compose+['down','--timeout','5'],capture_output=True,text=True,timeout=30)
                (self.log_dir/'docker_stop.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    print('Docker cleanup failed: '+result.stderr, file=sys.stderr, flush=True)
            except Exception as error:
                print('Docker cleanup error: '+str(error), file=sys.stderr, flush=True)
        self.clean_stale_shm('after')
        if self.resources:
            self.resources.close()
        if self.terminal:
            try: termios.tcsetattr(self.terminal.fileno(),termios.TCSADRAIN,self.terminal_settings)
            finally: self.terminal.close()
        for stream in self.files:
            stream.close()
        self.write_state('stopped')
        print('Session stopped. Logs: '+str(self.log_dir), flush=True)

def main():
    parser=argparse.ArgumentParser(description='Start factory simulation, Docker StitchFusion and independent vertical result viewer; Q/Esc/Ctrl+C exits all.')
    for name in ('headless','paused','no-viewer'):
        parser.add_argument('--'+name,action='store_true')
    parser.add_argument('--resource-config',default=str(ROOT/'config/resources.json'))
    parser.add_argument('--control-config',default=str(ROOT/'config/control.yaml'))
    parser.add_argument('--viewer-width',type=int,default=512)
    parser.add_argument('--max-fps',type=float,default=1)
    parser.add_argument('--save-every',type=int,default=0)
    parser.add_argument('--startup-timeout',type=float,default=180)
    args=parser.parse_args()
    if args.viewer_width<400 or args.max_fps<=0 or args.save_every<0 or args.startup_timeout<=0:
        parser.error('Invalid FPS, save interval or startup timeout')
    from .step_controller import Settings
    Settings.load(args.control_config)
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'session.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('A simulation session is already running. Use its terminal Q/Esc/Ctrl+C to stop.',file=sys.stderr)
            return 2
        session=Session(args)
        code=0
        try:
            session.run()
        except Exception as error:
            session.reason=str(error)
            print('Session failed: '+str(error),file=sys.stderr,flush=True)
            code=1
        finally:
            session.cleanup()
        return code

if __name__=='__main__':
    sys.exit(main())

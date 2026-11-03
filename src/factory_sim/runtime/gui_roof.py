"""Apply the GUI-only transparent view; leave server sensor materials intact."""
import subprocess
import time

def main():
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            info = subprocess.run(['gz','service','-i','-s','/gui/view/transparent'],capture_output=True,text=True,timeout=5)
            if 'gz.msgs.StringMsg' in info.stdout:
                # Let the GUI populate its own render scene before toggling once.
                time.sleep(3)
                result = subprocess.run(['gz','service','-s','/gui/view/transparent',
                    '--reqtype','gz.msgs.StringMsg','--reptype','gz.msgs.Boolean',
                    '--timeout','3000','--req','data: "factory_roof"'],capture_output=True,text=True,timeout=5)
                if result.returncode == 0 and 'data: true' in result.stdout:
                    print('factory_roof: GUI transparent; sensor scene unchanged',flush=True)
                    return
                raise RuntimeError('GUI roof transparency request failed: '+result.stdout+result.stderr)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
    raise RuntimeError('Gazebo GUI transparency service did not become available')

if __name__ == '__main__':
    main()

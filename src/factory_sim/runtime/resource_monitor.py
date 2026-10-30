"""Session-owned resource telemetry and sustained-pressure shutdown policy."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

HOST_SCRIPT = r"""
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
Add-Type -TypeDefinition 'public static class FusionHostInput { public static readonly System.Threading.Tasks.Task<string> Stop = System.Threading.Tasks.Task.Run(() => System.Console.In.ReadLine()); }'
$stopTask=[FusionHostInput]::Stop
while(-not $stopTask.IsCompleted){
 try {
  $m=Get-CimInstance Win32_OperatingSystem
  $c=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'"
  $v=Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue
  @{host_available_gib=$m.FreePhysicalMemory/1MB;host_cpu_percent=[double]$c.PercentProcessorTime;host_wsl_working_set_gib=($v|Measure-Object WorkingSet64 -Sum).Sum/1GB}|ConvertTo-Json -Compress
 } catch { @{host_error=$_.Exception.Message}|ConvertTo-Json -Compress }
 Start-Sleep -Seconds 2
}
"""

class PressurePolicy:
    def __init__(self, config):
        self.config=config
        self.since={}
        for name,rule in config['limits'].items():
            if rule['duration_s']<0 or not any(k in rule for k in ('stop_below','stop_above')):
                raise ValueError('Invalid resource limit: '+name)

    def evaluate(self, values, now):
        warnings=[]
        for name,rule in self.config['limits'].items():
            value=values.get(name)
            if value is None:
                continue
            critical=('stop_below' in rule and value<rule['stop_below']) or ('stop_above' in rule and value>rule['stop_above'])
            if rule.get('require_below'):
                critical=critical and all(values.get(other,float('inf'))<threshold for other,threshold in rule['require_below'].items())
            warning=critical or ('warn_below' in rule and value<rule['warn_below']) or ('warn_above' in rule and value>rule['warn_above'])
            if warning:warnings.append(name)
            if critical:
                self.since.setdefault(name,now)
                if now-self.since[name]>=rule['duration_s']:
                    return warnings,f'{name}={value:.2f}, sustained {now-self.since[name]:.1f}s'
            else:self.since.pop(name,None)
        return warnings,''

class ResourceMonitor:
    def __init__(self, config_path, state_dir, log_dir):
        self.config=json.loads(Path(config_path).read_text())
        self.interval=float(self.config['sample_interval_s'])
        if self.interval<.5 or self.config['host_stale_stop_s']<5:raise ValueError('Invalid resource monitor interval')
        self.policy=PressurePolicy(self.config)
        self.state_dir=Path(state_dir);self.log_dir=Path(log_dir)
        self.host=None;self.host_sample={};self.host_time=0.;self.started=0.
        self.stop_event=threading.Event();self.thread=None;self.reader=None
        self.reason='';self.last_cpu=None;self.gpu_last_good=0.
        self.is_wsl='microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower()

    def start(self):
        self.started=time.monotonic()
        if self.is_wsl:
            executable=shutil.which('powershell.exe')
            if not executable:raise RuntimeError('Windows telemetry requires powershell.exe on WSL PATH')
            encoded=base64.b64encode(HOST_SCRIPT.encode('utf-16-le')).decode('ascii')
            self.host_error=(self.log_dir/'windows_monitor.log').open('w')
            self.host=subprocess.Popen([executable,'-NoProfile','-NonInteractive','-WindowStyle','Hidden','-EncodedCommand',encoded],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.host_error,text=True,encoding='utf-8',errors='replace',start_new_session=True)
            self.reader=threading.Thread(target=self.read_host,daemon=True);self.reader.start()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def read_host(self):
        for line in self.host.stdout:
            try:
                row=json.loads(line.strip().lstrip('\ufeff'))
                if 'host_available_gib' in row:self.host_sample=row;self.host_time=time.monotonic()
            except (ValueError,TypeError):pass

    def sample(self):
        memory={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1);memory[key]=int(value.strip().split()[0])*1024
        c=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));cpu=(sum(c[:8]),c[3]+c[4])
        load=0. if self.last_cpu is None else 100*(1-(cpu[1]-self.last_cpu[1])/max(1,cpu[0]-self.last_cpu[0]));self.last_cpu=cpu
        values={'epoch':time.time(),'wsl_cpu_percent':round(load,2),'wsl_available_gib':memory['MemAvailable']/2**30,'swap_used_gib':(memory['SwapTotal']-memory['SwapFree'])/2**30,'wsl_total_gib':memory['MemTotal']/2**30}
        if self.is_wsl:
            values.update(self.host_sample)
            values['host_sample_age_s']=time.monotonic()-(self.host_time or self.started)
        try:
            query=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits','--id',os.environ.get('GPU_DEVICE_ID','0')],capture_output=True,text=True,timeout=2)
            a=list(map(float,query.stdout.strip().split(',')))
            values.update(gpu_util_percent=a[0],gpu_used_gib=a[1]/1024,gpu_total_gib=a[2]/1024,gpu_free_gib=(a[2]-a[1])/1024,gpu_temperature_c=a[3]);self.gpu_last_good=time.monotonic()
        except (OSError,ValueError,subprocess.TimeoutExpired):values['gpu_sample_missing']=True
        return values

    def save(self, row):
        temporary=self.state_dir/'resources.json.tmp'
        temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(self.state_dir/'resources.json')
        with (self.log_dir/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def run(self):
        while not self.stop_event.is_set():
            try:
                now=time.monotonic();row=self.sample();warnings,reason=self.policy.evaluate(row,now)
                if self.is_wsl and row['host_sample_age_s']>self.config['host_stale_stop_s']:reason='Windows resource telemetry unavailable/stale'
                if now-(self.gpu_last_good or self.started)>self.config['host_stale_stop_s']:reason='GPU resource telemetry unavailable/stale'
                if self.is_wsl and row.get('host_available_gib',100)<2:reason='Windows available memory below emergency 2 GiB threshold'
                if row['wsl_available_gib']<.35:reason='WSL available memory below emergency 0.35 GiB threshold'
                self.reason=reason
                row.update(state='stopping' if reason else ('warning' if warnings else 'ok'),warnings=warnings,reason=reason)
                self.save(row)
                if reason:
                    print('RESOURCE GUARD: '+reason,flush=True);return
            except Exception as error:
                self.reason='Resource monitor failed: '+str(error)
                print(self.reason,flush=True);return
            self.stop_event.wait(self.interval)

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=4)
        if self.host:
            try:
                self.host.stdin.close();self.host.wait(timeout=6)
            except (OSError,subprocess.TimeoutExpired):self.host.terminate()
            self.host.stdout.close();self.host_error.close()
        try:
            f=self.state_dir/'resources.json'
            row=json.loads(f.read_text());row['state']='stopped';row['monitor_stopped_epoch']=time.time();self.save(row)
        except (OSError,ValueError):pass

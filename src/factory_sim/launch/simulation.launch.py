"""Local Harmonic factory launcher. This file is not executed during setup."""
import os
import shlex
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[3]

def configure(context):
    def flag(name):
        return LaunchConfiguration(name).perform(context).lower() == 'true'
    world = ROOT / 'src/factory_sim/worlds/factory.sdf'
    gz_args = ['-v', '3']
    if not flag('paused'):
        gz_args.append('-r')
    is_wsl = 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower()
    graphics_env = []
    if is_wsl:
        graphics_env = [
            SetEnvironmentVariable('GALLIUM_DRIVER', os.environ.get('GALLIUM_DRIVER', 'd3d12')),
            SetEnvironmentVariable('MESA_D3D12_DEFAULT_ADAPTER_NAME', os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA'))]
    if flag('headless'):
        gz_args.append('-s')
        if is_wsl:
            if not os.environ.get('DISPLAY'):
                raise RuntimeError('WSL server rendering requires an active WSLg DISPLAY for D3D12 acceleration')
            # Server only: no Gazebo GUI; sensors use WSLg GLX/D3D12 instead of software EGL.
        else:
            gz_args.append('--headless-rendering')
    gz_args.append(str(world))
    actions = graphics_env + [SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
        str(ROOT/'src/factory_sim/models') + os.pathsep + os.environ.get('GZ_SIM_RESOURCE_PATH', '')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(
            get_package_share_directory('ros_gz_sim'))/'launch/gz_sim.launch.py')),
            launch_arguments={'gz_args': shlex.join(gz_args), 'on_exit_shutdown': 'true'}.items())]
    for kind, frame in [('system', None), ('camera', 'camera_optical_frame'), ('lidar', 'lidar_link')]:
        params = {'config_file':str(ROOT/f'src/factory_sim/config/bridge_{kind}.yaml'),'use_sim_time':True}
        if frame:
            params['override_frame_id'] = frame
        actions.append(Node(package='ros_gz_bridge',executable='parameter_bridge',
            name=f'fusion_{kind}_bridge',parameters=[params],output='screen'))
    actions.append(ExecuteProcess(cmd=['/usr/bin/python3','-m','factory_sim.runtime.static_transforms',
        '--config',str(ROOT/'src/factory_sim/config/extrinsics.yaml')],output='screen'))
    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless',default_value='false'),
        DeclareLaunchArgument('paused',default_value='false'),
        OpaqueFunction(function=configure)])

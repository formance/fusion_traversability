"""Check a captured patrol, then evaluate StitchFusion without starting ROS/Gazebo."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from .dataset import CaptureDataset
from .metrics import confusion_matrix, metrics
from .report import BLIND_FILL_RGB, legend_font_path, save_segmentation, write_report
from .regions import REGIONS, REGION_NAMES, front_blind_mask

ROOT = Path(__file__).resolve().parents[3]

RESULT_ARTIFACTS = ('semantic', 'comparisons',
                    'per_sample.csv', 'per_class.csv', 'confusion_matrix_lidar_valid.csv',
                    'confusion_matrix_evaluation_mask.csv', 'report.html', 'summary.json')


def publish_results(workspace, destination):
    """Publish only completed results; keep logs and temporary files outside the project."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in RESULT_ARTIFACTS:
        source = workspace/name
        if source.exists():
            shutil.move(str(source), str(destination/name))
    print(f'验证结果目录: {destination}', flush=True)
    if (destination/'report.html').is_file():
        print(f'评测报告: {destination/"report.html"}', flush=True)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(args, dataset, output, report):
    # Import torch only for inference; --check-only needs just NumPy and Pillow.
    import torch
    from perception.models.model_adapter import ModelAdapter

    config = dataset.metadata['model_input']
    expected = {'resize_mode': 'two_stage_bilinear_uint8_align_multiple_32',
                'align_to_multiple': 32, 'keep_aspect_ratio': True,
                'rgb_normalization': 'ImageNet', 'lidar_normalization': 'divide_by_255_only',
                'precision': 'float32'}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Dataset preprocessing differs from supported FP32 ModelAdapter settings')
    short_side = config['short_side']
    if type(short_side) is not int or not 128 <= short_side <= 1024:
        raise ValueError('Unsupported model short side (expected 128..1024)')
    checkpoint = Path(args.checkpoint).resolve()
    torch.set_num_threads(1)
    model = ModelAdapter(checkpoint, device='cuda', short_side=short_side, amp=False, model_kind='stitchfusion')
    torch.cuda.reset_peak_memory_stats()
    report['model'] = {'name': model.model_name, 'checkpoint_sha256': digest(checkpoint),
                       'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                       'execution': model.execution, 'model_input': config,
                       'source_to_target': dataset.mapping.tolist()}
    matrices = {name: np.zeros((11, 11), dtype=np.int64) for name in REGIONS}
    per_sample = []
    palette = json.loads((ROOT/'src/perception/config/classes.json').read_text())['palette']
    report['palette'] = palette
    legend_font_path()
    report['prediction_display'] = {'image_area_wh': [dataset.width, dataset.height],
                                     'legend_appended_below': True, 'blind_fill_rgb': list(BLIND_FILL_RGB),
                                     'legend_labels': 'class_names_only',
                                     'ground_truth_display': 'full_captured_labels_without_blind_fill; unlabelled 255 is white',
                                     'regions': {'lidar_valid': 'Dense model prediction inside continuous per-column encoded-LiDAR envelope; gray outside. Scoring still uses nonblack encoded support and valid GT.',
                                                 'evaluation_mask': 'Original RGB outside mask; gray only in front blind region'},
                                     'saved_images': ['semantic', 'comparisons']}
    started = time.perf_counter()
    for index, (ident, folder) in enumerate(dataset.samples, 1):
        images, truth, regions = dataset.read(folder)
        before = time.perf_counter()
        source, input_hw = model.predict(*images)
        model_ms = 1000*(time.perf_counter()-before)
        if source.shape != truth.shape or source.dtype != np.uint8 or np.any(source >= len(dataset.mapping)):
            raise ValueError(f'{ident}: invalid native model prediction')
        prediction = dataset.mapping[source]
        samples = {}
        for name, valid in regions.items():
            current = confusion_matrix(truth, prediction, valid)
            matrices[name] += current
            sample = samples[name] = metrics(current, dataset.classes)
            per_sample.append({'id': ident, 'region': name, 'valid_pixels': sample['valid_pixels'],
                               'pixel_accuracy': sample['pixel_accuracy'],
                               'model_ms': model_ms, 'input_height': input_hw[0], 'input_width': input_hw[1]})
        support = np.any(np.asarray(images[1]) != dataset.lidar_missing_rgb, axis=2)
        save_segmentation(output, ident, images[0], truth, prediction, palette, dataset.classes, regions, dataset.front_blind,
                          lidar_support=support)
        report.update(completed_samples=index, metrics={name: metrics(matrix, dataset.classes) for name, matrix in matrices.items()},
                      elapsed_s=time.perf_counter()-started)
        write_json(output/'summary.json', report)
        scores = ' · '.join(f'{REGION_NAMES[name]}有效像素={samples[name]["valid_pixels"]}' for name in REGIONS)
        print(f'[{index}/{len(dataset.samples)}] {ident}: {scores}, model={model_ms:.1f} ms', flush=True)
        # Only one pair is retained; no full-dataset tensors or image caches.
    write_csv(output/'per_sample.csv', per_sample)
    write_csv(output/'per_class.csv', [{'region': name, **row} for name in REGIONS for row in report['metrics'][name]['per_class']])
    report['confusion_matrices'] = {}
    for name, matrix in matrices.items():
        active = report['metrics'][name]['class_ids']
        with (output/f'confusion_matrix_{name}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ground_truth / prediction', *(dataset.classes[i] for i in active)])
            for i in active:
                writer.writerow([dataset.classes[i], *map(int, matrix[i, active])])
        report['confusion_matrices'][name] = {'class_ids': active, 'matrix': matrix[np.ix_(active, active)].tolist()}
    timings = [row['model_ms'] for row in per_sample if row['region'] == REGIONS[0]]
    report['model_latency_ms'] = {'mean': float(np.mean(timings)), 'median': float(np.median(timings)),
                                   'first_sample': timings[0], 'includes_first_sample_warmup': True}
    report['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    report['status'] = 'completed'
    write_report(output, report, per_sample, dataset.classes, palette)
    write_json(output/'summary.json', report)
    for name in REGIONS:
        print(f'{REGION_NAMES[name]}: 有效像素={report["metrics"][name]["valid_pixels"]}, '
              f'像素一致率（参考）={report["metrics"][name]["pixel_accuracy"]:.2%}; 流程完成', flush=True)
    return 0


def docker_evaluate(args, output):
    import yaml
    from factory_sim.runtime.resource_monitor import ResourceMonitor

    runtime = output/'runtime'
    runtime.mkdir(exist_ok=True)
    monitor = ResourceMonitor(args.resource_config, runtime, runtime)
    name = f'fusion-offline-validation-{os.getpid()}'
    source = ROOT/'config/docker/compose.inference.yaml'
    if not source.is_file():
        raise ValueError('Docker inference Compose configuration not found')
    specification = yaml.safe_load(source.read_text())
    service = specification['services']['inference']
    # Resolve project mounts independently of where the deployment file lives.
    # Keep generated configuration outside the checkout and use existing images.
    mounts = {'/app/src/perception': ROOT/'src/perception',
              '/opt/stitchfusion': ROOT/'src/third_party/stitchfusion',
              '/app/models/stitchfusion': ROOT/'models/stitchfusion'}
    for volume in service.get('volumes', []):
        if isinstance(volume, dict) and volume.get('type') == 'bind':
            volume['source'] = str(mounts.get(volume['target'], (source.parent/volume['source']).resolve()))
    service.pop('build', None)
    specification['services'] = {'inference': service}
    generated = runtime/'compose.yaml'
    generated.write_text(yaml.safe_dump(specification))
    compose = ['docker', 'compose', '--project-name', name, '-f', str(generated)]
    command = compose + ['run', '--rm', '--no-deps', '--pull', 'never', '--no-tty', '--interactive=false', '--name', name,
               '--user', f'{os.getuid()}:{os.getgid()}',
               '--env', 'PYTORCH_KERNEL_CACHE_PATH=/tmp',
               '--env', 'HOME=/tmp', '--env', 'XDG_CACHE_HOME=/tmp/.cache',
               '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--volume', f'{Path(args.dataset).resolve()}:/offline-data:ro',
               '--volume', f'{output}:/offline-output',
               '--volume', f'{Path(args.checkpoint).resolve()}:/offline-checkpoint.pth:ro',
               '--volume', f'{legend_font_path()}:/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:ro',
               'inference', 'python3', '-m', 'perception.evaluation.offline',
               '/offline-data', '--output', '/offline-output', '--checkpoint', '/offline-checkpoint.pth', '--worker',
               '--front-blind-mask', '/offline-output/runtime/front_blind_mask.png']
    for key, enabled in (('--allow-partial', args.allow_partial),):
        if enabled:
            command.append(key)
    child = None
    try:
        monitor.start()
        image = json.loads((ROOT/'src/perception/config/inference.json').read_text())['container_image']
        subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL, timeout=15)
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        while child.poll() is None:
            if monitor.reason:
                raise RuntimeError('resource_guard: '+monitor.reason)
            time.sleep(.2)
        return child.returncode
    finally:
        try:
            if child is not None:
                try:
                    subprocess.run(compose+['down', '--timeout', '5'], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=20, check=False)
                except (OSError, subprocess.TimeoutExpired) as error:
                    print('Container cleanup failed: '+str(error), file=sys.stderr)
            if child is not None and child.poll() is None:
                # Only the disposable container and process group created here.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        finally:
            monitor.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('dataset', help='Capture directory containing metadata.json and result.json')
    result.add_argument('--output', help='New validation result directory (default: outputs/offline_TIMESTAMP_PID); runtime files stay outside the project')
    result.add_argument('--checkpoint', default=str(ROOT/'models/stitchfusion/deliver_rgbl.pth'))
    result.add_argument('--check-only', action='store_true', help='Validate saved data without Docker or model inference')
    result.add_argument('--allow-partial', action='store_true', help='Explicitly evaluate committed samples from an interrupted patrol')
    result.add_argument('--front-blind-mask', help='Optional native-size binary front blind mask; otherwise use saved capture mask or recorded/current sensor geometry')
    result.add_argument('--local', action='store_true', help='Use CUDA in current Python environment instead of Docker')
    result.add_argument('--resource-config', default=str(ROOT/'config/resources.json'))
    result.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    output = None
    destination = None
    owns_output = False
    report = {}
    try:
        if not args.worker and os.geteuid() == 0:
            raise ValueError('Run val.sh as a regular user to preserve file ownership')
        dataset = CaptureDataset(args.dataset, args.allow_partial)
        checks = dataset.check()
        if not args.check_only and not Path(args.checkpoint).is_file():
            raise ValueError('Checkpoint not found: '+args.checkpoint)
        if args.worker:
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
        else:
            stamp = time.strftime(f'offline_%Y%m%d_%H%M%S_{os.getpid()}')
            destination = Path(args.output).resolve() if args.output else (ROOT/'outputs'/stamp).resolve()
            if destination.is_relative_to(ROOT) and not destination.is_relative_to((ROOT/'outputs').resolve()):
                raise ValueError('Project result paths must be under outputs/; source and data directories are protected')
            if destination.exists():
                raise FileExistsError('Refusing to overwrite existing results: '+str(destination))
            state = Path(os.environ.get('FUSION_STATE_DIR', str(Path(os.environ.get('XDG_STATE_HOME', str(Path.home()/'.local/state')))/'fusion_traversability'))).resolve()
            if state.is_relative_to(ROOT):
                raise ValueError('FUSION_STATE_DIR/XDG_STATE_HOME must keep process files outside the project')
            output = state/'validation_runtime'/stamp
            output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        report = {'report_version': 5, 'status': 'data_checked' if args.check_only else 'running', 'dataset': str(dataset.directory),
                  'assessment_mode': 'public_pretrained_model_observation',
                  'training_context': 'Public pretrained model, without scene-specific fine-tuning for this factory simulation. Diagnostic observations only; no accuracy acceptance thresholds.',
                  'data_checks': checks, 'completed_samples': 0,
                  'metadata_sha256': digest(dataset.directory/'metadata.json'),
                  'capture_record_sha256': digest(dataset.directory/'result.json'),
                  'metric_definition': 'Two independent regions: encoded LiDAR support AND GT != 255; stored evaluation mask AND GT != 255. Encoded support includes capture-time dilation. Pixel agreement and per-class diagnostics are descriptive references for pretrained-model inspection, not acceptance scores. Classes absent from both GT and prediction are omitted.',
                  'limitations': 'Simulation engineering 11-class evaluation; approximate 25-to-11 mapping, low-resolution GT remap and experimental LiDAR encoding; no visualization ground prior applied.'}
        if not args.check_only:
            dataset.front_blind, provenance = front_blind_mask(dataset, ROOT, args.front_blind_mask)
            if (dataset.front_blind & dataset.mask).any():
                raise ValueError('Front blind mask overlaps evaluation_mask.png; provide matching capture geometry/mask')
            if not args.worker:
                runtime = output/'runtime'
                runtime.mkdir()
                prepared = runtime/'front_blind_mask.png'
                Image.fromarray(dataset.front_blind.astype(np.uint8)*255).save(prepared)
                provenance['prepared_mask_sha256'] = digest(prepared)
                report['runtime_directory'] = str(runtime)
            report['front_blind_region'] = provenance
        write_json(output/'summary.json', report)
        counts = ', '.join(f'{name}={checks["regions"][name]["valid_pixels"]}' for name in REGIONS)
        print(f'Data checks passed: {checks["samples"]} samples, valid pixels: {counts}.', flush=True)
        if destination is not None:
            print(f'验证结果将保存到: {destination}', flush=True)
        if args.check_only:
            if destination is not None:
                publish_results(output, destination)
            return 0
        if args.local or args.worker:
            code = evaluate(args, dataset, output, report)
            if destination is not None:
                publish_results(output, destination)
            return code
        code = docker_evaluate(args, output)
        finished = json.loads((output/'summary.json').read_text())
        if code == 0 and finished.get('status') != 'completed':
            raise RuntimeError('Worker exited without a completed report')
        if code != 0:
            raise RuntimeError(f'Offline Docker worker exited with code {code}')
        # The worker sees bind-mount paths; record the original host paths.
        finished['dataset'] = str(dataset.directory)
        finished['checkpoint'] = str(Path(args.checkpoint).resolve())
        finished['runtime_directory'] = str(output/'runtime')
        finished['front_blind_region'] = report['front_blind_region']
        with (output/'per_sample.csv').open(encoding='utf-8-sig', newline='') as stream:
            samples = list(csv.DictReader(stream))
        write_report(output, finished, samples, dataset.classes, finished['palette'])
        write_json(output/'summary.json', finished)
        publish_results(output, destination)
        return code
    except (Exception, KeyboardInterrupt) as error:
        message = str(error) or 'Interrupted by user'
        if owns_output:
            path = output/'summary.json'
            if path.exists():
                report = json.loads(path.read_text())
            report.update(status='failed', error=message)
            write_json(path, report)
        print('VALIDATION FAILED: '+message, file=sys.stderr, flush=True)
        if owns_output:
            print(f'失败记录: {output/"summary.json"}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

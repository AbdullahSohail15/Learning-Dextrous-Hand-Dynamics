"""Select exactly one visible CUDA device before importing Isaac or PyTorch."""
import os
import sys


def restrict_simulator_gpu():
    # Preserve an explicit physical-device selection, otherwise derive it from
    # --device. Isaac/Warp may create a context on every visible CUDA device.
    device='cuda:0'
    index=None
    for i,arg in enumerate(sys.argv):
        if arg=='--device' and i+1<len(sys.argv):
            device=sys.argv[i+1];index=i+1
        elif arg.startswith('--device='):
            device=arg.split('=',1)[1];index=i
    selected=os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected is None:
        selected=device.split(':')[1] if device.startswith('cuda:') else '0'
        os.environ['CUDA_VISIBLE_DEVICES']=selected
        if index is not None and device.startswith('cuda:'):
            sys.argv[index]='--device=cuda:0' if sys.argv[index].startswith('--device=') else 'cuda:0'
    elif not selected or ',' in selected:
        raise RuntimeError('Simulator inference requires exactly one visible CUDA device')
    elif device.startswith('cuda:') and device!='cuda:0':
        raise RuntimeError('With CUDA_VISIBLE_DEVICES set, use logical --device cuda:0')
    return selected


def budget_cpus():
    """The logical CPUs granted to experiments on this machine (cpu_limits.json); without that file, the CPUs this process may use."""
    import json
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / 'cpu_limits.json').exists():
        return sorted(os.sched_getaffinity(0))
    return sorted(json.loads((Path(__file__).resolve().parents[1] / 'cpu_limits.json').read_text())['logical_cpus'])

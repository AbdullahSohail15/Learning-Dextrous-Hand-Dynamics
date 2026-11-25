"""Optional CPU budget, enforced before numerical libraries are imported.

If cpu_limits.json (repo: SHADOW/cpu_limits.json, see cpu_limits.example.json) is absent, this wrapper just runs the program.
If it is present, every job is pinned to the same two logical CPUs listed there and library threads are set to 1:

Every job uses the SAME mask, so the budget is aggregate, not per process.
seccomp prevents native libraries or descendants from widening the affinity mask.
This does not depend on a delegated cgroup CPU controller (absent on this host).
"""
import ctypes
import errno
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def apply_cpu_budget():
    policy = json.loads((ROOT / 'cpu_limits.json').read_text())
    cpus = set(policy['logical_cpus'])
    if len(cpus) != 2 or policy['maximum_combined_cpu_percent'] != 200:
        raise RuntimeError('Expected the user-authorized shared 200% CPU budget')
    for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'MKL_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'GOTO_NUM_THREADS', 'BLIS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'RAYON_NUM_THREADS', 'TBB_NUM_THREADS', 'OPENCV_FOR_THREADS_NUM',
                 'PXR_WORK_THREAD_LIMIT',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[name] = '1'
    os.environ.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
                      KMP_AFFINITY='disabled', TOKENIZERS_PARALLELISM='false',
                      SURGICAL_CPU_BUDGET='200', SURGICAL_MAX_PARALLEL_JOBS='1')
    # Cover any threads already created by the launcher. New threads inherit this.
    for thread in Path('/proc/self/task').iterdir():
        tid = int(thread.name)
        try:
            if os.sched_getaffinity(tid) != cpus:
                os.sched_setaffinity(tid, cpus)
            if os.sched_getaffinity(tid) != cpus:
                raise RuntimeError('CPU affinity could not be enforced')
        except ProcessLookupError:
            continue
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError('Unable to initialize CPU-affinity protection')
    try:
        syscall = lib.seccomp_syscall_resolve_name(b'sched_setaffinity')
        if syscall < 0:
            raise RuntimeError('Cannot resolve sched_setaffinity')
        # TSYNC installs the restriction on every existing thread, too.
        for code in [lib.seccomp_attr_set(context, 4, 1),
                     lib.seccomp_rule_add(context, 0x50000 | errno.EPERM, syscall, 0)]:
            if code != 0:
                raise RuntimeError(f'Unable to configure affinity protection: {code}')
        code = lib.seccomp_load(context)
        if code != 0:
            raise RuntimeError(f'Unable to install affinity protection: {code}')
    finally:
        lib.seccomp_release(context)
    return dict(logical_cpus=sorted(cpus), maximum_combined_cpu_percent=200,
                library_threads=1, affinity_widening_blocked=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python3 scripts/tools/cpu_budget.py PROGRAM [ARG ...]')
    if (ROOT / 'cpu_limits.json').exists():
        apply_cpu_budget()
    os.execvp(sys.argv[1], sys.argv[1:])

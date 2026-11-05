"""Install only parent-compatible regression tests in a clean baseline checkout."""
import argparse
from pathlib import Path
import subprocess

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('baseline',type=Path)
a=ap.parse_args();base=a.baseline.resolve();src=Path(__file__).resolve().parents[2]
if subprocess.check_output(['git','status','--porcelain'],cwd=base).strip():
    raise SystemExit('baseline must be clean; use a dedicated worktree')
for name in ['t_metal_tensor_lifetime.c']:
    (base/'tests'/name).write_bytes((src/'tests'/name).read_bytes())
p=base/'tests/test_internal.h';s=p.read_text();needle='void test_metal_v41_layer(void);'
assert s.count(needle)==1
p.write_text(s.replace(needle,needle+'\nvoid test_metal_tensor_lifetime(void);'))
p=base/'tests/t_main.c';s=p.read_text();needle='#ifdef __APPLE__\n'
assert needle in s
p.write_text(s.replace(needle,needle+'    {"--metal-tensor-lifetime", "metal-tensor-lifetime", "bounded buffer lifetime (no model)", test_metal_tensor_lifetime, 0},\n',1))
print('Installed test-only probes; production source is unchanged.')

#!/usr/bin/env python3
# pick_active_experts.py — turn a go1b routing dump into per-layer Go-active expert
# lists for a sparse residual (emit_residual --active-experts).
#
# Collect the dump first (raw top-k picks per layer, with repetition = frequency):
#   DS4_DUMP_ACTIVE=1 ./ds4 -m gguf/ds4-go1b.gguf --prompt-file go_program.txt \
#       -n 4 --temp 0 --metal 2>picks.log
# then:
#   pick_active_experts.py picks.log go_active.txt [coverage=1.0]
#
# coverage is the fraction of routed picks (per layer) the kept top-K experts must
# cover: 1.0 = every expert the workload routed to, 0.80 = the frequent ~14%. Only
# the kept experts get a residual; the rest stay pure 1-bit (residual slot = -1).
import re, sys, collections

src = sys.argv[1] if len(sys.argv) > 1 else "picks.log"
dst = sys.argv[2] if len(sys.argv) > 2 else "go_active.txt"
cov = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

per = collections.defaultdict(collections.Counter)
for line in open(src):
    m = re.match(r'PICKS L(\d+)((?: -?\d+)+)', line)
    if not m:
        continue
    L = int(m.group(1))
    for x in m.group(2).split():
        e = int(x)
        if e >= 0:
            per[L][e] += 1

if not per:
    sys.exit("no PICKS lines in %s (run with DS4_DUMP_ACTIVE=1)" % src)

tot = 0
with open(dst, "w") as out:
    for L in sorted(per):
        cnt = per[L]
        total = sum(cnt.values())
        acc, sel = 0, []
        for e, c in cnt.most_common():
            sel.append(e); acc += c
            if acc >= cov * total:
                break
        tot += len(sel)
        out.write("L%d: %s\n" % (L, " ".join(map(str, sorted(sel)))))

nL = len(per)
print("wrote %s: %d layers, %d expert-layers (avg %.1f/layer, %.1f%% of 256), ~%.1fG"
      % (dst, nL, tot, tot / nL, tot / (nL * 256) * 100, tot / (nL * 256) * 35))

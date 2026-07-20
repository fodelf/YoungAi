#!/bin/bash
# layer_report.sh — 逐层指标提取与跨配置对比(优化项跟踪的固定入口)。
# ds4quant_run 每层打印: 该层局部指标(局部R²/R‖S/校准/z^L k) + 到该层累积指标。
# 累积自 2026-07-10 起拆 fit/held 两列(z^L 优化目标=fit, held 列=逐层可见的泛化真相);
# 旧格式(混算)自动兼容, held 列显示 '-'。
#
# 用法:
#   ./layer_report.sh <raw>                # 单跑全表
#   ./layer_report.sh <raw1> <raw2> [...]  # 多跑并排(累积fit|held|局部R²), 逐层跟踪优化项
#   CSV=1 ./layer_report.sh <raw> > x.csv  # CSV 留档(小文本, 可进 repo reports/)
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }

extract(){ # → 层 档 cum_fit cum_held 路由 校准µ 空/命中 局部R2 R/S zk
  perl -ne '
    if(/^L(\d+) (\S+) /){ my($l,$c)=($1,$2);
      my($cf,$ch)=("-","-");
      if(/累积 relL2=([\d.]+)/){$cf=$1;}
      if(/累积relL2 fit=([\d.]+) held=([\d.]+)/){$cf=$1;$ch=$2;}
      next if $cf eq "-";
      my $rt = /路由一致=\s*([\d.]+)%/ ? $1 : "-";
      my($mu,$emp,$hit)=("-","-","-"); if(/校准µ=([\d.]+)行 空=(\d+)\/(\d+)/){($mu,$emp,$hit)=($1,$2,$3);}
      my($lr,$rs)=("-","-"); if(/局部R²=\s*(-?[\d.]+)% R\/S=([\d.]+)/){($lr,$rs)=($1,$2);}
      my $zk = /z\^L k=(\d+)/ ? $1 : "-";
      print join("\t",$l,$c,$cf,$ch,$rt,$mu,"$emp/$hit",$lr,$rs,$zk),"\n";
    }' "$1"
}

if [ $# -eq 1 ]; then
  if [ "${CSV:-0}" = 1 ]; then
    echo "layer,cfg,cum_relL2_fit,cum_relL2_held,route_agree,calib_mu,calib_empty_hit,loc_R2,RS,zk"
    extract "$1" | tr '\t' ','
  else
    printf "%-4s %-3s %-8s %-8s %-6s %-6s %-7s %-7s %-5s %-4s\n" 层 档 累fit 累held 路由% 校准µ 空/中 局部R2% R/S zk
    extract "$1" | awk -F'\t' '{printf "%-4s %-3s %-8s %-8s %-6s %-6s %-7s %-7s %-5s %-4s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10}'
    grep '^VERDICT' "$1" 2>/dev/null || true
  fi
else
  T=$(mktemp -d); hdr="层"; i=0
  for f in "$@"; do
    i=$((i+1)); extract "$f" | awk -F'\t' '{print $1"\t"$3"\t"$4"\t"$8}' > "$T/$i"
    hdr="$hdr\t$(basename "$f" .raw)(fit|held|局R²)"
  done
  echo -e "$hdr"
  for L in $(seq 0 42); do
    row="L$L"
    for j in $(seq 1 $i); do
      v=$(awk -F'\t' -v l=$L '$1==l{print $2"|"$3"|"$4}' "$T/$j" | head -1)
      row="$row\t${v:--}"
    done
    echo -e "$row"
  done
  rm -rf "$T"
fi

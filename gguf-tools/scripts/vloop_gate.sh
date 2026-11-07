#!/bin/bash
# 累进门只读汇总：整数正确位数严格增加，KL保留原+0.001保护带。
# Same两位小数会把真实+1位舍掉；缺失、重复或非法行必须失败，不能当0分接力。
vloop_metrics() {
    local rows=$1 expected=$2
    [[ "$expected" =~ ^[0-9]+$ ]] && [ "$expected" -gt 0 ] && [ -s "$rows" ] || return 2
    LC_ALL=C awk -v expected="$expected" '
      NF < 4 || $1 !~ /^[0-9]+$/ || $1>=expected || ($4 != "0" && $4 != "1") ||
      $2 !~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ {bad=1;next}
      seen[$1]++ {bad=1;next}
      {n++;same+=$4;kl+=$2}
      END {if(bad || n!=expected || kl<0)exit 2;printf "%d %d %.12g\n",same,n,kl/n}
    ' "$rows"
}

vloop_better() {
    local before=$1 after=$2 total=$3 oldkl=$4 newkl=$5
    [[ "$before" =~ ^[0-9]+$ && "$after" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || return 2
    [[ "$oldkl" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ && "$newkl" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || return 2
    LC_ALL=C awk -v a="$before" -v b="$after" -v n="$total" -v x="$oldkl" -v y="$newkl" \
        'BEGIN {exit !(n>0 && a<=n && b<=n && b>a && y<=x+0.001)}'
}

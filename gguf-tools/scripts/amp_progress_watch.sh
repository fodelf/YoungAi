#!/bin/bash
# amp_progress_watch.sh — 反修长跑的【中间指标看板】(2026-09-21, 用户令"中间的指标也要看的, 不然出错了可能都不知道")
#
# 【为什么要它】一条反修链 ≈ 100 分钟, 而五指标只在收工那一刻才打。中间任何一件出错 ——
# 某层增益解崩、路由自检掉线、夹具与 ids 错位、进程被看门狗杀掉 —— 都要等到最后才看得见,
# 那时候一个半小时已经烧掉了。本脚本把【每层刚落盘的读数】与【参照反修目录的同层读数】逐层对账,
# 偏了当场打 ★。
#
# 【读哪里】不读日志尾巴, 读 manifest.txt —— 解算器每解完一层就 fprintf + fflush 一次, 它是盘上最早
# 出现的真事实(日志可能还堵在管道里)。判决段读主日志里 anchor_metrics 的输出行。
#
# 【它不做什么】不判决、不停车、不改任何东西 —— 只报数。要停车是解算器自己的事(路由自检 <99% 硬停)。
#
# 用法: amp_progress_watch.sh <新反修目录> <参照反修目录|none> <主日志> [轮询秒]
#   例: amp_progress_watch.sh gguf/v41/…-grrb-…-engine gguf/v41/…-gr-…-engine /tmp/best_grrb.log 60
# 每行一个事件(给 Monitor 当事件流)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
NEW="${1:?新反修目录}"; REFD="${2:-none}"; LOG="${3:?主日志}"; IV="${4:-60}"
MF="$NEW/manifest.txt"; RMF="$REFD/manifest.txt"
# ★偏离阈值★: 09-20 实测"挂着 rb 解增益"每层 val 比独立解高 0.1~0.7 pp ⇒ 低 2 pp 以上不是耦合差, 是出事了。
DROP=2.0
STALL=600          # 多久没有新层就报卡住(先例: 一层 50~100 s, 落盘后重取那一趟翻倍)

say(){ echo "$(date +%H:%M:%S) $*"; }
# 参照表: 层 → 增益 val(挂上的层才有; 被闸的层记 skip); RB = 本趟每层的路由读数(等同层增益一起发)
declare -A RV RB
if [ -s "$RMF" ]; then
    while read -r a b c d e f g; do
        case "$b" in
          gr)   RV[$a]="$f" ;;                       # L00 gr λ dz train val [min,max]
          skip) [ "$c" = gr ] && RV[$a]="skip" ;;    # L30 skip gr - - train val -
        esac
    done < <(grep -E "^L[0-9][0-9] " "$RMF")
    say "[看板] 参照 $(basename "$REFD"): ${#RV[@]} 层"
else
    say "[看板] 没有参照 manifest($RMF), 只报绝对值不对账"
fi

seen_gr=""; seen_rb=""; nfix=-1; last_new=$(date +%s); done_judge=0
while :; do
    # ---- 段 A: 教师链态夹具(还没有 manifest 的时候) ----
    if [ ! -s "$MF" ]; then
        fl=$(ls gguf/v41judge/log_fpdump_*.txt 2>/dev/null | tail -1)
        if [ -n "$fl" ]; then
            n=$(grep -c "^  \[L[0-9]" "$fl" 2>/dev/null || echo 0)
            if [ "$n" -ge $((nfix + 5)) ]; then nfix=$n; say "[夹具] 第 $n/40 层 ($(du -sh gguf/v41judge/fpdump_* 2>/dev/null | head -1 | cut -f1))"; fi
            p=$(grep -oE "PPL[^=]*= *[0-9.]+" "$fl" | tail -1)
            [ -n "$p" ] && say "[夹具] 收工 $p ★参照: FP 教师在拟合份 a 8192 上 9.1389, 差 >1% 就是取料/ids 错位★"
        fi
    fi
    # ---- 段 B: 40 层合并序贯(manifest 每层一行, 落盘即报) ----
    if [ -s "$MF" ]; then
        while read -r a b c d e f g; do
            case "$b" in
              rb) case " $seen_rb " in *" $a "*) continue;; esac; seen_rb="$seen_rb $a"
                  # L00 rb 挂|skip α=1 自检 0.9999 val同集 67.50→70.40 拟合 .. 武装 .. 系统 ..
                  # ★一层只发一个事件★: 路由读数先存起来, 跟着同层的增益一起发(两者隔着一趟重取料, 分开发就是两倍噪声)。
                  # 只有"自检掉线"这种要立刻知道的才单独先发 —— 它意味着解算器下一步就停车。
                  chk=$(sed -n "s/^$a rb .*自检 \([0-9.]*\).*/\1/p" "$MF" | tail -1)
                  sam=$(sed -n "s/^$a rb .*val同集 \([0-9.]*→[0-9.]*\).*/\1/p" "$MF" | tail -1)
                  arm=$(sed -n "s/^$a rb .*武装 \([0-9]*\).*/\1/p" "$MF" | tail -1)
                  RB[$a]="路由 $c α$(sed -n "s/^$a rb .*α=\([0-9.]*\).*/\1/p" "$MF" | tail -1) 自检 $chk 同集 $sam 武装 $arm"
                  awk -v v="$chk" 'BEGIN{exit !(v+0 < 0.99)}' && say "★$a 路由自检 $chk < 99%: 重算口径不是引擎的, 解算器下一步停车★"
                  last_new=$(date +%s) ;;
              gr|skip)
                  [ "$b" = skip ] && [ "$c" != gr ] && continue
                  case " $seen_gr " in *" $a "*) continue;; esac; seen_gr="$seen_gr $a"
                  if [ "$b" = gr ]; then v="$f"; lam="$c"; st="挂"; else v="$g"; lam="-"; st="闸"; fi
                  r="${RV[$a]:-}"; cmp=""; flag=""
                  if [ -n "$r" ] && [ "$r" != skip ] && [ "$st" = 挂 ]; then
                      cmp=$(awk -v n="$v" -v o="$r" 'BEGIN{printf "(参照 %s, Δ%+.2f)", o, n-o}')
                      awk -v n="$v" -v o="$r" -v d="$DROP" 'BEGIN{exit !(n-o < -d)}' && flag=" ★比参照低 >${DROP}pp: 查取料/路由是否解错★"
                  elif [ "$r" = skip ] && [ "$st" = 挂 ]; then cmp="(参照 被闸)"
                  elif [ -n "$r" ] && [ "$r" != skip ] && [ "$st" = 闸 ]; then cmp="(参照 $r)"; flag=" ⚠参照挂着这层, 本趟被闸"
                  fi
                  say "$a ${RB[$a]:-路由 -} | 增益 $st λ=$lam val ${v}% $cmp$flag"; last_new=$(date +%s) ;;
            esac
        done < <(grep -E "^L[0-9][0-9] " "$MF")
        grep -q "^# 完成" "$MF" && say "[序贯] $(grep '^# 完成' "$MF") —— 进判决段"
    fi
    # ---- 段 C: 判决(anchor_metrics 的五指标行) ----
    if [ "$done_judge" = 0 ] && grep -q "Same top token" "$LOG" 2>/dev/null; then
        done_judge=1
        say "[五指标] $(grep -E 'PPL\(student\)|Σmin|Mean KLD|Same top token' "$LOG" | tail -4 | tr -s ' ' | tr '\n' ' ')"
        say "[对照] 现役对 v3+gr-only: PPL 8.9463 / Σmin 0.7421 / KLD 0.52164 / Same top 73.94%"
    fi
    # ---- 终止与卡住 ----
    if grep -qE "收工, 产物在|个档失败" "$LOG" 2>/dev/null; then
        grep -E "个档失败|收工, 产物在" "$LOG" | tail -2 | while read -r l; do say "[收工] $l"; done; exit 0
    fi
    if ! pgrep -f "v41_teacher[.]py|v41_amp_ru[n]|ds4 -m gguf/v41[/]" >/dev/null; then
        say "★没有活进程且没收工 —— 链死了, 尾巴:★"; tail -5 "$LOG"; exit 1
    fi
    now=$(date +%s)
    if [ $((now - last_new)) -gt "$STALL" ] && [ -s "$MF" ]; then
        say "★$(( (now-last_new)/60 )) 分钟没有新层落盘(先例一层 50~100 s) —— avail $(free -g | awk '/^内存|^Mem/{print $7}')G, 尾巴:★"
        tail -2 "$LOG"; last_new=$now
    fi
    sleep "$IV"
done

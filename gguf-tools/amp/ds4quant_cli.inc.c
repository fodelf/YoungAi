/* ds4quant_cli.inc.c — 反修引擎唯一的运行配置入口(2026-08-31 env 大扫除).
 *
 * 此前 ~160 处 env 读取是三类事故的共同温床: C 默认与脚本实际值漂移(GS_SCREEN_DIV 4↔12)、
 * 静默兜底(DS4_HF 写死路径 / GO2B 热表默认 top64 vs 实传 top49)、开关值误当路径写文件
 * (DS4_ROUTE_BIAS_FIT="1" 落盘成文件名 `1`)。全部收敛到本表: 不传 flag = 原 env 不设的
 * 行为(保底不变式, 逐处核对); 新旋钮只许加表项, 禁再碰 env 读取(铁律 2026-08-22)。 */

/* ===== 写死常量: 原 env 数值旋钮, 无活脚本设置者按 env-不设默认固化 =====
 * 改这里=改行为, 没有运行时开关。 */
enum {
    DSQ_GS_SCREEN_DIV  = 12,  /* 回扫探测/sweep 粗筛抽格密度 — p12 与 p13 原是同一概念两份拷贝(env 默认12 / SDIV=12), 合一 */
    DSQ_MINVOL_FLOOR   = 1,   /* 1=逐层地板门(精修遍); 0=快扫遍(跳地板) */
    DSQ_MV_FLOOR_LINE  = 0,   /* 1=每层现场裸评冠军配方当门线(未启用的实验路, 代码保留) */
    DSQ_MV_COAD_BASE   = 0,   /* 1=固定档(mv-baseline)也 coadapt 定稿 */
    DSQ_BF_ALT         = 2,   /* 同层联合(网格→向后→z)块坐标下降轮数 */
    DSQ_COADAPT_LZRANK = 16,  /* --coadapt>0 且 z 秩未定时的秩上限(原 DS4_LZ 未设静默兜底, 显式化) */
};
static const double DSQ_TUNE_GAIN     = 0.15;   /* 渐进调优: q2 升位需挽回的相对收益门 */
static const double DSQ_MINVOL_ALPHA  = 1.03;   /* 最小体积: 地板门 α(日志口径) */
static const double DSQ_MINVOL_TARGET = 0.768;  /* 最小体积: rr 预算门终线(冠军尺 Σmin) */
static const double DSQ_MINVOL_EPS    = 0.0005; /* 地板门容差: 实测≥冠军同层−EPS 即不比冠军差 */
static const double DSQ_MINVOL_SOFT   = 0.05;   /* 软门纯防灾掉幅(冠军配方浅层每层耗~0.046, 不误咬) */
static const double DSQ_GS_CONV_PCT   = 0.1;    /* 回扫轮间收敛判停: 改善占起点比例低于此即止(%) */
static const float  DSQ_ROUTE_GATE_TAU= 0.0f;   /* token 门控路由偏置阈(0=全量应用, 旧默认) */

/* ===== 运行配置(原 env 集群): main 起点解析一次, 全 TU 只读 ===== */
typedef struct {
    /* 路径/字符串(NULL=未传) */
    const char *hf, *anchor, *anchor2, *anchor_old, *chain_anchor;
    const char *layer_dir, *lcfg, *zfile, *zchain, *ckpt_dir, *plan;
    const char *rr_ids, *route_bias, *route_bias_out, *dump_logits;
    const char *merge_gguf, *merge_off, *export_gguf, *export_off, *layer_file;
    const char *go2b_hot_table, *go2b_dir, *vq_rplan, *xcap_out, *elm_probe, *rot_probe;
    const char *route_miss_out, *hsum_out, *dw_spec, *dump_hidden, *mc_probe, *sub_probe;   /* 针: 逐行路由错配落盘 / 逐层出口 hash 落盘(跑间对拍) */
    /* 数值(哨兵=env 不设) */
    int nfit;              /* -1=默认 3/4 切分; 1=判尺纯回放(caliper 既有约定) */
    int threads;           /* 0=按核数 */
    int nl;                /* 0=全 43 层 */
    int gsweep;            /* -1=不回扫; 值=轮数 */
    int coadapt;           /* -1=不设 */
    int export_bytes;      /* 0=反修段 dql 不可变 */
    int go2b_hot;
    int fp_ablate;         /* 针: 单层置 FP 链上敏感度(链填完即打, 不进 sweep) */
    int bf_term_maxp;
    int minvol_maxl;       /* 0=不设 */
    int calib_cap, calib_export_cap;
    int route_bias_mincnt;
    int export_layer;      /* -1=不导出(0 是合法层号) */
    double bf_memgb;       /* 0=物理内存一半 */
    double route_bias_alpha, bf_gain_gate, tune_min;
    double signref_mu;     /* -1=保持 qhelp 默认 1.0 */
    /* 布尔开关 */
    int bf_only, calib_fullset, fp_only, minvol, tune, mv_baseline, pure_vq;
    int zchain_only, merge_resume, merge_consume, repair_cold, fast;
    int bwd, bwd_final, route_bias_fit, vq;
    int fp_oracle;   /* 上界探针: B 回放逐层 routed←FP@链态(读数是天花板不是模型) */
    int lz;          /* z 秩上限(候选表 lz=0 时生效; 原 DS4_LZ, env 清退时漏建 CLI 口) */
} dsq_cli_t;
static dsq_cli_t g_cli = {
    .nfit=-1, .gsweep=-1, .coadapt=-1, .export_layer=-1,
    .export_bytes=1, .bf_term_maxp=1, .route_bias_mincnt=8,
    .route_bias_alpha=1.0, .bf_gain_gate=0.05, .signref_mu=-1.0,
    .anchor=NULL,   /* anchor_path() 补 /tmp 默认; 显式传才启用建锚 MAP_SHARED 直写 */
    .zfile="/tmp/ds4quant_zfile.bin",
    .ckpt_dir="/tmp/ds4q_ckpt", .plan="/tmp/ds4quant_plan.txt",
    .layer_file="/tmp/dql_L00.bin",
};

typedef enum { DQO_BOOL, DQO_INT, DQO_DBL, DQO_STR } dqo_ty;
typedef struct { const char *name; dqo_ty ty; void *dst; const char *help; } dqo_t;
static const dqo_t DQOPT[] = {
    {"hf",              DQO_STR, &g_cli.hf,              "HF safetensors 目录(必填, 写死回落已删)"},
    {"anchor",          DQO_STR, &g_cli.anchor,          "FP 锚缓存(默认 /tmp/ds4quant_anchor.bin)"},
    {"anchor2",         DQO_STR, &g_cli.anchor2,         "rr 判决锚(配 --rr-ids)"},
    {"anchor-old",      DQO_STR, &g_cli.anchor_old,      "冷专家修复的旧锚(配 --repair-cold)"},
    {"chain-anchor",    DQO_STR, &g_cli.chain_anchor,    "链态锚直写路径"},
    {"layer-dir",       DQO_STR, &g_cli.layer_dir,       "dql 层文件目录"},
    {"lcfg",            DQO_STR, &g_cli.lcfg,            "逐层档位串(NL 字符或 1 字符广播)"},
    {"nl",              DQO_INT, &g_cli.nl,              "前向层数上限(默认 43)"},
    {"nfit",            DQO_INT, &g_cli.nfit,            "校准行数(默认 3/4; 1=判尺纯回放)"},
    {"threads",         DQO_INT, &g_cli.threads,         "专家循环并行度(默认按核数)"},
    {"coadapt",         DQO_INT, &g_cli.coadapt,         "每层 base↔z 共适应轮数(0=关)"},
    {"calib-fullset",   DQO_BOOL,&g_cli.calib_fullset,   "校准全集喂入(不做命中过滤)"},
    {"calib-cap",       DQO_INT, &g_cli.calib_cap,       "量化路每专家校准行帽(0=全集)"},
    {"calib-export-cap",DQO_INT, &g_cli.calib_export_cap,"导出路校准行帽(0=全集)"},
    {"signref-mu",      DQO_DBL, &g_cli.signref_mu,      "signref 权重锚强度(默认 1)"},
    {"tgt-alpha",       DQO_DBL, &dq_tgt_alpha_v,        "GPTAQ 非对称目标强度[0,1]"},
    {"vq",              DQO_BOOL,&g_cli.vq,              "VQ 码本量化(读点在 quantize/vq_qc.h, dq_vq_set 直连)"},
    {"vq-rplan",        DQO_STR, &g_cli.vq_rplan,        "R28 每层计划表"},
    {"go2b-hot",        DQO_INT, &g_cli.go2b_hot,        "热专家 go2b 合并态量化(1=开)"},
    {"go2b-hot-table",  DQO_STR, &g_cli.go2b_hot_table,  "热表路径(开 --go2b-hot 必填, 禁默认表)"},
    {"go2b-dir",        DQO_STR, &g_cli.go2b_dir,        "go2b 侧车目录(默认=层文件同目录)"},
    {"minvol",          DQO_BOOL,&g_cli.minvol,          "最小体积·计划表驱动模式"},
    {"minvol-maxl",     DQO_INT, &g_cli.minvol_maxl,     "贪心探针: 锁 N 层提前收工(禁写 zchain)"},
    {"mv-baseline",     DQO_BOOL,&g_cli.mv_baseline,     "基线模式: 逐层冠军配方"},
    {"pure-vq",         DQO_BOOL,&g_cli.pure_vq,         "纯 VQ 量化(导出先行+B 回放链态)"},
    {"tune",            DQO_BOOL,&g_cli.tune,            "逐层渐进调优模式"},
    {"tune-min",        DQO_DBL, &g_cli.tune_min,        "总时间预算(分, 0=不限, 只选档不截断)"},
    {"ckpt-dir",        DQO_STR, &g_cli.ckpt_dir,        "调优激活 ckpt 目录(默认 /tmp/ds4q_ckpt)"},
    {"plan",            DQO_STR, &g_cli.plan,            "调优 plan 文件(默认 /tmp/ds4quant_plan.txt)"},
    {"bf-only",         DQO_BOOL,&g_cli.bf_only,         "只跑反修(复用既有层文件, SEARCH 跳过)"},
    {"bf-memgb",        DQO_DBL, &g_cli.bf_memgb,        "fp16 层缓存内存预算 GiB(默认=物理内存一半)"},
    {"bf-gain-gate",    DQO_DBL, &g_cli.bf_gain_gate,    "层落地增益门 %(默认 0.05)"},
    {"bf-term-maxp",    DQO_INT, &g_cli.bf_term_maxp,    "终局反修轮上限(默认 1)"},
    {"gsweep",          DQO_INT, &g_cli.gsweep,          "全局联合回扫轮数(不传=不回扫)"},
    {"fast",            DQO_BOOL,&g_cli.fast,            "快速走流程(验流程非出质量)"},
    {"bwd",             DQO_BOOL,&g_cli.bwd,             "反修段保留既有 VQ 载荷(不重编码)"},
    {"bwd-final",       DQO_BOOL,&g_cli.bwd_final,       "向后·终端反调(t 插值选优)"},
    {"route-bias",      DQO_STR, &g_cli.route_bias,      "路由偏置侧车(apply)"},
    {"route-bias-alpha",DQO_DBL, &g_cli.route_bias_alpha,"Δb 应用强度(默认 1.0)"},
    {"route-bias-mincnt",DQO_INT,&g_cli.route_bias_mincnt,"Δb 武装最小计数(默认 8)"},
    {"route-bias-fit",  DQO_BOOL,&g_cli.route_bias_fit,  "量化前向顺带统计 Δb"},
    {"route-bias-out",  DQO_STR, &g_cli.route_bias_out,  "Δb 落盘路径"},
    {"fp-only",         DQO_BOOL,&g_cli.fp_only,         "只建 FP 锚即止"},
    {"export-bytes",    DQO_INT, &g_cli.export_bytes,    "0=反修段 dql 不可变(默认 1=写字节)"},
    {"dump-logits",     DQO_STR, &g_cli.dump_logits,     "量化 logits 落盘路径"},
    {"rr-ids",          DQO_STR, &g_cli.rr_ids,          "rr 判决语料 ids(配 --anchor2)"},
    {"zfile",           DQO_STR, &g_cli.zfile,           "合并动态侧车 DQZ1(默认 /tmp/ds4quant_zfile.bin)"},
    {"zchain",          DQO_STR, &g_cli.zchain,          "DQZ2 全链侧车路径"},
    {"zchain-only",     DQO_BOOL,&g_cli.zchain_only,     "只从 dql 重建 zchain/opt(不量化)"},
    {"fp-oracle",       DQO_BOOL,&g_cli.fp_oracle,       "上界探针: B 回放逐层 routed←FP@链态(需 --hf; 读数≠模型)"},
    {"lz",              DQO_INT, &g_cli.lz,              "z 秩上限(冠军 64; 0=按候选表/coadapt 兜底 16; 原 DS4_LZ)"},
    {"merge-gguf",      DQO_STR, &g_cli.merge_gguf,      "合并: go1b 骨架 GGUF"},
    {"merge-off",       DQO_STR, &g_cli.merge_off,       "合并: 偏移表(gguf_offsets 输出)"},
    {"merge-resume",    DQO_BOOL,&g_cli.merge_resume,    "续并: 缺层=已消费跳过"},
    {"merge-consume",   DQO_BOOL,&g_cli.merge_consume,   "边并边释放层文件(峰值盘占恒定)"},
    {"export-gguf",     DQO_STR, &g_cli.export_gguf,     "直写导出目标 GGUF"},
    {"export-off",      DQO_STR, &g_cli.export_off,      "直写导出偏移表"},
    {"export-layer",    DQO_INT, &g_cli.export_layer,    "收官导出单层 L 的层文件"},
    {"layer-file",      DQO_STR, &g_cli.layer_file,      "--export-layer 目标(默认 /tmp/dql_L00.bin)"},
    {"repair-cold",     DQO_BOOL,&g_cli.repair_cold,     "②冷专家修复(需 --anchor-old + --layer-dir)"},
    {"xcap-out",        DQO_STR, &g_cli.xcap_out,        "量化链 x 捕获目录"},
    {"elm-probe",       DQO_STR, &g_cli.elm_probe,       "ELM 针层号列表"},
    {"rot-probe",       DQO_STR, &g_cli.rot_probe,       "B类底座旋转针层号列表"},
    {"route-miss-out",  DQO_STR, &g_cli.route_miss_out,  "针: 逐行路由错配计数落盘(学生top-k槽不在FP锚集)"},
    {"hsum-out",        DQO_STR, &g_cli.hsum_out,        "针: 逐层出口 hidden FNV64 落盘(同态两跑对拍定位非确定层)"},
    {"fp-ablate",       DQO_BOOL,&g_cli.fp_ablate,       "针: 单层置FP链上敏感度(43枪, 只读, 打完即退)"},
    {"dw-spectrum",     DQO_STR, &g_cli.dw_spec,         "针: 量化残差ΔW奇异谱(层号列表; 每层3专家×w1/w3/w2), 打完即退"},
    {"mc-probe",        DQO_STR, &g_cli.mc_probe,        "针: 巨值通道(层号列表): 通道 Σx² 集中度 + w1/w3 复原 top-c 列后输出误差剩余, 需 --anchor"},
    {"sub-probe",       DQO_STR, &g_cli.sub_probe,       "针: 专家输入子空间(层号列表): Fin 协方差前k主方向份额 + 子空间精确后输出误差剩余, 需 --anchor"},
    {"dump-hidden",     DQO_STR, &g_cli.dump_hidden,     "头前归一化隐状态落盘 [S][DIM] f32(另写 <路径>.head = head.weight [VOCAB][DIM]); 输出头放大器拟合料"},
};
#define NDQOPT ((int)(sizeof(DQOPT)/sizeof(DQOPT[0])))

static void dsq_usage(void){
    fprintf(stderr,"用法: ds4quant_run [ids文件=/tmp/rr_hard.ids] [ntok=64] [--flag [值]...]\n"
                   "值型 flag 接受 --key 值 与 --key=值 两种写法; 开关型不带值。\n");
    for(int k=0;k<NDQOPT;k++){
        const char*th=DQOPT[k].ty==DQO_BOOL?"":DQOPT[k].ty==DQO_STR?" <路径/串>":" <数>";
        fprintf(stderr,"  --%s%s\n      %s\n",DQOPT[k].name,th,DQOPT[k].help);
    }
}
static void dsq_cli_parse(int argc,char**argv,const char**idf,int*ntok){
    int npos=0;
    for(int i=1;i<argc;i++){
        const char*a=argv[i];
        if(a[0]=='-'&&a[1]=='-'){
            const char*eq=strchr(a,'=');
            size_t nl2=eq?(size_t)(eq-(a+2)):strlen(a+2);
            const dqo_t*o=NULL;
            for(int k=0;k<NDQOPT;k++)
                if(strlen(DQOPT[k].name)==nl2&&!strncmp(DQOPT[k].name,a+2,nl2)){ o=&DQOPT[k]; break; }
            if(!o){ fprintf(stderr,"未知参数 %s\n\n",a); dsq_usage(); exit(2); }
            if(o->ty==DQO_BOOL){
                if(eq){ fprintf(stderr,"--%s 是开关, 不带值\n",o->name); exit(2); }
                *(int*)o->dst=1; continue;
            }
            const char*v=eq?eq+1:(i+1<argc?argv[++i]:NULL);
            if(!v||!*v){ fprintf(stderr,"--%s 缺值\n\n",o->name); dsq_usage(); exit(2); }
            if(o->ty==DQO_INT)      *(int*)o->dst=atoi(v);
            else if(o->ty==DQO_DBL) *(double*)o->dst=atof(v);
            else                    *(const char**)o->dst=v;
        } else {   /* 位置参数保持原契约: [1]=ids 文件 [2]=ntok */
            if(npos==0)*idf=a;
            else if(npos==1)*ntok=atoi(a);
            else { fprintf(stderr,"多余位置参数 %s\n\n",a); dsq_usage(); exit(2); }
            npos++;
        }
    }
    if(!g_cli.hf){
        fprintf(stderr,"--hf <dir> is required (HF safetensors 目录)\n\n");
        dsq_usage(); exit(2);
    }
}
/* 层目录硬取: 原多处站点在 env 缺席时静默回落 "." 把层文件写进当前目录(口径漂移源),
 * 2026-08-31 统一为响亮报错。 */
static const char*dsq_layer_dir_req(void){
    if(!g_cli.layer_dir){
        fprintf(stderr,"★缺 --layer-dir(层文件目录) — 禁静默读写当前目录, 停★\n");
        exit(2);
    }
    return g_cli.layer_dir;
}

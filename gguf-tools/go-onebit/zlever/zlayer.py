"""zlever/zlayer.py — 每层 z 侧车一体化(建缓存→四损失闭式解→注入 dql, 反修段主力)。
解 = z^L rank K(对偶 ridge+dither 增广+colw classify, 折 colw 后精确 SVD 截 K)
   + GE(每专家门控, k截断后残差 ridge)。解算位置 0..1287(held 1287..1716 留给指标)。
注入 = dql_LXX.bin 追加 116B 头记录(bf.GE + zl.RRR), nrec+2; 回滚账本 zinject_manifest.txt
      记录原始 size/nrec, 已注入层拒重复注入。
用法: zlayer.py <hf> <layers_dir> <anchor> <层> [K=1024] [inject=1]
"""
import os, sys, time, struct, numpy as np
def zl_svd(M):
    """gesdd 偶发不收敛: 转置重试→微抖动兜底(1e-7 相对幅度, 数值语义不变)"""
    if _GPU:
        try:
            A,S,B=cp.linalg.svd(cp.asarray(M),full_matrices=False)
            return cp.asnumpy(A),cp.asnumpy(S),cp.asnumpy(B)
        except Exception: pass
    try: return np.linalg.svd(M,full_matrices=False)
    except np.linalg.LinAlgError: pass
    try:
        A2,S2,B2=np.linalg.svd(np.ascontiguousarray(M.T),full_matrices=False)
        return B2.T,S2,A2.T
    except np.linalg.LinAlgError: pass
    j=1e-7*float(np.abs(M).mean()+1e-30)
    r=np.random.RandomState(7)
    return np.linalg.svd(M+j*r.randn(*M.shape).astype(M.dtype),full_matrices=False)
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant, st_raw, FP4T
from probe_layer_behavior import anchor_layer, swiglu
# GPU 加速(2026-08-18 用户令"先解决反修速度"): cupy 有则热点上 GPU(dequant/前向/SVD),
# 无(M1)自动走原 numpy 路 — 能力探测非行为开关, 数值同语义(fp32, 顺序重排容差)。
try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
    _FP4T_G = cp.asarray(FP4T)
except Exception:
    cp = None; _GPU = False
def st_mxfp4_g(hf, wmap, name):
    if not _GPU: return st_mxfp4(hf, wmap, name)
    dt, sh, raw = st_raw(hf, wmap, name)
    R, Cc = sh; Cin = Cc*2; nblk = Cin//32
    _, _, sraw = st_raw(hf, wmap, name.replace(".weight", ".scale"))
    e = cp.asarray(np.frombuffer(sraw, dtype=np.uint8)).reshape(R, nblk).astype(cp.uint32)
    u = cp.where(e == 0, cp.uint32(0x00400000), e << 23).astype(cp.uint32)
    scale = u.view(cp.float32).reshape(R, nblk)
    b = cp.asarray(np.frombuffer(raw, dtype=np.uint8)).reshape(R, nblk, 16)
    w = cp.empty((R, nblk, 32), dtype=cp.float32)
    w[:, :, 0::2] = _FP4T_G[b & 0x0F]; w[:, :, 1::2] = _FP4T_G[(b >> 4) & 0x0F]
    w *= scale[:, :, None]
    return w.reshape(R, Cin)   # 返回 cupy 数组(调用侧 GPU 前向直接吃)
def _vq_dequant_g(blob_np, off):
    """vq_dequant 的 cupy 版: 位抽取向量化(3 字节窗), 数值与 numpy 版逐位同"""
    dim, nc = struct.unpack_from("<HH", blob_np, off + 4)
    rows, cols = struct.unpack_from("<II", blob_np, off + 8)
    p = off + 16
    cb = cp.asarray(np.frombuffer(blob_np, dtype=np.float16, count=nc*dim, offset=p)).astype(cp.float32).reshape(nc, dim); p += nc*dim*2
    gr = cp.asarray(np.frombuffer(blob_np, dtype=np.float16, count=rows, offset=p)).astype(cp.float32); p += rows*2
    nbit = max(1, (nc-1).bit_length()); nidx = rows*cols//dim
    if nbit == 8:
        idx = cp.asarray(np.frombuffer(blob_np, dtype=np.uint8, count=nidx, offset=p)).astype(cp.int32)
    else:
        nby = (nidx*nbit)//8 + 4
        # 3字节窗过读填充: 槽在 blob 末尾时越界 → 零垫(sh+nbit≤14, 第3字节永不进结果, 语义同)
        avail = len(blob_np) - p
        arr = np.frombuffer(blob_np, dtype=np.uint8, count=min(nby, avail), offset=p)
        if avail < nby: arr = np.concatenate([arr, np.zeros(nby-avail, np.uint8)])
        bb = cp.asarray(arr).astype(cp.uint32)
        pos = cp.arange(nidx, dtype=cp.int64)*nbit
        bi = pos >> 3; sh = (pos & 7).astype(cp.uint32)
        v = (bb[bi] | (bb[bi+1] << 8) | (bb[bi+2] << 16))
        idx = ((v >> sh) & ((1 << nbit)-1)).astype(cp.int32)
    return (cb[idx] * gr.repeat(cols//dim)[:, None]).reshape(rows, cols)
def _mxfp4_g_raw(sh, raw, sraw):
    """st_mxfp4_g 的字节版(预读复用, 免二次 IO)"""
    R, Cc = sh; Cin = Cc*2; nblk = Cin//32
    e = cp.asarray(np.frombuffer(sraw, dtype=np.uint8)).reshape(R, nblk).astype(cp.uint32)
    u = cp.where(e == 0, cp.uint32(0x00400000), e << 23).astype(cp.uint32)
    scale = u.view(cp.float32).reshape(R, nblk)
    b = cp.asarray(np.frombuffer(raw, dtype=np.uint8)).reshape(R, nblk, 16)
    w = cp.empty((R, nblk, 32), dtype=cp.float32)
    w[:, :, 0::2] = _FP4T_G[b & 0x0F]; w[:, :, 1::2] = _FP4T_G[(b >> 4) & 0x0F]
    w *= scale[:, :, None]
    return w.reshape(R, Cin)
def _swiglu_g(g, u, lim):
    if lim > 0: g = cp.clip(g, -lim, lim); u = cp.clip(u, -lim, lim)   # 与 probe_layer_behavior.swiglu 逐字同(双向)
    return g/(1+cp.exp(-g))*u
hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
K=int(sys.argv[5]) if len(sys.argv)>5 else 1024
INJ=int(sys.argv[6]) if len(sys.argv)>6 else 1
D=4096
# ★标量 GGUF 模式(2026-08-19 用户令"弃VQ用平权标量86G"): DS4_ZL_GGUF=模型路径 时
# 量化侧权重 Wq 从 GGUF 专家张量切片 dequant(gguf-py), 不再读 dql_vq blob。
# 布局: blk.L.ffn_{gate,up,down}_exps.weight, 专家=外维连续 → 按字节均分切单专家。
_GG=os.getenv("DS4_ZL_GGUF")
if _GG:
    from gguf import GGUFReader
    from gguf.quants import dequantize as _gg_deq
    _ggr=GGUFReader(_GG)
    _ggt={t.name:t for t in _ggr.tensors}
    _GGNM={"w1":"ffn_gate_exps","w3":"ffn_up_exps","w2":"ffn_down_exps"}
    def _gg_expert(l,nm,e):
        t=_ggt[f"blk.{l}.{_GGNM[nm]}.weight"]
        ne=[int(x) for x in t.shape]          # [内维..外维], 外维=专家数
        nexp=ne[-1]; rows=ne[-2]; cols=ne[0]  # 每专家 [rows,cols]
        db=t.data.reshape(-1)
        per=db.size//nexp
        w=_gg_deq(db[e*per:(e+1)*per], t.tensor_type)
        return np.ascontiguousarray(w.reshape(rows,cols).astype(np.float32))
# ★通用锚口径(2026-08-09 z天花板测试): DS4_ZL_NTOK/DS4_ZL_NFIT 注入新锚的 S 与 fit 界, 默认=历史 1716/1287
NTOK=int(os.getenv("DS4_ZL_NTOK","1716"))
if INJ==2:   # 外挂模式断点续跑: zrec 已在则整层跳过(解算也省)
    _zp=os.path.join(ld,f"zrec_L{L:02d}.bin")
    if os.path.exists(_zp):
        print(f"★L{L} zrec 已存在({os.path.getsize(_zp)}B), 跳过", flush=True); sys.exit(0)
t0=time.time()
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
# ★XCAP(2026-08-22 用户令"放大器要跟当层量化模型的一切计算求解")★
# x 与被乘量都换成引擎真值: DS4_CAP_DIR 捕的 raw_ffn_in(post-ffn_norm 的 MoE 输入 —— attn/
# shared/norm/router 的量化误差全部已经沿链带进来了) 与 raw_ffn_out(真实部署态 routed 输出)。
# FP 侧仍走锚(教师)。这样放大器作用的位置、看到的输入、乘的张量三者与运行时逐字节同源,
# 不再是"FP 输入 + Python 重算量化专家"的理想化口径。
# 第 7 个位置参数 = 引擎捕获目录("-"或缺省 = 不用)。写成位置参数而非 env:
# 发车命令里一眼可见, 不会因为漏设环境变量而静默换一套口径(2026-08-22 铁律: 禁新增 env)。
XCAP=sys.argv[7] if len(sys.argv)>7 and sys.argv[7] not in ("","-") else None
# 第 8 个位置参数 = 上一轮放大器所在目录。给了就是"第二轮反修":
#   第 L 层的输入只由 0..L-1 层决定, 下游放大器碰不到它 —— 所以带全部放大器捕的
#   raw_ffn_in_L 就等于"上游已修好"时本层该看到的输入, 不必重捕 43 次。
#   被乘量要把上一轮的增益除回去还原: y_未修(新输入) = raw_ffn_out(带) / (1 + g_旧(新输入)),
#   g_旧 由上一轮的 A/U/V 和新输入算得(引擎当时用的就是这个新输入)。
PREV=sys.argv[8] if len(sys.argv)>8 and sys.argv[8] not in ("","-") else None
YQE=None
if XCAP:
    def _capload(nm,cols):
        """引擎捕获对齐: 引擎走 DS4_EVAL_IDS 时会在流首插 BOS, 于是捕获比锚多一行
        (8193 vs 8192)。锚是量化器直喂 ids、不插 BOS。差这一行就是整体错位一格 ——
        每个 token 的 x 会被配到前一个 token 的目标上, 而且不报任何错、照样出挽回率。
        实测判据(L0, 逐行余弦中位数): offset=0 → 0.2627(垃圾), offset=1 → 0.8946(对)。"""
        _p=os.path.join(XCAP,f"{nm}_L{L}")
        _a=np.fromfile(_p,dtype=np.float16)
        _n=_a.size//cols
        assert _n in (NTOK, NTOK+1), f"{_p}: {_n} 行, 既非 NTOK={NTOK} 也非 NTOK+1(BOS) — 口径不明, 拒跑"
        _off = _n - NTOK          # 1 = 掐掉流首 BOS 行
        return _a[_off*cols:(_off+NTOK)*cols].reshape(NTOK,cols).astype(np.float32)
    _XFP=X0                    # 锚 fin, 留作对齐自检的参照
    X0=_capload("raw_ffn_in",D)
    YQE=_capload("raw_ffn_out",D)
    # ★教师用 FP 锚的路由(2026-08-22 用户裁决)★
    # 一度改成"教师也用引擎的量化路由", 理由是"选错专家乘性增益够不着, 该从目标里剔除"。
    # 那个推理是错的: 够不着 ≠ 该换靶子。教师必须是我们真正要还原的那个模型 ——
    # FP 模型在这一层就是用它自己的路由 + FP 权重算的。对着真目标做最小二乘, 本来就会
    # 自动做到它能做的那部分; 换成"专家选择照抄量化模型、只有权重是 FP"的虚构模型,
    # 等于主动放弃一块本该争取的东西。
    # (实测: 引擎 bug 修复前后, FP 锚与引擎路由的平均重合都是 ~3.17/6 —— 这是真实的
    #  量化路由漂移, 观察没错; 错的是据此换靶子。)
    # 所以 ridx/rw 保持 anchor_layer 读来的 FP 路由, x 与 y_q 仍取引擎真值。
    if PREV:
        import struct as _st
        _b=open(os.path.join(PREV,f"zrec_L{L:02d}.bin"),"rb").read()
        _nm=_b[:16].split(b"\0")[0].decode(); _psz=_st.unpack_from("<Q",_b,88)[0]
        assert "zl.AMPD" in _nm, f"L{L} 上一轮记录不是 AMPD({_nm}), 无法还原增益"
        _pay=_b[116:116+_psz]
        _k,_sc,_di,_do=_st.unpack_from("<IfII",_pay,0)
        _h=np.frombuffer(_pay[16:],dtype=np.float16).astype(np.float32)
        _n=_di*_k
        _A=_h[:_n].reshape(_di,_k); _U=_h[_n:_n+_do*_k].reshape(_do,_k)
        _V=_h[_n+_do*_k:_n+_do*_k+_di*_k].reshape(_di,_k)
        _g=(np.tanh(X0@_V/_sc)*np.tanh(X0@_A/_sc))@_U.T
        _den=1.0+_g
        _bad=np.abs(_den)<1e-3
        if _bad.any(): _den=np.where(_bad,np.sign(_den)*1e-3+ (_bad*0),_den)
        YQE=(YQE/_den).astype(np.float32)
        print(f"  L{L} 第二轮: 已除回上一轮增益 |g_旧| 中位 {float(np.median(np.abs(_g))):.5f} "
              f"p99 {float(np.percentile(np.abs(_g),99)):.5f}",flush=True)
    # ★对齐自检★ 判的是"两种对齐哪个对"这个离散问题, 不是拿绝对余弦当质量闸。
    # 绝对值本来就随层数衰减(量化误差沿链累积): 实测 L0 0.89 → L17 0.59 → L42 0.56,
    # 早先我拍了个 0.6 的常数当门, 把 L17 之后全误拦了 —— 魔法常数的典型翻车。
    # 正确判据 = 正确对齐 vs 错位一格 的比值, 实测全层 1.4×~3.4×, 从不接近 1。
    def _cosmed(a,b):
        _n=min(len(a),len(b)); a,b=a[:_n],b[:_n]
        return float(np.median((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)))
    # 错位假设 = 采用对齐再平移一行。这样两种捕获都能判:
    #   批量捕获(NTOK+1 行, 含流首 BOS) 与 解码捕获(NTOK 行, 不插 BOS) 都适用。
    #   早先拿 _raw[:NTOK] 当错位版, 在解码捕获上它与正确版是同一个数组, 比值恒为 1 → 误报。
    _c_ok=_cosmed(_XFP,X0); _c_bad=_cosmed(_XFP[1:],X0[:-1])
    assert _c_ok > _c_bad*1.15, (f"L{L} XCAP 对齐自检失败: 采用对齐 {_c_ok:.4f} 未明显优于错位版 "
                                 f"{_c_bad:.4f} — 口径可疑, 停车")
    print(f"  L{L} XCAP: 对齐 {_c_ok:.4f} vs 错位 {_c_bad:.4f} ({_c_ok/max(_c_bad,1e-6):.1f}×) "
          f"|x|={np.linalg.norm(X0):.1f} |y_q|={np.linalg.norm(YQE):.1f}",flush=True)
ADDON=None; ge_old=None; zo=None   # 叠加式默认(非链跑安全)
XAP=os.getenv("DS4_ZL_XANCHOR")   # ★链模式(2026-08-10 反修v4): 部署链态锚(x_q/路由_q)
if XAP:
    XQ0,ridxq,rwq,_=anchor_layer(XAP,L,NTOK)
# ★叠加式链修正(2026-08-10 用户设计: 链修正贪心而非替换)★ DS4_ZL_ADDON=1:
    # 部署侧带上本层既有记录(GE乘入权重, z_old出力从dH扣除) → dH=贪心修后残差
    ADDON=os.getenv("DS4_ZL_ADDON")
    ge_old=None; zo=None
    if ADDON:
        _raw=open(os.path.join(ld,f"dql_L{L:02d}.bin"),"rb").read()
        _nr,=struct.unpack_from("<I",_raw,8); _off=12
        for _ in range(_nr):
            _nm=_raw[_off:_off+16].split(b"\0")[0].decode("ascii","replace")
            _psz,=struct.unpack_from("<Q",_raw,_off+88)
            _vd,=struct.unpack_from("<i",_raw,_off+112)
            _pay=_raw[_off+116:_off+116+_psz]; _off+=116+_psz
            if _vd!=1: continue
            if "bf.GE" in _nm and _psz>=512:
                ge_old=np.frombuffer(_pay[:512],dtype=np.float16).astype(np.float32).copy()
            elif "zl.RRR" in _nm and _psz>=16:
                _k0,_t0,_di,_do=struct.unpack_from("<IfII",_pay,0)
                _h=np.frombuffer(_pay[16:16+2*(_k0+_k0*_do+_k0*_di)],dtype=np.float16).astype(np.float32)
                zo=(int(_k0),int(_di),int(_do),_h[:_k0].copy(),
                    _h[_k0:_k0+_do*_k0].reshape(_do,_k0).copy(),
                    _h[_k0+_do*_k0:].reshape(_di,_k0).copy())
        print(f"  L{L} ADDON: 既有记录 GE={'有' if ge_old is not None else '无'} z={'k%d/din%d'%(zo[0],zo[1]) if zo else '无'}",flush=True)
cache=os.path.join(ld,f"zcache_L{L:02d}.npz")
# ★cache-only 模式(2026-08-20 用户"速度改坏了"): amp 战役里加性解算是纯陪跑浪费,
# 只产 zcache 即退, 求解全交 amp_solve。"0"/"" 视为关。
_CACHE_ONLY=os.getenv("DS4_ZL_CACHE_ONLY","0") not in ("","0")
if os.path.exists(cache):
    zc=np.load(cache)
    dH=zc["dH"]; prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]; pYQ=zc["pYQ"]
else:
    wmap=st_index(hf)
    blob=None if _GG else open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()

    if XAP:
        need=sorted(set(int(e) for e in ridx.reshape(-1))|set(int(e) for e in ridxq.reshape(-1)))
    else:
        need=sorted(set(int(e) for e in ridx.reshape(-1)))
    dH=np.zeros((NTOK,D),dtype=np.float32)
    prow=[];pe=[];pw=[];pYQ=[];pDY=[]
    if _GPU:
        # ★批量 GPU 驻留(2026-08-18 用户令"没利用好设备"): ①threadpool 并行预读原始字节
        # ②dequant/前向全 GPU, 数据零往返, dH/pYQ 在 GPU 累计, 收尾一次 D2H。
        # 数值=逐专家序同(浮点累计序一致), swiglu 双向 clip 同式。
        from concurrent.futures import ThreadPoolExecutor
        _lim=float(os.getenv("DS4_ZL_SWLIM","10"))
        def _read_e(e):
            out={}
            for nm in ("w1","w3","w2"):
                dt,sh,raw=st_raw(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
                _,_,sraw=st_raw(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.scale")
                out[nm]=(sh,raw,sraw,(_gg_expert(L,nm,e) if _GG else None))
            return e,out
        with ThreadPoolExecutor(8) as _ex: _raws=dict(_ex.map(_read_e,need))
        dH_g=cp.zeros((NTOK,D),dtype=cp.float32)
        X0_g=cp.asarray(X0.astype(np.float32))
        XQ0_g=cp.asarray(XQ0.astype(np.float32)) if XAP else X0_g
        _pYQ_g=[]
        for i,e in enumerate(need):
            rows,slots=np.where(ridx==e)
            if XAP: rowsq,slotsq=np.where(ridxq==e)
            else:   rowsq,slotsq=rows,slots
            if len(rows)==0 and len(rowsq)==0: continue
            P={}
            for nm,wh in (("w1",0),("w3",1),("w2",2)):
                sh_,raw_,sraw_,_wqn=_raws[e][nm]
                Wf=_mxfp4_g_raw(sh_,raw_,sraw_)
                if _GG: Wq=cp.asarray(_wqn)
                else:
                    off=vq_slot(blob,e,wh); Wq=_vq_dequant_g(blob,off) if off else None
                if Wq is not None and Wf.shape!=Wq.shape: Wf=Wf.T
                P[nm]=(Wf,Wq if Wq is not None else Wf)
            if len(rows)>0:
                w=rw[rows,slots].astype(np.float32)
                xg=X0_g[cp.asarray(rows)]
                Yf=_swiglu_g(xg@P["w1"][0].T,xg@P["w3"][0].T,_lim)@P["w2"][0].T
                dH_g[cp.asarray(rows)]+=cp.asarray(w)[:,None]*Yf
            if len(rowsq)>0 and not XCAP:   # XCAP: 量化侧直接用引擎 raw_ffn_out, 不重算
                wq=(rwq[rowsq,slotsq] if XAP else rw[rowsq,slotsq]).astype(np.float32)
                if ADDON and ge_old is not None: wq=wq*ge_old[e]
                xqg=XQ0_g[cp.asarray(rowsq)]
                Yq=_swiglu_g(xqg@P["w1"][1].T,xqg@P["w3"][1].T,_lim)@P["w2"][1].T
                dH_g[cp.asarray(rowsq)]-=cp.asarray(wq)[:,None]*Yq
                prow.append(rowsq); pe.append(np.full(len(rowsq),e,dtype=np.int32))
                pw.append(wq); _pYQ_g.append(Yq); pDY.append(np.zeros((len(rowsq),D),dtype=np.float32))
            if (i+1)%64==0: print(f"  L{L} 缓存(GPU) …{i+1}/{len(need)}", flush=True)
        dH=cp.asnumpy(dH_g)
        if XCAP: dH-=YQE          # dH = Σw·Y_fp(锚教师) − 引擎真实量化 routed 输出
        pYQ_all=cp.asnumpy(cp.vstack(_pYQ_g)) if _pYQ_g else np.zeros((0,D),dtype=np.float32)
        prow=np.concatenate(prow) if prow else np.zeros(0,dtype=np.int64)
        pe=np.concatenate(pe) if len(pe) else np.zeros(0,dtype=np.int32)
        pw=np.concatenate(pw) if len(pw) else np.zeros(0,dtype=np.float32)
        pYQ=pYQ_all.astype(np.float32)
        pDY=np.vstack(pDY) if pDY else np.zeros((0,D),dtype=np.float32)
        if ADDON and zo is not None:
            _k0,_di,_do,_z0,_U0,_V0=zo
            _Xq=(XQ0 if XAP else X0).astype(np.float32)
            if _di==3*D:
                _n=np.sqrt((_Xq*_Xq).mean(1,keepdims=True))+1e-6
                _Phi=np.concatenate([_Xq,(_Xq*_Xq)/_n,np.maximum(_Xq,0)],1)
            else: _Phi=_Xq
            dH -= (((_Phi@_V0)*_z0)@_U0.T).astype(np.float32)
            del _Phi
        np.savez(cache+".tmp.npz",dH=dH,prow=prow,pe=pe,pw=pw,pYQ=pYQ,pDY=pDY,**({"yqe":YQE, "xcap":X0} if XCAP else {}))
        os.replace(cache+".tmp.npz",cache)   # 原子换名: 并行 amp_solve 只见完整 zcache
        _batched_done=True
    else:
        _batched_done=False
    if _batched_done: pass
    elif True:
      for i,e in enumerate(need):
        rows,slots=np.where(ridx==e)
        if XAP: rowsq,slotsq=np.where(ridxq==e)
        else:   rowsq,slotsq=rows,slots
        if len(rows)==0 and len(rowsq)==0: continue
        P={}
        _lim=float(os.getenv("DS4_ZL_SWLIM","10"))
        for nm,wh in (("w1",0),("w3",1),("w2",2)):
            Wf=st_mxfp4_g(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
            if _GG: Wq=_gg_expert(L,nm,e)
            else:
                off=vq_slot(blob,e,wh); Wq=vq_dequant(blob,off) if off else None
            if _GPU and Wq is not None: Wq=cp.asarray(Wq)
            if Wq is not None and Wf.shape!=Wq.shape: Wf=Wf.T
            P[nm]=(Wf,Wq if Wq is not None else Wf)
        if len(rows)>0:   # FP 目标侧: FP 锚 x + FP 路由 + FP 权重
            xs=X0[rows].astype(np.float32); w=rw[rows,slots].astype(np.float32)
            if _GPU:
                xg=cp.asarray(xs)
                Yf=cp.asnumpy(_swiglu_g(xg@P["w1"][0].T,xg@P["w3"][0].T,_lim)@P["w2"][0].T)
            else:
                Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
            dH[rows]+=w[:,None]*Yf
        if len(rowsq)>0:  # 部署侧: 链态 x_q + 部署路由 + 量化权重(链模式=真部署口径)
            xq=(XQ0[rowsq] if XAP else X0[rowsq]).astype(np.float32)
            wq=(rwq[rowsq,slotsq] if XAP else rw[rowsq,slotsq]).astype(np.float32)
            if ADDON and ge_old is not None: wq=wq*ge_old[e]
            if _GPU:
                xqg=cp.asarray(xq)
                Yq=cp.asnumpy(_swiglu_g(xqg@P["w1"][1].T,xqg@P["w3"][1].T,_lim)@P["w2"][1].T)
            else:
                Yq=swiglu(xq@P["w1"][1].T,xq@P["w3"][1].T)@P["w2"][1].T
            if not XCAP: dH[rowsq]-=wq[:,None]*Yq
            prow.append(rowsq); pe.append(np.full(len(rowsq),e,dtype=np.int32))
            pw.append(wq); pYQ.append(Yq.astype(np.float32)); pDY.append(Yq.astype(np.float32)*0)
        if (i+1)%64==0: print(f"  L{L} 缓存 …{i+1}/{len(need)}", flush=True)
    if (not _batched_done) and ADDON and zo is not None:
        _k0,_di,_do,_z0,_U0,_V0=zo
        _Xq=(XQ0 if XAP else X0).astype(np.float32)
        if _di==3*D:
            _n=np.sqrt((_Xq*_Xq).mean(1,keepdims=True))+1e-6
            _Phi=np.concatenate([_Xq,(_Xq*_Xq)/_n,np.maximum(_Xq,0)],1)
        else: _Phi=_Xq
        dH -= (((_Phi@_V0)*_z0)@_U0.T).astype(np.float32)
        del _Phi
    if not _batched_done:
        if XCAP: dH-=YQE
        prow=np.concatenate(prow) if prow else np.zeros(0,dtype=np.int64)
        pe=np.concatenate(pe) if len(pe) else np.zeros(0,dtype=np.int32)
        pw=np.concatenate(pw) if len(pw) else np.zeros(0,dtype=np.float32)
        pYQ=np.vstack(pYQ) if pYQ else np.zeros((0,D),dtype=np.float32)
        pDY=np.vstack(pDY) if pDY else np.zeros((0,D),dtype=np.float32)
        np.savez(cache+".tmp.npz",dH=dH,prow=prow,pe=pe,pw=pw,pYQ=pYQ,pDY=pDY,**({"yqe":YQE, "xcap":X0} if XCAP else {}))
        os.replace(cache+".tmp.npz",cache)   # 原子换名: 并行 amp_solve 只见完整 zcache
if _CACHE_ONLY:
    print(f"★L{L} zcache-only: 就绪({time.time()-t0:.0f}s), 解算交外部", flush=True)
    sys.exit(0)
t1=time.time()
X=(XQ0 if XAP else X0).astype(np.float64)
_NF=int(os.getenv("DS4_ZL_NFIT","1287"))
# ★非连续fit区间(2026-08-09 拼接修正): DS4_ZL_FIT_RANGES="a:b,c:d" + DS4_ZL_EV_RANGE="a:b"
#   用于"上下文钉死只换解算池"的干扰实验; 不设=历史默认切分
_fr=os.getenv("DS4_ZL_FIT_RANGES"); _er=os.getenv("DS4_ZL_EV_RANGE")
if _fr:
    tr=np.concatenate([np.arange(int(a),int(b)) for a,b in (seg.split(":") for seg in _fr.split(","))])
    # EV 多段(2026-08-14 用户令"全能力不跑偏"): 组件门评审区支持跨域混合(如 prog+EN), 单段语法兼容
    ev=np.concatenate([np.arange(int(a),int(b)) for a,b in (seg.split(":") for seg in _er.split(","))])
else:
    tr=np.arange(0,_NF); ev=np.arange(_NF,NTOK)
colw=np.sqrt(dH[tr].var(0)+1e-12)
rms=np.sqrt((X[tr]**2).mean(1,keepdims=True))
r2=np.random.RandomState(1)
Xa=np.vstack([X[tr],(X[tr]+r2.randn(len(tr),D)*0.04*rms)*np.sqrt(0.25)])
Ra=np.vstack([dH[tr]*colw,(dH[tr]*colw)*np.sqrt(0.25)])
G=Xa@Xa.T; G[np.diag_indices_from(G)]+=3.0*np.trace(G)/Xa.shape[1]+1e-10
al=np.linalg.solve(G,Ra)
Wz=(Xa.T@al)/colw[None,:]
A,S,Bt=zl_svd(Wz)
# ★ftA 特征提升支线(2026-08-09 md86 战役: 非线性侧车, 部署=type6 din=3D)★
# φ(x)=[x, x⊙x/rms, relu(x)]; dither 在 x 空间打后过 φ; colw 感知加权与线性完全同款(四损失 parity)
# ★策略开关(2026-08-19 用户令"只留z变量/四损失/感知"): DS4_ZL_FTA=0 关非线性支线,
#   DS4_ZL_GE=0 关每专家门控(连同 GE-only 兜底), DS4_ZL_ERF=0 关死层部件(原有开关)。
_FTA=int(os.getenv("DS4_ZL_FTA","1"))
_GE_ON=int(os.getenv("DS4_ZL_GE","1"))
def zl_phi(M):
    n=np.sqrt((M*M).mean(1,keepdims=True))+1e-6
    return np.concatenate([M,(M*M)/n,np.maximum(M,0)],1).astype(np.float32)
if _FTA:
    Fa=zl_phi(Xa)
    Gf=Fa@Fa.T; Gf[np.diag_indices_from(Gf)]+=3.0*np.trace(Gf)/Fa.shape[1]+1e-10
    alf=np.linalg.solve(Gf,Ra)
    Wf=(Fa.T@alf)/colw[None,:]
    Af,Sf,Bf=zl_svd(Wf)
    del Fa,Gf,alf
    Phi_ev=zl_phi(X[ev].astype(np.float32))
else:
    Af=Sf=Bf=None; Phi_ev=None
Phi_all=None
# 活 k_L(产物③): held 段扫网格, 取达到 k=K 收益 95%(容差0.5点)的最小 k → 体积逐层动态
e0=float((dH[ev].astype(np.float64)**2).sum())
def rk(k):
    p=(X[ev]@(A[:,:k]*S[:k]))@Bt[:k]
    return 1-float(((dH[ev]-p.astype(np.float32)).astype(np.float64)**2).sum())/e0
def rkf(k):
    p=(Phi_ev@(Af[:,:k]*Sf[:k]))@Bf[:k]
    return 1-float(((dH[ev]-p.astype(np.float32)).astype(np.float64)**2).sum())/e0
KG=sorted(set([64,128,256,384,512,768,1024,1536,K]))
KG=[k for k in KG if k<=K]
KG=[k for k in KG if k<=1024]   # ★引擎帽(Metal pvS[1024]): 曲线在1024后平台≤0.2%, 帽死零风险
curve={k:rk(k) for k in KG}
curvef={k:rkf(k) for k in KG} if _FTA else {k:0.0 for k in KG}
rz_lin=max(curve.values()); rz_fta=max(curvef.values())
FORM="lin"
# ★收益闸(2026-08-19 用户令">0 即入"): 旧 1%挂载线/0.2%注入线退役, DS4_ZL_GATE 默认 0
_GATE=float(os.getenv("DS4_ZL_GATE","0.0"))
if max(rz_lin,rz_fta)<=_GATE:
    K=0; rz=0.0   # held 无正收益 → 本层不挂 z^L(深层位置非平稳保护)
elif rz_fta>rz_lin:
    FORM="ftA"; K=max(curvef,key=curvef.get); rz=curvef[K]
else:
    K=max(curve,key=curve.get); rz=curve[K]
print(f"  L{L} k曲线lin "+" ".join(f"{k}:{curve[k]*100:.1f}" for k in KG), flush=True)
if _FTA:
    print(f"  L{L} k曲线ftA "+" ".join(f"{k}:{curvef[k]*100:.1f}" for k in KG)+f" → 赢家={FORM} k_L={K}", flush=True)
else:
    print(f"  L{L} 纯z(ftA关) → k_L={K}", flush=True)
if K==0:
    R=dH
elif FORM=="ftA":
    Phi_all=zl_phi(X.astype(np.float32))
    R=dH-((Phi_all@(Af[:,:K]*Sf[:K]))@Bf[:K]).astype(np.float32)
else:
    R=dH-((X@(A[:,:K]*S[:K]))@Bt[:K]).astype(np.float32)
tok_pairs=[[] for _ in range(NTOK)]
for p in range(len(prow)): tok_pairs[prow[p]].append(p)
if _GE_ON:
    Gg=np.zeros((256,256)); bg=np.zeros(256)
    for t in tr:
        ps=tok_pairs[t]
        vecs=[(int(pe[p]),(pw[p]*pYQ[p]).astype(np.float64)) for p in ps]
        rt=R[t].astype(np.float64)
        for a_,(ea,va) in enumerate(vecs):
            bg[ea]+=va@rt
            for b_ in range(a_,len(vecs)):
                eb,vb=vecs[b_]; d=va@vb; Gg[ea,eb]+=d
                if ea!=eb: Gg[eb,ea]+=d
    _gelam=float(os.getenv("DS4_ZL_GE_LAM","1e-3"))   # ★GE收缩旋钮(2026-08-13 PPL过锐化根因): λ↑把增益往1收
    Gg[np.diag_indices_from(Gg)]+=_gelam*max(np.trace(Gg)/256,1.0)
    g=np.linalg.solve(Gg,bg)
    ge=(1.0+g).astype(np.float16)
    gf=ge.astype(np.float32)
    # 组合终验: z^L(K)+GE 在 held 段的总增益, ≤0.2% 整层不注入
    eng=0.0
    for t in ev:
        r=R[t].astype(np.float64).copy()
        for p in tok_pairs[t]:
            r-=(gf[pe[p]]-1.0)*(pw[p]*pYQ[p]).astype(np.float64)
        eng+=float((r**2).sum())
    comb=1-eng/e0
    # ★组件级门(2026-08-10 用户令)★: {纯z, z+GE} held 择优, GE 负贡献即弃; K=0 时唯一候选=GE-only
    USE_GE = (comb >= rz - 1e-9) or (K==0)
else:
    ge=np.ones(256,dtype=np.float16); gf=ge.astype(np.float32)
    comb=rz; USE_GE=False
eff = comb if USE_GE else rz
# ★分域终验(2026-08-09 用户令: 标的按域拆): held 前半=prog(3874:4303) 后半=fin(4303:4732) 各报
def _dom_recov(idx):
    if len(idx)==0: return float("nan")
    e0d=float((dH[idx].astype(np.float64)**2).sum())
    if e0d<=0: return float("nan")
    ed=0.0
    for t in idx:
        r=(dH[t] if K==0 else R[t]).astype(np.float64).copy()
        for p2 in tok_pairs[t]:
            r-=(gf[pe[p2]]-1.0)*(pw[p2]*pYQ[p2]).astype(np.float64)
        ed+=float((r**2).sum())
    return 1-ed/e0d
if len(ev)>=2:
    half=len(ev)//2
    print(f"  L{L} 分域组合: prog={_dom_recov(ev[:half])*100:.1f}%  fin={_dom_recov(ev[half:])*100:.1f}%  [组件门: z={rz*100:.1f} z+GE={comb*100:.1f} → {'z+GE' if USE_GE and K>0 else ('GE-only' if K==0 else '纯z')}]", flush=True)
t2=time.time()
def rec(nm,pay):
    h=bytearray(116); h[0:len(nm)]=nm.encode()
    struct.pack_into('<Q',h,88,len(pay)); struct.pack_into('<i',h,112,1)
    return bytes(h)+pay
# ★ERF死层部件(2026-08-12 用户架构令"修死层并入反修阶段, 不另起炉灶")★:
# 触发线(2026-08-13 用户令"低于1%都要"): 组合<1% 即上 ERF, 叠加在 z/GE 之上 —
# 拟合目标=扣除已中标 z 与 GE 效应后的真残差。ΔW_w2 加权SVD r 方向+α重加权+token能量门。
ERF_ADD=None; ERF_GAIN=0.0; ERF_NE=0
if (not ADDON) and eff<float(os.getenv("DS4_ZL_ERF_BAR","0.01")) and int(os.getenv("DS4_ZL_ERF","1")):
    try:
        def _rsvd(M,k):
            G=np.random.RandomState(11).randn(M.shape[1],k+8).astype(M.dtype)
            Q,_=np.linalg.qr(M@G)
            Ub,Sb,Vb=np.linalg.svd(Q.T@M,full_matrices=False)
            return (Q@Ub)[:,:k],Sb[:k],Vb[:k]
        _wmap=st_index(hf); _blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
        _X = XQ0 if XAP else X0
        _trm=np.zeros(NTOK,bool); _trm[tr]=True
        _evm=np.zeros(NTOK,bool); _evm[ev]=True
        # 叠加基残差: 扣除已中标 z(R=post-z)与 GE 增益效应, ERF 只修剩下的
        _R=(dH if K==0 else R).astype(np.float64).copy()
        if USE_GE:
            for _t in range(NTOK):
                for _p2 in tok_pairs[_t]:
                    _R[_t]-=(gf[pe[_p2]]-1.0)*(pw[_p2]*pYQ[_p2]).astype(np.float64)
        _pred=np.zeros((len(ev),D),np.float64)
        _evpos={int(t):i for i,t in enumerate(ev)}
        _r=int(os.getenv("DS4_ZL_ERF_R","8")); _payload=[]; _ne=0   # 死带加秩(2026-08-13): r16 未饱和实测2.2%
        _Rev0=_R[ev].copy()   # 叠加基残差的 ev 快照(ERF 增益的分母口径)
        _order=sorted(np.unique(pe).tolist(),key=lambda e:-int((pe==e).sum()))[:128]
        for _e in _order:
            _idx=np.where(pe==_e)[0]
            _ftr=_idx[_trm[prow[_idx]]]; _fev=_idx[_evm[prow[_idx]]]
            if len(_ftr)<24 or len(_fev)<2: continue
            _o1=vq_slot(_blob,_e,0); _o3=vq_slot(_blob,_e,1); _o2=vq_slot(_blob,_e,2)
            if not (_o1 and _o3 and _o2): continue
            _Wf=st_mxfp4(hf,_wmap,f"layers.{L}.ffn.experts.{_e}.w2.weight").astype(np.float64)
            _Wq=vq_dequant(_blob,_o2).astype(np.float64)
            if _Wf.shape!=_Wq.shape: _Wf=_Wf.T
            _W1q=vq_dequant(_blob,_o1); _W3q=vq_dequant(_blob,_o3)
            _xt=_X[prow[_ftr]].astype(np.float64); _xe=_X[prow[_fev]].astype(np.float64)
            if _W1q.shape[1]!=_xt.shape[1]: _W1q=_W1q.T
            if _W3q.shape[1]!=_xt.shape[1]: _W3q=_W3q.T
            _ht=swiglu(_xt@_W1q.T,_xt@_W3q.T); _he=swiglu(_xe@_W1q.T,_xe@_W3q.T)
            _sh=np.sqrt((_ht**2).mean(0))+1e-8
            _dW=_Wf-_Wq
            _U,_S,_V=_rsvd((_dW*_sh[None,:]).astype(np.float32),_r)
            _U=_U.astype(np.float64); _S=_S.astype(np.float64); _V=_V.astype(np.float64)/_sh[None,:]
            _wt=pw[_ftr].astype(np.float64); _we=pw[_fev].astype(np.float64)
            _Gt=(_ht@_V.T)*_wt[:,None]; _Ge=(_he@_V.T)*_we[:,None]
            _Rr=_R[prow[_ftr]]
            _A=(_Gt.T@_Gt)*(_U.T@_U)*np.outer(_S,_S)
            _b=_S*np.einsum("ni,nd,di->i",_Gt,_Rr,_U)
            _a=np.linalg.solve(_A+(np.trace(_A)/_r+1e-12)*np.eye(_r),_b)
            _ct=(_Gt*(_a*_S)[None,:])@_U.T
            _ce=(_Ge*(_a*_S)[None,:])@_U.T
            _en=(_ct**2).sum(1); _ben=2*(_Rr*_ct).sum(1)-_en
            _oi=np.argsort(-_en); _cum=np.cumsum(_ben[_oi]); _m=int(_cum.argmax())+1
            if _cum[_m-1]<=0: continue
            _tau=float(_en[_oi[_m-1]])
            _ct=_ct*(_en>=_tau)[:,None]; _ce=_ce*(((_ce**2).sum(1))>=_tau)[:,None]
            if float(2*(_Rr*_ct).sum()-(_ct**2).sum())<=0: continue
            _R[prow[_ftr]]-=_ct
            for _j,_i2 in enumerate(_fev): _pred[_evpos[int(prow[_i2])]]+=_ce[_j]
            _Ue=np.ascontiguousarray((_U*_S).astype(np.float16))          # D×r, 折S; 应用=w·(h@Vᵀ)@Uᵀ
            _Ve=np.ascontiguousarray((_a[:,None]*_V).astype(np.float16))  # r×F, 折α
            _payload.append(struct.pack('<If',_e,_tau)+_Ue.tobytes()+_Ve.tobytes())
            _ne+=1
        if _ne:
            _e0a=float((_Rev0.astype(np.float64)**2).sum())
            ERF_GAIN=1-float(((_Rev0.astype(np.float64)-_pred)**2).sum())/max(_e0a,1e-18)
            if ERF_GAIN>0.002:
                ERF_ADD=rec('zl.ERF',struct.pack('<IHH',_ne,_r,0)+b"".join(_payload))
                ERF_NE=_ne
        _tot=eff+max(ERF_GAIN,0.0)*(1-eff)
        print(f"  L{L} ERF死层部件: 专家={_ne} 残差挽回={ERF_GAIN*100:.1f}% 组合总计={_tot*100:.1f}% → {'注入' if ERF_ADD is not None else '不过闸'}",flush=True)
    except Exception as _ex:
        print(f"  L{L} ERF死层部件异常: {_ex}",flush=True)
if ADDON:
    # ★叠加式合并注入★: 新Δ与既有记录合并成单条(引擎"末条胜出"安全)
    ge_new = gf if USE_GE else np.ones(256,dtype=np.float32)
    ge_base = ge_old if ge_old is not None else np.ones(256,dtype=np.float32)
    ge_m = (ge_base*ge_new).astype(np.float16)
    DIW = 3*D if ((zo is not None and zo[1]==3*D) or (K>0 and FORM=="ftA")) else D
    def _lift(Vm,di):
        if di==DIW: return Vm
        out=np.zeros((DIW,Vm.shape[1]),dtype=np.float64); out[:di]=Vm; return out
    P_list=[]; Q_list=[]
    if zo is not None:
        _k0,_di,_do,_z0,_U0,_V0=zo
        P_list.append(_lift(_V0.astype(np.float64)*_z0,_di)); Q_list.append(_U0.astype(np.float64))
    if K>0:
        if FORM=="ftA": Vn=(Af[:,:K]*Sf[:K]).astype(np.float64); Un=Bf[:K].T.astype(np.float64)
        else:           Vn=(A[:,:K]*S[:K]).astype(np.float64);   Un=Bt[:K].T.astype(np.float64)
        P_list.append(_lift(Vn,3*D if FORM=="ftA" else D)); Q_list.append(Un)
    if P_list:
        Pc=np.concatenate(P_list,1); Qc=np.concatenate(Q_list,1)   # M = Pc @ Qc.T
        Qp,Rp=np.linalg.qr(Pc); Qq,Rq=np.linalg.qr(Qc)
        Am,Sm,Bm=zl_svd(Rp@Rq.T)
        km=min(1024,int((Sm>1e-8).sum()) or 1)
        Vm=(Qp@Am[:,:km]).astype(np.float16)          # (DIW,km) 含奇异值前吸收? 需 diag 拆分:
        zm=Sm[:km].astype(np.float16)
        Um=(Qq@Bm.T[:,:km]).astype(np.float16)        # (D,km)
        add=rec('bf.GE',ge_m.tobytes())
        add+=rec('zl.RRR',struct.pack('<IfII',km,0.5,DIW,D)+zm.tobytes()
                 +np.ascontiguousarray(Um).tobytes()+np.ascontiguousarray(Vm).tobytes())
        nrec_add=2
        K_report=km
    else:
        add=rec('bf.GE',ge_m.tobytes()); nrec_add=1; K_report=0
else:
    add=rec('bf.GE',ge.tobytes()) if USE_GE else b''
    if K>0:
        if FORM=="ftA":   # ★din=3D: 引擎/回放按 din 现场构建 φ(x), 载荷布局与线性完全同构
            V16=Af[:,:K].astype(np.float16); z16=Sf[:K].astype(np.float16); U16=np.ascontiguousarray(Bf[:K].T).astype(np.float16)
            add+=rec('zl.RRR',struct.pack('<IfII',K,0.5,3*D,D)+z16.tobytes()+U16.tobytes()+V16.tobytes())
        else:
            V16=A[:,:K].astype(np.float16); z16=S[:K].astype(np.float16); U16=np.ascontiguousarray(Bt[:K].T).astype(np.float16)
            add+=rec('zl.RRR',struct.pack('<IfII',K,0.5,D,D)+z16.tobytes()+U16.tobytes()+V16.tobytes())
    nrec_add=(1 if USE_GE else 0)+(1 if K>0 else 0)
status="解算完"
if INJ==2:   # 外挂模式(2026-08-09 q4): 记录落 zrec 文件(dql 被清道夫清, 部署走外挂 zchain)
    zp=os.path.join(ld,f"zrec_L{L:02d}.bin")
    if eff<=_GATE:
        open(zp,'wb').write(b"")
        status=f"组合增益 {comb*100:.1f}% ≤闸{_GATE*100:.1f}% → 空 zrec(skip 标记)"
    else:
        open(zp,'wb').write(add)
        status=f"zrec 落盘(+{nrec_add}记录 {len(add)/2**20:.1f}MB)"
elif INJ:
    import fcntl
    man=os.path.join(ld,"zinject_manifest.txt")
    _lk=open(man+".lock","a"); fcntl.flock(_lk,fcntl.LOCK_EX)   # ★双路并发: 账本读写全程持锁
    done=set()
    if os.path.exists(man):
        for ln in open(man):
            done.add(int(ln.split()[0]))
    if ADDON:
        # ★叠加式落地★: 闸拒=贪心原样保留(零动作); 落地=截回账本原尺寸(去旧记录)再写合并记录
        if eff<=_GATE:
            status=f"Δ增益 {eff*100:.1f}% ≤闸{_GATE*100:.1f}% → 保留贪心原记录(叠加闸)"
        else:
            ent=None
            if os.path.exists(man):
                for ln in open(man):
                    pp=ln.split()
                    if int(pp[0])==L and int(pp[1])>0: ent=(int(pp[1]),int(pp[2]))
            dql=os.path.join(ld,f"dql_L{L:02d}.bin")
            if ent:   # 有贪心账 → 截回裸底座再写合并
                osz,n0=ent
                f=open(dql,'r+b'); f.truncate(osz); f.seek(8); f.write(struct.pack('<I',n0))
                f.seek(0,2); f.write(add)
                f.seek(8); f.write(struct.pack('<I',n0+nrec_add)); f.close()
                status=f"合并注入完(+{nrec_add}记录, 基于原账 {osz})"
            else:     # 无贪心账(该层此前 skip) → 常规追加+记账
                osz=os.path.getsize(dql)
                f=open(dql,'r+b'); f.seek(8); n0=struct.unpack('<I',f.read(4))[0]
                f.seek(0,2); f.write(add)
                f.seek(8); f.write(struct.pack('<I',n0+nrec_add)); f.close()
                open(man,'a').write(f"{L} {osz} {n0}\n")
                status=f"合并注入完(+{nrec_add}记录, 新账 {osz})"
    elif L in done:
        status="已注入过, 跳过(回滚请按账本截断)"
    elif eff<=_GATE:
        if ERF_ADD is not None:   # ★死层部件接管: z拒→ERF过闸, 同账本注入
            dql=os.path.join(ld,f"dql_L{L:02d}.bin")
            osz=os.path.getsize(dql)
            f=open(dql,'r+b'); f.seek(8); n0=struct.unpack('<I',f.read(4))[0]
            f.seek(0,2); f.write(ERF_ADD)
            f.seek(8); f.write(struct.pack('<I',n0+1)); f.close()
            open(man,'a').write(f"{L} {osz} {n0}\n")
            status=f"ERF死层注入(+1记录 {len(ERF_ADD)/2**20:.1f}MB, held+{ERF_GAIN*100:.1f}%, 专家{ERF_NE})"
        else:
            open(man,'a').write(f"{L} -1 -1\n")
            status=f"组合增益 {comb*100:.1f}% ≤闸{_GATE*100:.1f}% → 本层不注入(闸)"
    else:
        if ERF_ADD is not None:   # ★叠加(2026-08-13 用户令"<1%都要"): z/GE 之上追加 ERF 残差补丁
            add+=ERF_ADD; nrec_add+=1
        dql=os.path.join(ld,f"dql_L{L:02d}.bin")
        osz=os.path.getsize(dql)
        f=open(dql,'r+b'); f.seek(8); n0=struct.unpack('<I',f.read(4))[0]
        f.seek(0,2); f.write(add)
        f.seek(8); f.write(struct.pack('<I',n0+nrec_add)); f.close()
        open(man,'a').write(f"{L} {osz} {n0}\n")
        status=f"注入完(+{nrec_add}记录, 原长 {osz} 入账本)"
t3=time.time()
print(f"★L{L} z侧车: held挽回 z^L {rz*100:.1f}% 组合 {comb*100:.1f}%  GE均值 {float(ge.mean()):.4f}  "
      f"体积 {len(add)/2**20:.1f}MB | 缓存 {t1-t0:.0f}s 解算 {t2-t1:.0f}s 总 {t3-t0:.0f}s | {status}", flush=True)

# -*- coding: utf-8 -*-
"""在赛事模板上填充问题定义正文 + 配图，输出 4 页内 docx。"""
import copy, os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = "/Users/fodelf/git/ds4-main/开放探索赛初赛问题定义大纲模板_4页内.docx"
OUT = os.environ.get("OUT_DOCX", os.path.join(HERE,
      "开放探索赛初赛问题定义_极低比特量化的质量上界与损失归属.docx"))


def IMG(name):
    """图片优先取 figs/ 子目录，兼容与脚本同级的情形。"""
    for cand in (os.path.join(HERE, "figs", name), os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(name)

TITLE = "AI for Research 赛道｜开放探索赛初赛问题定义文档"
SUBTITLE = "极低比特量化的质量上界与损失归属"
SUBTITLE2 = ("——等体积、零训练约束下，"
             "以开源社区同档 2-bit 量化件为参照系的自主探索")

ONELINE = ("一句话问题：给定与开源社区已发布的 DeepSeek-V4-Flash 2-bit 量化件"
           "（IQ2_XXS 档，86.7 GB）完全相同的体积预算，"
           "并且全程不允许训练，同一个大模型还能被量化到多好？"
           "等体积下还剩多少质量没被榨干、它藏在哪个环节、能不能兑现成「可用」"
           "——本文把它定义成一个 AI Agent 可长期自主探索的环境。")

# ---- 正文（键 = 模板中该"应说明"段落的索引）----
BODY = {
7: [  # 1.1 真实问题或需求
 ("b", "目标与对象。 目标模型是 DeepSeek-V4-Flash（284B 总参数、13B 激活的 MoE）。"
       "本地与私有部署必须用量化模型，而 2-bit 档是能在可负担硬件上真正跑起来的档位，"
       "也是开源社区为此发布量化件的档位。用它的人是个人研究者、中小团队，"
       "以及数据不能出内网的组织。"),
 ("b", "我要的不是更小，是同样大小、更好用。 体积预算取自社区已发布的同档量化件"
       "（IQ2_XXS 档 86.7 GB）；目标不是把模型压得更小，而是在这个固定体积上把质量"
       "做到最好——超过社区同档件，并且好到「可用」。"),
 ("b", "困扰的精确形态。 社区件本身已是很强的基线（wikitext-2 上 KL 0.4207／top-1 77.92%）；"
       "要在同体积、不训练、且不靠裁剪的条件下超过它没有现成方法，"
       "而且判据一旦选错，很容易误以为已经赢了（见 1.2）。"),
 ("b", "所以问题是： 体积一分不加、且不训练，能否在一把不能被这样刷的尺子上"
       "——与校准语料零重叠的公开通用文本——全面超过社区件。"),
],
9: [  # 1.2 为什么尚未被结构化
 ("b", "不是「没人做过极低比特」。 业界 1-bit、1.58-bit 工作已经成熟，"
       "但绝大多数依赖训练或量化感知训练；在「拿到已发布权重、不训练、"
       "只做后训练量化」的条件下同体积能做到多好，没有公认答案。"),
 ("b", "缺的是上界，不是又一个方法。 现有工作报的都是「我的方法比某基线好多少」，"
       "没有人回答「这个体积预算下最好能到多少、当前方法离它还有多远」。"
       "没有上界就无法判断该继续投入还是换方向——这正是本探索最需要的判据。"),
 ("b", "损失藏在哪里没有定论。 直觉是「量化损失＝权重重建误差」，补偿理应加在输出侧。"
       "本探索用三个互证的作弊上界实验证明这个直觉是错的（图 1②、3.1②）——"
       "这类问题从论文指标里读不出来，只能自己做上界实验。"),
 ("b", "度量本身不可信。 同一套方案换一批判据集，能力保留率 0.90 → 0.52（图 1③）。"
       "而且校准语料与判据同域会造成系统性假象：已实测，私有语料上很漂亮的成绩"
       "换到通用文本上完全不迁移。"),
 ("b", "搜索空间大而反馈慢。 配置组合数以万计、无理论可预判，"
       "单次完整验证以小时计——人力穷举不现实。"),
],
11: [  # 1.3 研究价值与合适切片
 ("b", "上界价值。 回答「这个体积预算下最好能到多少、当前离它多远」："
       "目前距社区件的剩余 KL 差距 0.032，而单是路由一项的作弊上界就值 0.070。"
       "上界告诉你还有多少可拿，这是决定继续投入还是换方向的依据。"),
 ("b", "归属价值。 把量化损失定位到具体环节，决定整个方向该往哪里投入。"
       "本探索已把它定位到路由选择侧而非权重重建侧，并产出一个可推广的方法："
       "oracle 上界法（3.1②）。"),
 ("b", "判据价值。 给出一把不能被「换个容易判据集」或「让校准语料与判据同域」刷高的尺子——"
       "这是本地部署选型时最缺的东西。"),
 ("b", "AI 为什么可以介入。 探索循环高度结构化却极其繁琐："
       "提假设→落成配置→量化出候选→跑判决→读日志→下结论→改假设，"
       "单轮以小时计、已累计上百轮。Agent 提供三件人做不好的事："
       "不疲劳地扫描上百个配置并归纳；把每次失败完整记录成可检索证据；"
       "在负结果出现时如实判否，而不是换个指标继续证明自己。"),
 ("b", "合适切片。 不做通用量化理论，只切一个可闭环的切片：单一模型、单一体积档、零训练、"
       "公开通用判据集，以对未量化模型的逼近程度为终判。"),
],
16: [  # 2.1 固定规则
 ("b", "体积冻结。 候选必须与社区参照件同档（86.7 GB）。不允许加体积换质量——"
       "这是本问题的定义性约束，一旦放开，问题就退化成「更大的模型当然更好」。"),
 ("b", "零训练。 只允许后训练量化与闭式求解，不许重训、不许量化感知训练。"
       "这条决定了本探索与业界多数极低比特工作处在不同的约束面上。"),
 ("b", "判据集冻结。 判据取公开文本，与校准语料零重叠；生成设置与判定脚本开跑前冻结并"
       "版本留档（防止「打不过就换指标」）。参照件为社区已发布的同档量化件，"
       "其公开数字可被第三方逐项复算。"),
 ("b", "全层全专家平权。 不做专家裁剪、不丢弃任何部件、不按域倾斜本体的比特预算——"
       "裁剪换来的分数会把偏置固化进权重，与「还原模型本身」相悖。"),
],
18: [  # 2.2 观察/行动/反馈
 ("b", "可观察。 候选相对未量化模型的 KL 散度、top-1 一致率与困惑度比；逐层误差及其"
       "沿层链的传播；判据集上的逐 token 原始输出；耗时与内存账单；全部历史实验记录。"),
 ("b", "可执行。 在等体积约束下腾挪各层各部位的预算；附加或移除等体积的修正；"
       "改变校准语料的构成与摆位；构造 oracle 上界实验（允许作弊，只用于定位）；"
       "以及——申请修改假设或问题边界。"),
 ("b", "反馈。 ①过程诊断指标；②终判信号，即通用尺三项指标对社区件的胜负；"
       "③资源与耗时账单；④失败现场，含原始输出与日志。"),
 ("b", "一条关键约定。 过程诊断指标只在同一方案内部用于定位，不得跨方案比较、"
       "不得当作成绩——曾用未过终判方案的中间指标去比已过终判的方案，整轮结论作废。"),
],
20: [  # 2.3 记录与预算
 ("b", "状态与日志。 每次实验（成功与失败同等）写入统一日志，含假设、配置、原始输出、"
       "判决与「本次学到什么」；候选按轮次归档。负结果另立清单——它是已关闭分支的地图，"
       "也是本探索的主要累积资产。"),
 ("b", "资源预算。 单次验证 ≤ 10 分钟——任何方案必须先具备分钟级判决设计才允许启动；"
       "量化全程看门狗守内存红线；每轮必须产出一个可陈述的结论，包括负结论。"),
],
24: [  # 3.1 什么算发现
 ("b", "①正向发现。 等体积、零训练下相对社区件可复现的提升。实例：仅改变校准语料的"
       "组织方式（不加体积、不加训练、不改结构），未加修正的量化本体 KL 四连降"
       "0.5547→0.5335→0.5134→0.4956；再叠加等体积修正后到 0.4522，为当前最好。"),
 ("b", "②上界与归属发现（本探索最主要的产出形态）。 方法是 oracle 上界法：给某个环节"
       "一个允许作弊的上界（用未量化模型的真值替该环节做决策），看端到端能回收多少。"
       "三个互证实例——输出残差侧：允许用真值检索补偿，回收 ≈ 0 且容量再大也永不转正，"
       "判定该处信息不存在；深层复验：对深层残差再做分域、条件化与非线性共十二种变体，"
       "全部 ≈ 0，重建侧在深层已穷尽；路由选择侧：把选择钉在真值上，"
       "KL 由 0.4522 降到 0.3823（−15.5%），三项指标全面越过社区件。"
       "结论：量化损失的主体不在「权重重建得不够准」，而在「它扰动了模型的内部选择」。"
       "路由这一项值 0.070 KL，而剩余全部差距只有 0.032——兑现一半即可全面反超。"),
 ("b", "③对角线定律与域适配的位置。 早期实测某改动在专业域内 +32%、换到通用域只剩 3%"
       "＝域内记忆效应。进一步测出定律：每个域的最优解都是本域校准（51.6%／28.5%／21.1%），"
       "无单一校准能通吃。据此得到的原则不是「挑一个域去刷」，而是让域适配与量化本体解耦、按域可替换："
       "链上实测专业域 KL 0.386→0.205（−47%）、top-1 85.6→88.3，"
       "通用尺仅由 0.452 微降到 0.480、仍远胜未加修正的 0.496，且全程不改动本体权重。"),
 ("b", "④对原问题定义的修正。 原目标是刷高某个专业域的分数，实测证明那是域内记忆效应，以通用能力换来的域内分数没有意义。修正后分两层：通用能力是基础，以通用尺为主判、不为任何单项域让步；在此前提下专业域的提升与本体解耦、按域可替换，不写进本体权重。这类修正记为发现而非返工——它同时改变了目标与架构。"),
],
26: [  # 3.2 平凡解/随机/无干预
 ("b", "三个参照。 平凡解＝开源社区已发布的 DeepSeek-V4-Flash 同档 2-bit 量化件"
       "（IQ2_XXS 档 86.7 GB），wikitext-2 上的公开数字为 KL 0.4207／top-1 77.92%／"
       "困惑度比 1.3575，第三方可逐项复算；随机参照＝把等体积预算随机分配的配置；"
       "无干预上界＝未量化模型，三项指标都对它测。"),
 ("b", "口径说明。 我方为连续单流回放、社区公开表为 ctx512 分块跑全测试集，故困惑度绝对值"
       "不可直接比，对未量化模型的相对指标可比；判据集与校准语料零重叠。"),
 ("b", "当前所处位置（诚实报账）。 top-1 78.78 对 77.92——反超 0.86 点；"
       "但 KL 0.4522 对 0.4207、困惑度比 1.430 对 1.3575 仍落后（图 1①）。"
       "形态是「主峰已胜、分布尾部仍有差距」，剩余 KL 差距 0.032，"
       "已由 3.1② 归属完毕。"),
 ("b", "何种差异才算超过参照。 必须在与校准语料零重叠的公开通用判据集上，"
       "三项相对社区件不倒退且至少一项可复现地超过。"
       "只在容易判据集或单一领域上的提升一律不采信——图 1③ 与 3.1③ 已两次证明其为假象。"),
],
28: [  # 3.3 最低成功与失败标准
 ("b", "最低成功门槛（同时满足）。 ①体积不超社区件、全程零训练、全层全专家平权；"
       "②通用尺三项指标相对社区件不倒退，且至少一项可复现地超过（top-1 已达成）；"
       "③终极门槛是三项全面超过。专业域的额外提升算加分，不能顶替这三条。"),
 ("b", "明确失败判定。 ①靠增加体积赢；②靠训练或量化感知训练赢；"
       "③靠校准语料与判据同域赢（换域即失效）；"
       "④以牺牲通用能力或裁剪部件为代价换取单项领域分数——这类「赢」没有意义，直接判失败；"
       "⑤提升不可复现，或只在单一判据集上出现。"),
 ("b", "据此如何更新。 失败先做归因消融，区分「方法无效」「信息不存在」"
       "与「度量或流程被污染」。判为信息不存在的写入负结果清单永久关闭；"
       "判为方法无效的只关闭该实现分支；若失败点指向问题定义本身，则更新边界并重述目标。"),
],
32: [  # 4.1 一次试跑怎么做
 ("b", "目标。 一轮之内判定：某个等体积改动，能否把三项指标朝社区件的方向推进一步。"),
 ("b", "输入。 冻结的公开判据集；未量化模型在同一输入上的参考输出；"
       "社区同档件的公开数字；待验证配置。"),
 ("b", "步骤。 ①在少量代表性层上做小规模扫描取参数甜点（分钟级）；"
       "②按甜点量化出完整候选件；③通用尺终判——在与校准语料零重叠的公开文本上，"
       "取候选对未量化模型的 KL／top-1／困惑度比，与社区件逐项对照；"
       "④若判负，做归因消融定位是方法、信息还是流程问题。"),
 ("b", "输出与关键证据。 三方对照表（候选／社区件／未量化上界）；"
       "判据集上的逐 token 原始记录（不摘编）；耗时与内存账单；一句话判决。"
       "图 2 给出该闭环的完整接口。"),
],
34: [  # 4.2 主要风险与失败路径
 ("b", "判据风险。 宽松判据造成虚高（0.90 对 0.52）；校准语料与判据同域也会让成绩虚高"
       "且不迁移（均已实测）。对策：主判固定在与校准语料零重叠的公开通用文本上，"
       "宽松指标只作参考。"),
 ("b", "上界误读风险。 把「方法无效」当成「信息不存在」，或反过来。对策：上界实验须随"
       "容量单调外推，并与另一环节的上界互证后才下判。"),
 ("b", "链传播风险。 局部账变好、整条链反而更差（已实测一次净负并回滚）。"
       "对策：一律以链上端到端指标判决，不采信逐层局部账。"),
 ("b", "域增益的误用。 专业域语料能在本域拿到数倍增益（51.6% 对 21.1%），但一旦写进本体权重"
       "或用来倾斜比特预算就会固化偏置。对策：域适配必须与本体解耦、可按域替换，"
       "本体与主判始终是通用的。"),
 ("b", "失败后如何处理。 保留现场 → 归因消融 → 写入日志并明确「关闭分支」或「改假设」，"
       "下一轮不重复。"),
],
36: [  # 4.3 复现与开源计划
 ("b", "数据来源。 DeepSeek-V4-Flash 开源权重（量化对象与真值参照）、"
       "社区已发布的同档 2-bit 量化件（参照系）与 wikitext-2（判据集）；"
       "三者均为公开，不使用任何非公开数据。"),
 ("b", "依赖工具与复现方式。 自建的本地量化与评测工具链，全程本地运行、不依赖云服务；"
       "量化出候选与取三项指标各由一条命令完成。全部实验（含失败）带配置与原始输出归档、"
       "按轮次可检索；判据集与判定脚本随文档公开。"),
 ("b", "开源计划。 初赛后开源评测协议、判据集构造方法、oracle 上界法实现与全部实验日志"
       "（含负结果清单）；量化与评测工具链在赛程内分阶段开源。"),
],
}

# 已关闭分支表（放 3.1 之后）
TABLE_ROWS = [
    ("已关闭的搜索分支", "关闭依据（多规模一致复现）"),
    ("输出侧补偿量化残差", "作弊上界回收 ≈ 0 且永不转正 → 该处信息不存在（任意容量、线性与非线性）"),
    ("深层残差的分域／条件化补偿", "十二种变体全部 ≈ 0 → 重建侧在深层已穷尽，损失不在此处"),
    ("改用另一优化目标重解", "不增容量只挪容量，在自家口径上反输 → 过拟合形态"),
    ("按某一域倾斜本体比特预算", "域内 +32% 换到通用域仅剩 3%＝把域偏置固化进权重"),
]


def set_run(r, size=10.5, bold=False, color=None, name="Times New Roman",
            east="宋体"):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = name
    rPr = r._element.get_or_add_rPr()
    rFonts = rPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
    if rFonts is None:
        from docx.oxml.ns import qn
        rFonts = rPr.makeelement(qn('w:rFonts'), {})
        rPr.append(rFonts)
    from docx.oxml.ns import qn
    rFonts.set(qn('w:eastAsia'), east)
    if color:
        r.font.color.rgb = color


def style_para(p, space_after=3, line=1.0, first_indent=None):
    pf = p.paragraph_format
    pf.space_before = Pt(0.5)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent is not None:
        pf.first_line_indent = Cm(first_indent)


def clear_para(p):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)


def fill(p, chunks):
    """chunks: 列表[(bold_text, normal_text)] 或 纯文本"""
    clear_para(p)
    if isinstance(chunks, str):
        chunks = [(None, chunks)]
    for b, t in chunks:
        if b:
            r = p.add_run(b)
            set_run(r, 10.0, bold=True)
        if t:
            r = p.add_run(t)
            set_run(r, 10.0)


def new_para_after(doc, anchor_p):
    np = copy.deepcopy(anchor_p._p)
    for child in list(np):
        if child.tag.endswith('}r'):
            np.remove(child)
    anchor_p._p.addnext(np)
    from docx.text.paragraph import Paragraph
    return Paragraph(np, anchor_p._parent)


def main():
    doc = Document(TPL)
    sec = doc.sections[0]
    sec.left_margin = Cm(1.65)
    sec.right_margin = Cm(1.65)
    sec.top_margin = Cm(1.4)
    sec.bottom_margin = Cm(1.25)
    sec.footer_distance = Cm(0.65)
    ps = doc.paragraphs

    # --- 标题区 ---
    p0 = ps[0]
    clear_para(p0)
    r = p0.add_run(TITLE)
    set_run(r, 15, bold=True, east="黑体")
    p0.paragraph_format.space_after = Pt(2)
    p0.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p1 = ps[1]
    clear_para(p1)
    r = p1.add_run(SUBTITLE)
    set_run(r, 13, bold=True, east="黑体")
    p1.paragraph_format.space_before = Pt(3)
    p1.paragraph_format.space_after = Pt(1)
    p1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p1b = new_para_after(doc, p1)
    clear_para(p1b)
    r = p1b.add_run(SUBTITLE2)
    set_run(r, 10, east="黑体")
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    p1b.paragraph_format.space_after = Pt(5)
    p1b.paragraph_format.first_line_indent = Cm(0)
    p1b.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 段落2 = 一句话问题；段落3、4（赛事要求）删除
    p2 = ps[2]
    clear_para(p2)
    r = p2.add_run(ONELINE)
    set_run(r, 10, east="楷体")
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    style_para(p2, space_after=4.5, line=1.0)
    pPr = p2._p.get_or_add_pPr()
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    pbdr = OxmlElement('w:pBdr')
    for side in ('top', 'bottom'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4')
        el.set(qn('w:space'), '3'); el.set(qn('w:color'), 'BFBFBF')
        pbdr.append(el)
    pPr.append(pbdr)

    for idx in (3, 4):
        ps[idx]._p.getparent().remove(ps[idx]._p)

    # --- 正文填充 ---
    for idx, blocks in BODY.items():
        p = ps[idx]
        first = True
        anchor = p
        for kind, text in blocks:
            if first:
                target = p
                first = False
            else:
                target = new_para_after(doc, anchor)
            if kind == "b" and "。 " in text:
                head, rest = text.split("。 ", 1)
                fill(target, [(head + "。", rest)])
            else:
                fill(target, text)
            style_para(target, space_after=1.2, line=1.0, first_indent=0.0)
            anchor = target

    # 删除模板里多余空段
    for idx in (12, 13, 21, 29, 37):
        try:
            ps[idx]._p.getparent().remove(ps[idx]._p)
        except Exception:
            pass

    # 缩紧标题间距
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            p.paragraph_format.space_before = Pt(4.5)
            p.paragraph_format.space_after = Pt(1.5)
            for r in p.runs:
                set_run(r, 13, bold=True, east="黑体")
        elif p.style.name == "Heading 2":
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                set_run(r, 11, bold=True, east="黑体")

    def add_figure(anchor_p, img, width_cm, caption):
        pic_p = new_para_after(doc, anchor_p)
        clear_para(pic_p)
        pic_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_p.paragraph_format.space_before = Pt(2.5)
        pic_p.paragraph_format.space_after = Pt(1)
        pic_p.paragraph_format.first_line_indent = Cm(0)
        pic_p.add_run().add_picture(img, width=Cm(width_cm))
        cap_p = new_para_after(doc, pic_p)
        clear_para(cap_p)
        cap_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_p.paragraph_format.space_after = Pt(3)
        cap_p.paragraph_format.first_line_indent = Cm(0)
        r = cap_p.add_run(caption)
        set_run(r, 9)
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        return cap_p

    # 图2 -> 1.2 之后； 图1 -> 2.3 之后
    def last_para_of(section_head_text):
        found = False
        last = None
        for p in doc.paragraphs:
            if p.style.name in ("Heading 1", "Heading 2"):
                if found:
                    break
                found = (p.text.strip() == section_head_text)
                continue
            if found and p.text.strip():
                last = p
        return last

    a = last_para_of("1.2 为什么尚未被结构化")
    add_figure(a, IMG("fig1_evidence.png"), 14.4,
               "图 1　DeepSeek-V4-Flash 等体积（86.7 GB 档）、零训练下的三组关键实测："
               "①与社区同档 2-bit 量化件相比，"
               "我方 top-1 已反超、KL 与困惑度比仍落后；②作弊上界定位——"
               "补偿输出残差侧回收 ≈ 0，而钉住路由选择侧 −15.5% 并全面越过社区件，"
               "说明损失主体在「选择」而非「重建」；③同一方案换一批判据集，结论相差一倍以上。")

    a = last_para_of("2.3 记录与预算")
    add_figure(a, IMG("fig2_loop.png"), 11.0,
               "图 2　探索环境接口与一轮最小闭环：左栏为四条不可改的规则"
               "（体积冻结／零训练／判据集冻结／全层全专家平权），"
               "①–⑦ 为一轮完整的观察—行动—反馈—记录回路。")

    # 已关闭分支表 -> 3.1 之后
    a = last_para_of("3.1 什么算发现")
    tbl = doc.add_table(rows=0, cols=2)
    tbl.style = "Table Grid"
    for i, (c1, c2) in enumerate(TABLE_ROWS):
        row = tbl.add_row()
        for cell, txt in zip(row.cells, (c1, c2)):
            cp = cell.paragraphs[0]
            clear_para(cp)
            r = cp.add_run(txt)
            set_run(r, 9, bold=(i == 0))
            cp.paragraph_format.space_before = Pt(1)
            cp.paragraph_format.space_after = Pt(1)
            cp.paragraph_format.line_spacing = 1.0
    tbl.columns[0].width = Cm(5.0)
    tbl.columns[1].width = Cm(12.6)
    for row in tbl.rows:
        row.cells[0].width = Cm(5.0)
        row.cells[1].width = Cm(12.6)
    a._p.addnext(tbl._tbl)
    cap = new_para_after(doc, a)
    clear_para(cap)
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(1)
    cap.paragraph_format.first_line_indent = Cm(0)
    r = cap.add_run("表 1　已产出的稳定负结果（节选）——负结果同样是发现，其价值是关闭搜索分支：")
    set_run(r, 9, bold=True)

    doc.save(OUT)
    print("saved:", OUT)
    d2 = Document(OUT)
    n = sum(len(p.text) for p in d2.paragraphs)
    print("正文总字数约:", n)


if __name__ == "__main__":
    main()

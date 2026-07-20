// mm_ui -- P4 多模态·前端开发域图像编码器: UI 截图/设计稿 -> 结构化文本。
//
// ds4_multimodal.c "image" 族的外部编码器 (popen 契约: argv[1]=图片文件,
// stdout=文本)。面向前端工作流设计: 文本模型拿到的不是一串裸 OCR 行, 而是
// 一张可推理的 UI 结构草图 -- 尺寸 / 调色板 / 匀色块几何 / 文字(含前景/背景
// 色, CSS 同款左上原点像素坐标) -- 足以回答"这个按钮什么颜色 / 为什么没对齐 /
// 照这张图写出 HTML+CSS"。报错截图逐字还原 (mm_ocr 的老门槛) 被 [text ...]
// 行完整继承: OCR 一律 usesLanguageCorrection=false, 代码/报错不被"纠错"。
//
// 诚实边界: 这是结构草图(几何+颜色+文字), 不是语义视觉 -- 编码器不猜"这是
// 头像/图表", 语义推断留给拿到草图的语言模型。
//
// 输出格式 (原图像素坐标, 左上原点):
//   [img 1280x800]
//   [palette #f5f6f8 56% #101828 8% ...]      主色按覆盖率降序 (>=1%)
//   [rect X,Y WxH #rrggbb]                    近匀色矩形块, 面积降序
//   [+N smaller rects omitted]                超出上限时诚实报数, 不静默截断
//   [text X,Y WxH #fg on #bg "..."]           OCR 行, 阅读序 (行分组+行内按 x)
//
// 用法:
//   mm_ui IMAGE.png            # 编码到 stdout
//   mm_ui --selftest           # 自画登录卡片 UI -> 全管线校验 (退出码判决)
//   mm_ui --render-ui OUT.png  # 把 selftest 的合成 UI 写盘 (供端到端测试)
//
// 构建: make mm-ui  (swiftc -O tools/mm_ui.swift -o mm-ui)
import Foundation
import Vision
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

// ---- 降采样分析位图 (几何/颜色都在这层算, OCR 用原图保精度) ----

struct AnalysisBitmap {
    var px: [UInt8]        // RGBA8, rowStride 字节/行, 行 0 = 视觉顶部
    var w: Int
    var h: Int
    var rowStride: Int
    var toOrigX: Double    // 分析坐标 -> 原图坐标
    var toOrigY: Double
    @inline(__always) func rgb(_ x: Int, _ y: Int) -> (Int, Int, Int) {
        let o = y * rowStride + x * 4
        return (Int(px[o]), Int(px[o + 1]), Int(px[o + 2]))
    }
}

// 分析一律在 sRGB 里做: 报出的 hex 才是能直接进 CSS 的 sRGB 值, 且与源图
// 内嵌 profile (P3/…) 和宿主显示器无关 -- 同一张图在任何机器上编码出同一段
// 文本 (KV 前缀键确定性)。
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

func analysisBitmap(_ img: CGImage, maxDim: Int = 768) -> AnalysisBitmap? {
    let ow = img.width, oh = img.height
    if ow < 1 || oh < 1 { return nil }
    let scale = min(1.0, Double(maxDim) / Double(max(ow, oh)))
    let w = max(1, Int((Double(ow) * scale).rounded()))
    let h = max(1, Int((Double(oh) * scale).rounded()))
    guard let ctx = CGContext(data: nil, width: w, height: h,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: srgb,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    ctx.interpolationQuality = .medium
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let data = ctx.data else { return nil }
    let stride = ctx.bytesPerRow
    let buf = [UInt8](UnsafeBufferPointer(
        start: data.assumingMemoryBound(to: UInt8.self), count: stride * h))
    return AnalysisBitmap(px: buf, w: w, h: h, rowStride: stride,
                          toOrigX: Double(ow) / Double(w),
                          toOrigY: Double(oh) / Double(h))
}

// ---- 调色板: 5bit/通道桶直方图, 桶内均值 = 代表色 ----
// 5bit 是前端底线: #ffffff 卡片 vs #f5f6f8 页底 (Δ7-10/通道) 必须分桶,
// 4bit 会把这对最常见的"白卡浅灰底"合并成一条。

let bucketCount = 32768
@inline(__always) func bucketKey(_ r: Int, _ g: Int, _ b: Int) -> Int {
    return ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
}

struct PaletteEntry { var r: Int; var g: Int; var b: Int; var pct: Double }

func palette(_ d: AnalysisBitmap, maxEntries: Int = 6) -> [PaletteEntry] {
    var cnt = [Int](repeating: 0, count: bucketCount)
    var sr = [Int](repeating: 0, count: bucketCount)
    var sg = [Int](repeating: 0, count: bucketCount)
    var sb = [Int](repeating: 0, count: bucketCount)
    for y in 0..<d.h {
        for x in 0..<d.w {
            let (r, g, b) = d.rgb(x, y)
            let k = bucketKey(r, g, b)
            cnt[k] += 1; sr[k] += r; sg[k] += g; sb[k] += b
        }
    }
    let total = d.w * d.h
    let order = (0..<bucketCount).filter { cnt[$0] > 0 }
        .sorted { cnt[$0] != cnt[$1] ? cnt[$0] > cnt[$1] : $0 < $1 }
    var out: [PaletteEntry] = []
    for k in order {
        let pct = Double(cnt[k]) * 100.0 / Double(total)
        if pct < 1.0 || out.count >= maxEntries { break }
        out.append(PaletteEntry(r: sr[k] / cnt[k], g: sg[k] / cnt[k],
                                b: sb[k] / cnt[k], pct: pct))
    }
    return out
}

// ---- 匀色块检测: seed 固定容差 flood fill + 矩形度过滤 ----
// UI 的按钮/卡片/条幅都是匀色填充: 与 seed 色近的 4 连通块, 填充率(块像素 /
// bbox 面积) >= 78% 视为矩形结构。抗锯齿边像素天然掉队, 渐变自然碎掉 --
// 不是 UI 色块的东西不会被硬拗成矩形。

struct RectBlock {
    var x: Int; var y: Int; var w: Int; var h: Int   // 原图像素坐标
    var r: Int; var g: Int; var b: Int               // 块均色
    var count: Int                                   // 分析分辨率下的像素数
}

// tol=6: 前端的"白卡片(#ffffff)在浅灰底(#f5f6f8)上"只差 7-10/通道, 容差
// 必须小于它; 匀色填充的降采样内部噪声 <=±4, 不会碎。
func rectBlocks(_ d: AnalysisBitmap, tol: Int = 6, cap: Int = 24) -> (kept: [RectBlock], dropped: Int) {
    let n = d.w * d.h
    var visited = [Bool](repeating: false, count: n)
    var queue = [Int32](); queue.reserveCapacity(4096)
    var all: [RectBlock] = []
    let minCount = max(48, n / 2000)     // >=0.05% 面积的连通块才算结构
    for sy in 0..<d.h {
        for sx in 0..<d.w {
            let si = sy * d.w + sx
            if visited[si] { continue }
            let (r0, g0, b0) = d.rgb(sx, sy)
            visited[si] = true
            queue.removeAll(keepingCapacity: true)
            queue.append(Int32(si))
            var head = 0
            var count = 0
            // 颜色取【众数桶】而非全块均值: 均值会被边缘混合像素/块内小字拉偏
            // (#3b82f6 按钮被拉成 #4998f8), 众数桶=纯填充色, 是能直接进 CSS 的
            // design token。
            var hist: [Int: (n: Int, r: Int, g: Int, b: Int)] = [:]
            var minX = sx, maxX = sx, minY = sy, maxY = sy
            while head < queue.count {
                let i = Int(queue[head]); head += 1
                let x = i % d.w, y = i / d.w
                let (r, g, b) = d.rgb(x, y)
                count += 1
                let k = bucketKey(r, g, b)
                let cur = hist[k] ?? (0, 0, 0, 0)
                hist[k] = (cur.n + 1, cur.r + r, cur.g + g, cur.b + b)
                if x < minX { minX = x }; if x > maxX { maxX = x }
                if y < minY { minY = y }; if y > maxY { maxY = y }
                let nbs = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                for (nx, ny) in nbs {
                    if nx < 0 || ny < 0 || nx >= d.w || ny >= d.h { continue }
                    let ni = ny * d.w + nx
                    if visited[ni] { continue }
                    let (nr, ng, nb2) = d.rgb(nx, ny)
                    if abs(nr - r0) > tol || abs(ng - g0) > tol || abs(nb2 - b0) > tol { continue }
                    visited[ni] = true
                    queue.append(Int32(ni))
                }
            }
            if count < minCount { continue }
            let bw = maxX - minX + 1, bh = maxY - minY + 1
            if bw < 3 || bh < 3 { continue }
            if count * 100 < bw * bh * 78 { continue }
            // Dictionary 遍历序不确定: 平局按最小桶键裁决, 保证同图同输出
            // (编码文本是 KV 前缀键的一部分, 必须确定性)
            var modeKey = -1
            var mode = (n: 0, r: 0, g: 0, b: 0)
            for (k, v) in hist {
                if v.n > mode.n || (v.n == mode.n && (modeKey < 0 || k < modeKey)) {
                    mode = v
                    modeKey = k
                }
            }
            if mode.n == 0 { continue }
            let ox0 = Int((Double(minX) * d.toOrigX).rounded())
            let oy0 = Int((Double(minY) * d.toOrigY).rounded())
            let ox1 = Int((Double(maxX + 1) * d.toOrigX).rounded())
            let oy1 = Int((Double(maxY + 1) * d.toOrigY).rounded())
            all.append(RectBlock(x: ox0, y: oy0, w: ox1 - ox0, h: oy1 - oy0,
                                 r: mode.r / mode.n, g: mode.g / mode.n,
                                 b: mode.b / mode.n, count: count))
        }
    }
    all.sort { $0.count != $1.count ? $0.count > $1.count
                                    : ($0.y != $1.y ? $0.y < $1.y : $0.x < $1.x) }
    if all.count > cap { return (Array(all[0..<cap]), all.count - cap) }
    return (all, 0)
}

// ---- OCR: 原图全精度, 逐字优先, 带几何; 阅读序 = 行分组 + 行内按 x ----

struct TextItem {
    var x: Int; var y: Int; var w: Int; var h: Int   // 原图像素坐标 (左上原点)
    var s: String
    var fg: (Int, Int, Int)?
    var bg: (Int, Int, Int)?
}

// 同形折叠: usesLanguageCorrection=false 时 Vision 对形状全同的字符
// (ASCII 'c' vs 西里尔 'с' U+0441) 是任选的, 语言表也钉不住。代码/报错里
// 混进同形非 ASCII 会毒化 KV 前缀键与模型输入, 所以对【拉丁主导】(>=60%
// ASCII) 的行确定性折回 ASCII -- 域先验消歧, 中文 UI 文案行不受影响。
let confusables: [Character: Character] = [
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I",
    "Ј": "J",
]

func foldConfusables(_ s: String) -> String {
    let total = s.unicodeScalars.count
    guard total > 0 else { return s }
    let ascii = s.unicodeScalars.lazy.filter { $0.isASCII }.count
    guard ascii * 5 >= total * 3 else { return s }
    return String(s.map { confusables[$0] ?? $0 })
}

func ocrItems(_ img: CGImage) throws -> [TextItem] {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false   // 代码/报错逐字优先, 不要"纠错"
    // 语言表必须钉死: 不设时 Vision 会同形漂移 (ASCII 'c' 识成西里尔 с
    // U+0441), 毒化 KV 前缀键和模型输入。en 在前保代码/报错字母表,
    // zh-Hans 兜中文 UI 文案。
    req.recognitionLanguages = ["en-US", "zh-Hans"]
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    try handler.perform([req])
    let W = CGFloat(img.width), H = CGFloat(img.height)
    var items: [TextItem] = []
    for ob in req.results ?? [] {
        guard let cand = ob.topCandidates(1).first, !cand.string.isEmpty else { continue }
        let bb = ob.boundingBox   // 归一化, 左下原点
        let x = Int((bb.minX * W).rounded())
        let y = Int(((1 - bb.maxY) * H).rounded())
        let w = max(1, Int((bb.width * W).rounded()))
        let h = max(1, Int((bb.height * H).rounded()))
        items.append(TextItem(x: x, y: y, w: w, h: h,
                              s: foldConfusables(cand.string), fg: nil, bg: nil))
    }
    items.sort { $0.y != $1.y ? $0.y < $1.y : $0.x < $1.x }
    var lines: [(top: Int, bot: Int, v: [TextItem])] = []
    for it in items {
        let itBot = it.y + it.h
        if let li = lines.indices.last {
            let ov = min(lines[li].bot, itBot) - max(lines[li].top, it.y)
            if ov * 2 >= min(it.h, lines[li].bot - lines[li].top) {
                lines[li].v.append(it)
                lines[li].top = min(lines[li].top, it.y)
                lines[li].bot = max(lines[li].bot, itBot)
                continue
            }
        }
        lines.append((it.y, itBot, [it]))
    }
    var out: [TextItem] = []
    for var ln in lines {
        ln.v.sort { $0.x < $1.x }
        out.append(contentsOf: ln.v)
    }
    return out
}

// 文字前景/背景色: bbox 内主导色桶 = 背景(字形像素是少数); 前景 = 距背景够远
// (L1 > 96) 的像素里【众数桶】的均值 -- 字形笔画核心是纯色同桶, 抗锯齿混合
// 像素散在很多桶里, 取众数桶避免"均值发糊"(#dc2626 被拉成 #ec8982 那类)。
// 在【原图分辨率】采样: 降采样会把细笔画整根糊成混合色, 纯色核心只在原图有。
// 采样不足时诚实置 nil, 输出里省略颜色子句。
func sampleTextColors(_ d: AnalysisBitmap, _ it: inout TextItem) {
    let x0 = max(0, Int(Double(it.x) / d.toOrigX))
    let y0 = max(0, Int(Double(it.y) / d.toOrigY))
    let x1 = min(d.w, Int((Double(it.x + it.w) / d.toOrigX).rounded(.up)))
    let y1 = min(d.h, Int((Double(it.y + it.h) / d.toOrigY).rounded(.up)))
    if x1 - x0 < 2 || y1 - y0 < 1 { return }
    var cnt = [Int](repeating: 0, count: bucketCount)
    var sr = [Int](repeating: 0, count: bucketCount)
    var sg = [Int](repeating: 0, count: bucketCount)
    var sb = [Int](repeating: 0, count: bucketCount)
    for y in y0..<y1 {
        for x in x0..<x1 {
            let (r, g, b) = d.rgb(x, y)
            let k = bucketKey(r, g, b)
            cnt[k] += 1; sr[k] += r; sg[k] += g; sb[k] += b
        }
    }
    var bestK = -1
    for k in 0..<bucketCount where cnt[k] > 0 {
        if bestK < 0 || cnt[k] > cnt[bestK] { bestK = k }
    }
    guard bestK >= 0 else { return }
    let bg = (sr[bestK] / cnt[bestK], sg[bestK] / cnt[bestK], sb[bestK] / cnt[bestK])
    it.bg = bg
    var fcnt = [Int](repeating: 0, count: bucketCount)
    var fn = 0
    for y in y0..<y1 {
        for x in x0..<x1 {
            let (r, g, b) = d.rgb(x, y)
            if abs(r - bg.0) + abs(g - bg.1) + abs(b - bg.2) > 96 {
                fcnt[bucketKey(r, g, b)] += 1
                fn += 1
            }
        }
    }
    let boxN = (x1 - x0) * (y1 - y0)
    guard fn >= 4 && fn * 50 >= boxN else { return }   // >=2% 的远色像素才敢报前景
    var fgK = -1
    for k in 0..<bucketCount where fcnt[k] > 0 {
        if fgK < 0 || fcnt[k] > fcnt[fgK] { fgK = k }
    }
    if fgK >= 0 {
        // 众数桶的代表色用桶内均值 (sr/sg/sb 是全 bbox 的桶累加, 含背景桶,
        // 但 fgK 桶只会被远色像素以外的同色像素稀释 -- 同桶即同色, 无碍)
        it.fg = (sr[fgK] / cnt[fgK], sg[fgK] / cnt[fgK], sb[fgK] / cnt[fgK])
    }
}

// ---- 编码主流程 ----

func hexColor(_ c: (Int, Int, Int)) -> String {
    return String(format: "#%02x%02x%02x", c.0, c.1, c.2)
}

func escText(_ s: String) -> String {
    var o = ""
    for c in s.unicodeScalars {
        switch c {
        case "\"": o += "\\\""
        case "\\": o += "\\\\"
        case "\n": o += "\\n"
        case "\r": o += "\\r"
        default: o.unicodeScalars.append(c)
        }
    }
    return o
}

func describe(_ img: CGImage) throws -> String {
    var out = "[img \(img.width)x\(img.height)]\n"
    guard let d = analysisBitmap(img) else {
        throw NSError(domain: "mm_ui", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "bitmap alloc failed"])
    }
    let pal = palette(d)
    if !pal.isEmpty {
        out += "[palette"
        for p in pal { out += " \(hexColor((p.r, p.g, p.b))) \(Int(p.pct.rounded()))%" }
        out += "]\n"
    }
    let (rects, dropped) = rectBlocks(d)
    for r in rects {
        out += "[rect \(r.x),\(r.y) \(r.w)x\(r.h) \(hexColor((r.r, r.g, r.b)))]\n"
    }
    if dropped > 0 { out += "[+\(dropped) smaller rects omitted]\n" }
    // 文字颜色在原图分辨率采样 (细笔画的纯色核心只在原图有); 超大图退回分析图
    let colorSrc = (img.width * img.height <= 8_000_000
                    ? analysisBitmap(img, maxDim: max(img.width, img.height))
                    : nil) ?? d
    var texts = try ocrItems(img)
    for i in texts.indices {
        sampleTextColors(colorSrc, &texts[i])
        let t = texts[i]
        var colors = ""
        if let fg = t.fg, let bg = t.bg {
            colors = " \(hexColor(fg)) on \(hexColor(bg))"
        } else if let bg = t.bg {
            colors = " on \(hexColor(bg))"
        }
        out += "[text \(t.x),\(t.y) \(t.w)x\(t.h)\(colors) \"\(escText(t.s))\"]\n"
    }
    return out
}

func loadImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)
    else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

// ---- 合成 UI (selftest 与 --render-ui 共用): 登录卡片 + 前端报错行 ----

let stW = 1280, stH = 800
let stButton = (x: 512, y: 436, w: 256, h: 44)        // #3b82f6
let stHeaderH = 64                                     // #101828
let stErrorText = "TypeError: Cannot read properties of undefined (reading 'map')"

func drawSyntheticUI() -> CGImage? {
    guard let ctx = CGContext(data: nil, width: stW, height: stH,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: srgb,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    // 颜色必须与位图同在 sRGB: CGColor(red:...) 是 generic RGB, 画进 sRGB 位图
    // 会被转换偏移 (#3b82f6 → ~#4998f8), selftest 的"真值"就不真了。
    func rgb(_ r: Int, _ g: Int, _ b: Int) -> CGColor {
        CGColor(colorSpace: srgb,
                components: [CGFloat(r) / 255, CGFloat(g) / 255,
                             CGFloat(b) / 255, 1])!
    }
    // CG 原点在左下; 以下均从左上原点 spec 转换 (cgY = H - y - h)
    func fillTL(_ x: Int, _ y: Int, _ w: Int, _ h: Int, _ c: CGColor) {
        ctx.setFillColor(c)
        ctx.fill(CGRect(x: x, y: stH - y - h, width: w, height: h))
    }
    func text(_ s: String, _ x: Int, _ baselineY: Int, _ size: CGFloat,
              _ c: CGColor, font: String = "Helvetica") -> Double {
        let f = CTFontCreateWithName(font as CFString, size, nil)
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): f,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): c,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: s, attributes: attrs))
        ctx.textPosition = CGPoint(x: CGFloat(x), y: CGFloat(stH - baselineY))
        CTLineDraw(line, ctx)
        return CTLineGetTypographicBounds(line, nil, nil, nil)
    }
    func measure(_ s: String, _ size: CGFloat, font: String = "Helvetica") -> Double {
        let f = CTFontCreateWithName(font as CFString, size, nil)
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): f,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: s, attributes: attrs))
        return CTLineGetTypographicBounds(line, nil, nil, nil)
    }
    fillTL(0, 0, stW, stH, rgb(245, 246, 248))                     // 页面底 #f5f6f8
    fillTL(0, 0, stW, stHeaderH, rgb(16, 24, 40))                  // 顶栏 #101828
    fillTL(480, 280, 320, 240, rgb(255, 255, 255))                 // 卡片 #ffffff
    let bpath = CGPath(roundedRect: CGRect(x: stButton.x,
                                           y: stH - stButton.y - stButton.h,
                                           width: stButton.w, height: stButton.h),
                       cornerWidth: 8, cornerHeight: 8, transform: nil)
    ctx.setFillColor(rgb(59, 130, 246))                            // 按钮 #3b82f6
    ctx.addPath(bpath)
    ctx.fillPath()
    _ = text("Acme Console", 40, 40, 22, rgb(255, 255, 255))
    _ = text("Sign in", 512, 338, 24, rgb(16, 24, 40))
    let cw = measure("Continue", 15)
    _ = text("Continue", stButton.x + (stButton.w - Int(cw.rounded())) / 2,
             stButton.y + 29, 15, rgb(255, 255, 255))
    _ = text(stErrorText, 480, 574, 16, rgb(220, 38, 38), font: "Menlo")
    return ctx.makeImage()
}

func writePNG(_ img: CGImage, _ path: String) -> Bool {
    guard let dest = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { return false }
    CGImageDestinationAddImage(dest, img, nil)
    return CGImageDestinationFinalize(dest)
}

// ---- selftest: 合成 UI 全管线判决 ----
// 门 (前端域必备): ①报错/文案逐字 OCR; ②按钮几何+颜色; ③顶栏结构;
// ④白卡片在浅灰底上被分开检出 (Δ7-10/通道, 前端最常见的布局对);
// ⑤文字前景/背景色关联 (Continue = 白字蓝底, TypeError = 红字)。

func selftest() -> Int32 {
    guard let img = drawSyntheticUI() else { err("selftest: draw failed"); return 2 }
    let desc: String
    do { desc = try describe(img) } catch { err("selftest: \(error)"); return 2 }
    print(desc, terminator: "")

    var failures: [String] = []
    // ① 逐字文本 (前端报错 + UI 文案)
    for needle in ["Acme Console", "Sign in", "Continue",
                   "Cannot read properties of undefined"] {
        if !desc.contains(needle) { failures.append("missing text \"\(needle)\"") }
    }
    // 重新算结构块用于几何断言 (与 describe 相同管线)
    guard let d = analysisBitmap(img) else { err("selftest: bitmap"); return 2 }
    let (rects, _) = rectBlocks(d)
    // ② 按钮: 几何 ±12/±20, 颜色 L∞ <= 40 贴 #3b82f6
    let button = rects.first { r in
        abs(r.x - stButton.x) <= 12 && abs(r.y - stButton.y) <= 12 &&
        abs(r.w - stButton.w) <= 20 && abs(r.h - stButton.h) <= 12 &&
        abs(r.r - 59) <= 40 && abs(r.g - 130) <= 40 && abs(r.b - 246) <= 40
    }
    if button == nil { failures.append("button rect ~(512,436 256x44 #3b82f6) not detected") }
    // ③ 顶栏: 贴顶、全宽、暗色
    let header = rects.first { r in
        r.y <= 6 && r.h >= 52 && r.h <= 76 && r.w >= stW - 16 &&
        (r.r + r.g + r.b) / 3 <= 60
    }
    if header == nil { failures.append("dark header bar not detected") }
    // ④ 白卡片 (480,280 320x240 #ffffff) 从 #f5f6f8 底上分出来
    let card = rects.first { r in
        abs(r.x - 480) <= 12 && abs(r.y - 280) <= 12 &&
        abs(r.w - 320) <= 20 && abs(r.h - 240) <= 16 &&
        r.r >= 250 && r.g >= 250 && r.b >= 250
    }
    if card == nil { failures.append("white card on #f5f6f8 not separated (subtle-shade wall)") }
    // ⑤ 文字前景/背景 (原图分辨率采样, 与 describe 同源)
    let colorSrc = analysisBitmap(img, maxDim: max(stW, stH)) ?? d
    var texts = (try? ocrItems(img)) ?? []
    var contOK = false
    var errRedOK = false
    for i in texts.indices {
        sampleTextColors(colorSrc, &texts[i])
        let t = texts[i]
        if t.s.contains("Continue"), let fg = t.fg, let bg = t.bg,
           (fg.0 + fg.1 + fg.2) / 3 >= 180,
           abs(bg.0 - 59) <= 48, abs(bg.1 - 130) <= 48, abs(bg.2 - 246) <= 48 {
            contOK = true
        }
        if t.s.contains("TypeError"), let fg = t.fg,
           fg.0 > fg.1 + 30, fg.0 > fg.2 + 30 {   // 红色主导 (#dc2626 系)
            errRedOK = true
        }
    }
    if !contOK { failures.append("\"Continue\" fg/bg (white on #3b82f6) not resolved") }
    if !errRedOK { failures.append("TypeError line fg not red-dominant") }
    // 调色板首位 = 页面底色
    if let p0 = palette(d).first {
        if abs(p0.r - 245) > 20 || abs(p0.g - 246) > 20 || abs(p0.b - 248) > 20 {
            failures.append("palette[0] != page background #f5f6f8")
        }
    } else {
        failures.append("empty palette")
    }

    if failures.isEmpty { err("selftest OK"); return 0 }
    for f in failures { err("selftest FAIL: \(f)") }
    return 1
}

// ---- main ----

let args = CommandLine.arguments
if args.count == 2 && args[1] == "--selftest" {
    exit(selftest())
}
if args.count == 3 && args[1] == "--render-ui" {
    guard let img = drawSyntheticUI(), writePNG(img, args[2]) else {
        err("render-ui failed"); exit(2)
    }
    exit(0)
}
guard args.count == 2, let img = loadImage(args[1]) else {
    err("usage: mm_ui IMAGE | --selftest | --render-ui OUT.png")
    exit(2)
}
do {
    print(try describe(img), terminator: "")
} catch {
    err("mm_ui error: \(error)")
    exit(1)
}

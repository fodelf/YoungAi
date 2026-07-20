// mm_ocr -- P4 多模态第一个真实编码器: 图片 -> 文本 (macOS Vision OCR)。
//
// ds4_multimodal.c 的 "image" 族编码器外部实现: C 侧经 ds4_mm_register 注册一个
// popen 包装 (图片字节 -> 本工具 -> 文本 -> tokenizer 回调 -> tokens)。服务
// tiny-coder-plan P4 场景: Claude Code 里贴一张报错截图 -> 模型读出报错文本。
//
// 用法:
//   mm_ocr IMAGE.png        # OCR 图片, 文本到 stdout
//   mm_ocr --selftest       # 自画一张含报错文本的图 -> OCR -> 校验 (退出码判决)
//
// 构建: swiftc -O tools/mm_ocr.swift -o mm-ocr  (产物在 repo 根, 已 gitignore 类别)
import Foundation
import Vision
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

func ocr(_ cgImage: CGImage) throws -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false   // 报错文本/代码逐字优先, 不要"纠错"
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])
    let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    return lines.joined(separator: "\n")
}

func loadImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

// --selftest: 画一段典型 Go 报错到位图, OCR 回来必须含关键词。
func selftest() -> Int32 {
    let text = "panic: runtime error: index out of range [3] with length 3"
    let width = 900, height = 60
    guard let ctx = CGContext(data: nil, width: width, height: height,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        FileHandle.standardError.write("selftest: no context\n".data(using: .utf8)!)
        return 2
    }
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let font = CTFontCreateWithName("Menlo" as CFString, 20, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String):
            CGColor(red: 0, green: 0, blue: 0, alpha: 1),
    ]
    let attr = NSAttributedString(string: text, attributes: attrs)
    let line = CTLineCreateWithAttributedString(attr)
    ctx.textPosition = CGPoint(x: 12, y: 22)
    CTLineDraw(line, ctx)
    guard let img = ctx.makeImage() else { return 2 }
    do {
        let got = try ocr(img)
        print(got)
        // 判决: 关键 token 必须逐字回来 (OCR 大小写/空格漂移容忍, 关键词不容忍)
        for needle in ["panic", "runtime error", "index out of range"] {
            if !got.contains(needle) {
                FileHandle.standardError.write("selftest FAIL: missing \"\(needle)\"\n".data(using: .utf8)!)
                return 1
            }
        }
        FileHandle.standardError.write("selftest OK\n".data(using: .utf8)!)
        return 0
    } catch {
        FileHandle.standardError.write("selftest OCR error: \(error)\n".data(using: .utf8)!)
        return 2
    }
}

func renderText(_ text: String, to path: String) -> Int32 {
    let width = 900, height = 60
    guard let ctx = CGContext(data: nil, width: width, height: height,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return 2 }
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let font = CTFontCreateWithName("Menlo" as CFString, 20, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String):
            CGColor(red: 0, green: 0, blue: 0, alpha: 1),
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
    ctx.textPosition = CGPoint(x: 12, y: 22)
    CTLineDraw(line, ctx)
    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                     UTType.png.identifier as CFString, 1, nil) else { return 2 }
    CGImageDestinationAddImage(dest, img, nil)
    return CGImageDestinationFinalize(dest) ? 0 : 2
}

let args = CommandLine.arguments
if args.count == 2 && args[1] == "--selftest" {
    exit(selftest())
}
if args.count == 4 && args[1] == "--render" {
    exit(renderText(args[2], to: args[3]))
}
guard args.count == 2, let img = loadImage(args[1]) else {
    FileHandle.standardError.write("usage: mm_ocr IMAGE | --selftest | --render TEXT OUT.png\n".data(using: .utf8)!)
    exit(2)
}
do {
    print(try ocr(img))
} catch {
    FileHandle.standardError.write("ocr error: \(error)\n".data(using: .utf8)!)
    exit(1)
}

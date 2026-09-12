//
//  PullFigure.swift
//  VeilKit
//
//  Where a route pulls, before and after the guard: per-position pull of the route over its
//  anchor, drawn over washed-out copies of the person's own held-out photos, on a fixed absolute
//  scale (never per-image normalized, which would always show a hot spot). The only pixels are
//  the photos'; no latent is decoded. The figure contains the person's photos: it is a local
//  artifact, not part of the shareable report.
//

import CoreGraphics
import CoreText
import Foundation

public struct PullMap: Sendable {
    public let rows: Int
    public let cols: Int
    /// Relative pull per cell: (ℓ_anchor − ℓ_route) / mean ℓ_anchor.
    public let values: [Float]

    public static func relative(anchor: [Float], route: [Float], rows: Int, cols: Int) -> PullMap {
        let mean = anchor.reduce(0, +) / Float(max(anchor.count, 1))
        return PullMap(rows: rows, cols: cols, values: zip(anchor, route).map { ($0 - $1) / max(mean, 1e-12) })
    }

    func sample(x: Double, y: Double, width: Double, height: Double) -> Float {
        let fx = x / width * Double(cols) - 0.5, fy = y / height * Double(rows) - 0.5
        let x0 = Int(floor(fx)), y0 = Int(floor(fy))
        let tx = Float(fx - Double(x0)), ty = Float(fy - Double(y0))
        func v(_ i: Int, _ j: Int) -> Float { values[min(max(j, 0), rows - 1) * cols + min(max(i, 0), cols - 1)] }
        let top = v(x0, y0) * (1 - tx) + v(x0 + 1, y0) * tx
        let bottom = v(x0, y0 + 1) * (1 - tx) + v(x0 + 1, y0 + 1) * tx
        return top * (1 - ty) + bottom * ty
    }
}

public struct PullFigure {
    public var colormap = Colormap.pull
    public var threshold: Float = 0.02
    public var scaleMax: Float = 0.5
    public var panelSide = 300

    public init() {}

    /// One row per photo: the photo, the base pull overlay, the guarded pull overlay.
    public func render(photos: [SubjectPhoto], base: [PullMap], guarded: [PullMap], title: String, subtitle: String) -> CGImage? {
        let n = min(photos.count, base.count, guarded.count, 3)
        guard n > 0 else { return nil }
        let pad = 20, header = 64, legend = 70, label = 22
        let width = 3 * panelSide + 4 * pad
        let height = header + n * (panelSide + label + pad) + legend
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(srgbRed: 0.98, green: 0.98, blue: 0.97, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let ink = CGColor(srgbRed: 0.1, green: 0.1, blue: 0.1, alpha: 1)
        let muted = CGColor(srgbRed: 0.38, green: 0.38, blue: 0.37, alpha: 1)
        func rect(_ x: Int, _ y: Int, _ w: Int, _ h: Int) -> CGRect { CGRect(x: x, y: height - y - h, width: w, height: h) }
        text(ctx, title, x: pad, top: 30, size: 18, bold: true, color: ink, canvas: height)
        text(ctx, subtitle, x: pad, top: 52, size: 12, bold: false, color: muted, canvas: height)

        for i in 0..<n {
            let photo = photos[i]
            let s = Double(panelSide) / Double(max(photo.width, photo.height))
            let pw = max(1, Int(Double(photo.width) * s)), ph = max(1, Int(Double(photo.height) * s))
            let y = header + i * (panelSide + label + pad)
            let pixels = photo.rgba(width: pw, height: ph)
            if let img = ImageWriter.image(rgba: pixels, width: pw, height: ph) { ctx.draw(img, in: rect(pad, y, pw, ph)) }
            for (column, map) in [(1, base[i]), (2, guarded[i])] {
                var out = [UInt8](repeating: 255, count: pw * ph * 4)
                for yy in 0..<ph {
                    for xx in 0..<pw {
                        let k = (yy * pw + xx) * 4
                        let lum = (0.2126 * Float(pixels[k]) + 0.7152 * Float(pixels[k + 1]) + 0.0722 * Float(pixels[k + 2])) / 255
                        let g = 0.5 + 0.5 * lum
                        var (r, gg, b) = (g, g, g)
                        let v = map.sample(x: Double(xx) + 0.5, y: Double(yy) + 0.5, width: Double(pw), height: Double(ph))
                        if let c = colormap.rgba(v, threshold: threshold, scaleMax: scaleMax) {
                            r = (1 - c.a) * g + c.a * c.r
                            gg = (1 - c.a) * g + c.a * c.g
                            b = (1 - c.a) * g + c.a * c.b
                        }
                        out[k] = UInt8(255 * min(max(r, 0), 1))
                        out[k + 1] = UInt8(255 * min(max(gg, 0), 1))
                        out[k + 2] = UInt8(255 * min(max(b, 0), 1))
                    }
                }
                if let img = ImageWriter.image(rgba: out, width: pw, height: ph) {
                    ctx.draw(img, in: rect(pad + column * (panelSide + pad), y, pw, ph))
                }
            }
            let labels = ["Held-out photo \(i + 1)", "Pull, base model", "Pull, guarded"]
            for (c, l) in labels.enumerated() {
                text(ctx, l, x: pad + c * (panelSide + pad), top: y + ph + 16, size: 11.5, bold: c > 0, color: ink, canvas: height)
            }
        }

        let lx = pad + panelSide + pad, ly = height - legend + 18, lw = min(2 * panelSide, 360), lh = 12
        for s in 0..<120 {
            let v = threshold + (scaleMax - threshold) * Float(s) / 119
            guard let c = colormap.rgba(v, threshold: threshold, scaleMax: scaleMax) else { continue }
            ctx.setFillColor(CGColor(srgbRed: CGFloat(c.r), green: CGFloat(c.g), blue: CGFloat(c.b), alpha: 1))
            ctx.fill(rect(lx + s * lw / 120, ly, lw / 120 + 1, lh))
        }
        text(ctx, String(format: "%.2f", threshold), x: lx, top: ly + lh + 15, size: 10.5, bold: false, color: muted, canvas: height)
        text(ctx, String(format: "%.1f", scaleMax), x: lx + lw - 18, top: ly + lh + 15, size: 10.5, bold: false, color: muted, canvas: height)
        text(ctx, "Relative pull of the route over its anchor · clear below \(String(format: "%.2f", threshold))",
             x: pad, top: ly + 10, size: 10.5, bold: false, color: muted, canvas: height)
        return ctx.makeImage()
    }

    private func text(_ ctx: CGContext, _ s: String, x: Int, top: Int, size: CGFloat, bold: Bool, color: CGColor, canvas: Int) {
        let font = CTFontCreateWithName((bold ? "Helvetica-Bold" : "Helvetica") as CFString, size, nil)
        let attributed = NSAttributedString(string: s, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
        ])
        ctx.textPosition = CGPoint(x: x, y: canvas - top)
        CTLineDraw(CTLineCreateWithAttributedString(attributed), ctx)
    }
}

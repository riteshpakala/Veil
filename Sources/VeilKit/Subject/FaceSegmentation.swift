//
//  FaceSegmentation.swift
//  VeilKit
//  Restored from Scorpion's archive (tag archive/likeness-needle-v1: FaceRegions.swift,
//  FaceSegmentation.swift), GPL-3.0; trimmed to the face mask.
//
//  Face segmentation, on-device with Vision. The face mask is the convex hull of the jaw
//  contour and the eyebrows lifted toward the forehead, intersected with the person matte (so
//  background pixels inside the hull drop out), and feathered. Masks are rasters over image
//  coordinates, so any executor samples them at its own latent grid. Veil uses the mask to
//  weight the denoising loss toward the likeness: identity lives in the face, not in the
//  background or the clothes.
//

import CoreGraphics
import CoreVideo
import Foundation
import Vision

public struct FaceRegion: Codable, Sendable, Hashable {
    public let index: Int
    /// Pixel coordinates, top-left origin.
    public let bounds: CGRect
    public let confidence: Float
    /// Landmark polygons (pixels, top-left origin): faceContour, leftEye, rightEye,
    /// leftEyebrow, rightEyebrow, nose, outerLips.
    public var polygons: [String: [CGPoint]] = [:]
}

public enum FaceDetector {
    /// Faces, largest first.
    public static func detect(_ photo: SubjectPhoto) throws -> [FaceRegion] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: photo.image, options: [:])
        try handler.perform([request])
        let w = CGFloat(photo.width), h = CGFloat(photo.height)
        let size = CGSize(width: w, height: h)
        let faces = (request.results ?? []).sorted {
            $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height
        }
        return faces.enumerated().map { i, obs in
            let bb = obs.boundingBox   // normalized, bottom-left origin
            let bounds = CGRect(x: bb.minX * w, y: (1 - bb.maxY) * h, width: bb.width * w, height: bb.height * h)
            var polygons: [String: [CGPoint]] = [:]
            if let lm = obs.landmarks {
                let regions: [(String, VNFaceLandmarkRegion2D?)] = [
                    ("faceContour", lm.faceContour), ("leftEye", lm.leftEye), ("rightEye", lm.rightEye),
                    ("leftEyebrow", lm.leftEyebrow), ("rightEyebrow", lm.rightEyebrow), ("nose", lm.nose),
                    ("outerLips", lm.outerLips),
                ]
                for (name, region) in regions {
                    guard let pts = region?.pointsInImage(imageSize: size), !pts.isEmpty else { continue }
                    polygons[name] = pts.map { CGPoint(x: $0.x, y: h - $0.y) }
                }
            }
            return FaceRegion(index: i, bounds: bounds, confidence: obs.confidence, polygons: polygons)
        }
    }
}

/// A mask raster over image coordinates (top-left origin), sampled bilinearly.
public struct RasterMask: Codable, Sendable {
    public let width: Int
    public let height: Int
    public let imageSize: CGSize
    public let values: [Float]

    public func sample(x: CGFloat, y: CGFloat) -> Float {
        let fx = x / imageSize.width * CGFloat(width) - 0.5
        let fy = y / imageSize.height * CGFloat(height) - 0.5
        let x0 = Int(floor(fx)), y0 = Int(floor(fy))
        let tx = Float(fx - CGFloat(x0)), ty = Float(fy - CGFloat(y0))
        func v(_ i: Int, _ j: Int) -> Float {
            guard i >= 0, i < width, j >= 0, j < height else { return 0 }
            return values[j * width + i]
        }
        let top = v(x0, y0) * (1 - tx) + v(x0 + 1, y0) * tx
        let bottom = v(x0, y0 + 1) * (1 - tx) + v(x0 + 1, y0 + 1) * tx
        return top * (1 - ty) + bottom * ty
    }

    /// Mean value — the mask's area fraction of the image.
    public var areaFraction: Double { Double(values.reduce(0, +)) / Double(max(values.count, 1)) }

    /// Mask sampled at the cell centers of a cols×rows grid tiling the whole image (row-major).
    public func grid(rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for j in 0..<rows {
            for i in 0..<cols {
                let x = (CGFloat(i) + 0.5) / CGFloat(cols) * imageSize.width
                let y = (CGFloat(j) + 0.5) / CGFloat(rows) * imageSize.height
                out[j * cols + i] = sample(x: x, y: y)
            }
        }
        return out
    }
}

public struct FaceSegment: Sendable {
    public let faceIndex: Int
    public let mask: RasterMask
    public let usedPersonMatte: Bool
    /// "landmarks+matte", "landmarks" or "box" (no contour landmarks).
    public let method: String
}

public enum FaceSegmenter {
    /// Raster resolution on the image's long side.
    public static var rasterSide = 128
    /// Brows are lifted by this multiple of the brow-to-eye distance to cover the forehead.
    public static var foreheadLift: CGFloat = 0.6

    public static func segment(_ photo: SubjectPhoto, face: FaceRegion, useMatte: Bool = true) -> FaceSegment {
        let size = CGSize(width: photo.width, height: photo.height)
        return segment(face: face, imageSize: size, matte: useMatte ? personMatte(photo) : nil)
    }

    /// Pure geometry: testable without Vision.
    public static func segment(face: FaceRegion, imageSize: CGSize, matte: RasterMask?) -> FaceSegment {
        let (w, h) = rasterSize(imageSize)
        let scale = CGFloat(w) / imageSize.width
        let featherRadius = max(1, Int((0.03 * face.bounds.width * scale).rounded()))
        guard let contour = face.polygons["faceContour"], contour.count >= 3 else {
            // No contour: the detector box, slightly enlarged, feathered.
            let b = face.bounds.insetBy(dx: -0.08 * face.bounds.width, dy: -0.12 * face.bounds.height)
            let box = [CGPoint(x: b.minX, y: b.minY), CGPoint(x: b.maxX, y: b.minY),
                       CGPoint(x: b.maxX, y: b.maxY), CGPoint(x: b.minX, y: b.maxY)]
            let values = feather(rasterize(box, imageSize: imageSize, width: w, height: h), width: w, height: h, radius: featherRadius)
            return FaceSegment(faceIndex: face.index, mask: RasterMask(width: w, height: h, imageSize: imageSize, values: values),
                               usedPersonMatte: false, method: "box")
        }

        // Forehead: lift the brows away from the eyes.
        var points = contour
        let eyes = (face.polygons["leftEye"] ?? []) + (face.polygons["rightEye"] ?? [])
        let brows = (face.polygons["leftEyebrow"] ?? []) + (face.polygons["rightEyebrow"] ?? [])
        if !brows.isEmpty {
            let eyeY = eyes.isEmpty ? face.bounds.midY : eyes.map(\.y).reduce(0, +) / CGFloat(eyes.count)
            let browY = brows.map(\.y).reduce(0, +) / CGFloat(brows.count)
            let lift = foreheadLift * max(eyeY - browY, 0)
            points += brows + brows.map { CGPoint(x: $0.x, y: $0.y - lift) }
        }
        var values = rasterize(convexHull(points), imageSize: imageSize, width: w, height: h)

        var usedMatte = false
        if let matte {
            let intersected = zip(values, cellCenters(imageSize: imageSize, width: w, height: h))
                .map { v, p in v * matte.sample(x: p.x, y: p.y) }
            // Only trust the matte when it keeps most of the face (it can miss on paintings).
            if intersected.reduce(0, +) >= 0.4 * values.reduce(0, +) {
                values = intersected
                usedMatte = true
            }
        }
        values = feather(values, width: w, height: h, radius: featherRadius)
        return FaceSegment(faceIndex: face.index, mask: RasterMask(width: w, height: h, imageSize: imageSize, values: values),
                           usedPersonMatte: usedMatte, method: usedMatte ? "landmarks+matte" : "landmarks")
    }

    static func rasterSize(_ size: CGSize) -> (Int, Int) {
        let longSide = max(size.width, size.height)
        return (max(8, Int((CGFloat(rasterSide) * size.width / longSide).rounded())),
                max(8, Int((CGFloat(rasterSide) * size.height / longSide).rounded())))
    }

    static func cellCenters(imageSize: CGSize, width: Int, height: Int) -> [CGPoint] {
        (0..<(width * height)).map { idx in
            CGPoint(x: (CGFloat(idx % width) + 0.5) / CGFloat(width) * imageSize.width,
                    y: (CGFloat(idx / width) + 0.5) / CGFloat(height) * imageSize.height)
        }
    }

    /// Andrew's monotone chain.
    static func convexHull(_ points: [CGPoint]) -> [CGPoint] {
        let p = points.sorted { ($0.x, $0.y) < ($1.x, $1.y) }
        guard p.count >= 3 else { return p }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [CGPoint] = [], upper: [CGPoint] = []
        for pt in p {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], pt) <= 0 { lower.removeLast() }
            lower.append(pt)
        }
        for pt in p.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], pt) <= 0 { upper.removeLast() }
            upper.append(pt)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    /// 1 inside the polygon, 0 outside, sampled at raster cell centers.
    static func rasterize(_ polygon: [CGPoint], imageSize: CGSize, width: Int, height: Int) -> [Float] {
        guard polygon.count >= 3 else { return [Float](repeating: 0, count: width * height) }
        return cellCenters(imageSize: imageSize, width: width, height: height).map { pt in
            var inside = false
            var j = polygon.count - 1
            for i in polygon.indices {
                let a = polygon[i], b = polygon[j]
                if (a.y > pt.y) != (b.y > pt.y), pt.x < (b.x - a.x) * (pt.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
                j = i
            }
            return inside ? 1 : 0
        }
    }

    /// Two passes of a box blur (≈ Gaussian feathering).
    static func feather(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return values }
        func blur(_ v: [Float]) -> [Float] {
            var tmp = [Float](repeating: 0, count: v.count), out = tmp
            for y in 0..<height {
                for x in 0..<width {
                    var s: Float = 0, n: Float = 0
                    for dx in -radius...radius where x + dx >= 0 && x + dx < width { s += v[y * width + x + dx]; n += 1 }
                    tmp[y * width + x] = s / n
                }
            }
            for y in 0..<height {
                for x in 0..<width {
                    var s: Float = 0, n: Float = 0
                    for dy in -radius...radius where y + dy >= 0 && y + dy < height { s += tmp[(y + dy) * width + x]; n += 1 }
                    out[y * width + x] = s / n
                }
            }
            return out
        }
        return blur(blur(values))
    }

    /// Vision person matte as a raster (nil when unavailable).
    static func personMatte(_ photo: SubjectPhoto) -> RasterMask? {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cgImage: photo.image, options: [:])
        guard (try? handler.perform([request])) != nil, let buffer = request.results?.first?.pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var values = [Float](repeating: 0, count: w * h)
        for y in 0..<h { for x in 0..<w { values[y * w + x] = Float(bytes[y * stride + x]) / 255 } }
        return RasterMask(width: w, height: h, imageSize: CGSize(width: photo.width, height: photo.height), values: values)
    }
}

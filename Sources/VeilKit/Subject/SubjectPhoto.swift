//
//  SubjectPhoto.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit: ReferenceImage), GPL-3.0.
//
//  A photo, content-addressed: its identity is the SHA-256 of the source bytes. Reports carry
//  that hash, never pixels; Veil never copies the original file anywhere.
//

import CoreGraphics
import Foundation
import ImageIO

public enum SubjectPhotoError: Error, LocalizedError {
    case unreadable(String)
    case empty(String)
    case looseImages(String, Int)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let s): return "Could not read image \(s)"
        case .empty(let s): return "No images found in \(s)"
        case .looseImages(let s, let n):
            return """
                \(n) image(s) sit directly in \(s), so Veil doesn't know whose they are. Put each person's photos in a \
                subfolder named after them (e.g. \(s)/Jane Doe/…).
                """
        }
    }
}

public final class SubjectPhoto: @unchecked Sendable {
    /// SHA-256 hex of the source bytes (or of the rendered pixels for derived images).
    public let id: String
    public let name: String
    /// Orientation-corrected image.
    public let image: CGImage

    public var width: Int { image.width }
    public var height: Int { image.height }
    public var bounds: CGRect { CGRect(x: 0, y: 0, width: width, height: height) }

    public static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "webp", "bmp", "gif"]

    public static func load(url: URL) throws -> SubjectPhoto {
        guard let data = try? Data(contentsOf: url) else { throw SubjectPhotoError.unreadable(url.path) }
        return try SubjectPhoto(data: data, name: url.lastPathComponent)
    }

    /// Every image file directly inside `directory`, sorted by name.
    public static func loadDirectory(_ directory: URL) throws -> [SubjectPhoto] {
        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.naturalLess($1.lastPathComponent) }
        return try urls.map(load(url:))
    }

    /// Files and directories (a directory contributes its images), in order.
    public static func load(paths: [URL]) throws -> [SubjectPhoto] {
        var out: [SubjectPhoto] = []
        for url in paths {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
                throw SubjectPhotoError.unreadable(url.path)
            }
            if isDir.boolValue { out += try loadDirectory(url) } else { out.append(try load(url: url)) }
        }
        return out
    }

    public init(data: Data, name: String = "photo") throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            throw SubjectPhotoError.unreadable(name)
        }
        let w = props[kCGImagePropertyPixelWidth] as? Int ?? 0
        let h = props[kCGImagePropertyPixelHeight] as? Int ?? 0
        // A full-size "thumbnail" with the EXIF transform applied gives an upright image.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(w, h, 1),
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw SubjectPhotoError.unreadable(name)
        }
        self.id = Hashing.sha256Hex(data)
        self.name = name
        self.image = image
    }

    public init(image: CGImage, name: String = "photo") {
        self.image = image
        self.name = name
        self.id = Hashing.sha256Hex(Data(SubjectPhoto.render(image, rect: nil, width: image.width, height: image.height)))
    }

    /// RGBA8 (sRGB, premultiplied) of `rect` resampled to width×height, rows top to bottom.
    public func rgba(rect: CGRect? = nil, width: Int, height: Int) -> [UInt8] {
        Self.render(image, rect: rect, width: width, height: height)
    }

    /// Planar RGB floats in [0, 1], shape (3, height, width).
    public func rgbPlanar(rect: CGRect? = nil, width: Int, height: Int) -> [Float] {
        let px = rgba(rect: rect, width: width, height: height)
        let n = width * height
        var out = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            out[i] = Float(px[4 * i]) / 255
            out[n + i] = Float(px[4 * i + 1]) / 255
            out[2 * n + i] = Float(px[4 * i + 2]) / 255
        }
        return out
    }

    static func render(_ image: CGImage, rect: CGRect?, width: Int, height: Int, flip: Bool = false) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let source: CGImage
        if let rect, let cropped = image.cropping(to: rect.integral) {
            source = cropped
        } else {
            source = image
        }
        pixels.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.setFillColor(CGColor(gray: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            if flip {
                ctx.translateBy(x: CGFloat(width), y: 0)
                ctx.scaleBy(x: -1, y: 1)
            }
            ctx.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return pixels
    }
}

// MARK: - Augmentation (single-photo subjects)

public enum PhotoAugment {
    /// Deterministic variants of one photo — mirror, and keyed crops of 86–94% of the frame —
    /// used as the held-out set when only one photo exists. Flagged in every report: they
    /// share the photo's pose, lighting and background, so they overstate generalization.
    public static func variants(of photo: SubjectPhoto, count: Int, schedule: SeedSchedule) -> [SubjectPhoto] {
        var rng = schedule.rng(.augment, photo.id)
        var out: [SubjectPhoto] = []
        for i in 0..<count {
            let flip = i % 2 == 0
            let frac = 0.86 + 0.08 * rng.unit()
            let w = Double(photo.width) * frac, h = Double(photo.height) * frac
            let x = (Double(photo.width) - w) * rng.unit(), y = (Double(photo.height) - h) * rng.unit()
            let rect = CGRect(x: x, y: y, width: w, height: h)
            let (ow, oh) = (Int(w.rounded()), Int(h.rounded()))
            let px = SubjectPhoto.render(photo.image, rect: rect, width: ow, height: oh, flip: flip)
            if let image = ImageWriter.image(rgba: px, width: ow, height: oh) {
                out.append(SubjectPhoto(image: image, name: "\(photo.name)#aug\(i)"))
            }
        }
        return out
    }
}

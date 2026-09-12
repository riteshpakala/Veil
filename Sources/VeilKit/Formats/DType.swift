//
//  DType.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  Tensor element types as they appear in safetensors headers, plus decoding of raw
//  little-endian bytes to Float. Half and fp8 are decoded by bit manipulation so the
//  path works regardless of platform Float16 support.
//

import Foundation

public enum TensorDType: String, Codable, Sendable, CaseIterable {
    case f64 = "F64"
    case f32 = "F32"
    case f16 = "F16"
    case bf16 = "BF16"
    case f8e4m3 = "F8_E4M3"
    case f8e5m2 = "F8_E5M2"
    case i64 = "I64"
    case i32 = "I32"
    case i16 = "I16"
    case i8 = "I8"
    case u64 = "U64"
    case u32 = "U32"
    case u16 = "U16"
    case u8 = "U8"
    case bool = "BOOL"

    public var byteSize: Int {
        switch self {
        case .f64, .i64, .u64: return 8
        case .f32, .i32, .u32: return 4
        case .f16, .bf16, .i16, .u16: return 2
        case .f8e4m3, .f8e5m2, .i8, .u8, .bool: return 1
        }
    }

    public var isFloatingPoint: Bool {
        switch self {
        case .f64, .f32, .f16, .bf16, .f8e4m3, .f8e5m2: return true
        default: return false
        }
    }
}

public enum DTypeError: Error, LocalizedError {
    case unsupported(String)
    case sizeMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let t): return "Unsupported tensor dtype \(t)"
        case .sizeMismatch(let e, let a): return "Tensor byte size mismatch: expected \(e), got \(a)"
        }
    }
}

public enum TensorDecoding {
    /// Decode `count` elements of `dtype` from little-endian `data` into Float.
    public static func floats(from data: Data, dtype: TensorDType, count: Int) throws -> [Float] {
        let expected = count * dtype.byteSize
        guard data.count == expected else {
            throw DTypeError.sizeMismatch(expected: expected, actual: data.count)
        }
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            switch dtype {
            case .f32:
                for i in 0..<count {
                    out[i] = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
                }
            case .f64:
                for i in 0..<count {
                    out[i] = Float(Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self))))
                }
            case .f16:
                for i in 0..<count {
                    out[i] = halfToFloat(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)))
                }
            case .bf16:
                for i in 0..<count {
                    let bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    out[i] = Float(bitPattern: UInt32(bits) << 16)
                }
            case .f8e4m3:
                let lut = f8e4m3LUT
                for i in 0..<count { out[i] = lut[Int(raw[i])] }
            case .f8e5m2:
                let lut = f8e5m2LUT
                for i in 0..<count { out[i] = lut[Int(raw[i])] }
            case .i8:
                for i in 0..<count { out[i] = Float(Int8(bitPattern: raw[i])) }
            case .u8, .bool:
                for i in 0..<count { out[i] = Float(raw[i]) }
            case .i16:
                for i in 0..<count {
                    out[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self)))
                }
            case .u16:
                for i in 0..<count {
                    out[i] = Float(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)))
                }
            case .i32:
                for i in 0..<count {
                    out[i] = Float(Int32(littleEndian: raw.loadUnaligned(fromByteOffset: i * 4, as: Int32.self)))
                }
            case .u32:
                for i in 0..<count {
                    out[i] = Float(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
                }
            case .i64:
                for i in 0..<count {
                    out[i] = Float(Int64(littleEndian: raw.loadUnaligned(fromByteOffset: i * 8, as: Int64.self)))
                }
            case .u64:
                for i in 0..<count {
                    out[i] = Float(UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self)))
                }
            }
        }
        return out
    }

    /// IEEE 754 binary16 → binary32.
    public static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = Int((h >> 10) & 0x1F)
        let mant = UInt32(h & 0x03FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            // Subnormal: value = mant * 2^-24
            let v = Float(mant) * Float(sign: .plus, exponent: -24, significand: 1)
            return sign != 0 ? -v : v
        }
        if exp == 0x1F {
            return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13))   // inf / NaN
        }
        return Float(bitPattern: sign | UInt32(exp - 15 + 127) << 23 | (mant << 13))
    }

    /// binary32 → binary16 (round-to-nearest-even). Used by fixture writers and the bank.
    public static func floatToHalf(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF)
        var mant = bits & 0x007F_FFFF
        if exp == 0xFF { return sign | 0x7C00 | (mant != 0 ? 0x200 : 0) }
        let e = exp - 127 + 15
        if e >= 0x1F { return sign | 0x7C00 }
        if e <= 0 {
            if e < -10 { return sign }
            mant |= 0x0080_0000
            let shift = UInt32(14 - e)
            var half = UInt16(mant >> shift)
            let rem = mant & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if rem > halfway || (rem == halfway && (half & 1) == 1) { half += 1 }
            return sign | half
        }
        var half = UInt16(e << 10) | UInt16(mant >> 13)
        let rem = mant & 0x1FFF
        // A carry out of the mantissa bumps the exponent; at the top it lands on inf.
        if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { half += 1 }
        return sign | half
    }

    /// float8 e4m3fn (PyTorch `float8_e4m3fn`): bias 7, no infinities, NaN = S.1111.111.
    static let f8e4m3LUT: [Float] = (0..<256).map { i in
        let b = UInt8(i)
        let sign: Float = (b & 0x80) != 0 ? -1 : 1
        let exp = Int((b >> 3) & 0x0F)
        let mant = Float(b & 0x07)
        if exp == 0x0F && (b & 0x07) == 0x07 { return .nan }
        if exp == 0 { return sign * (mant / 8) * pow(2, -6) }
        return sign * (1 + mant / 8) * pow(2, Float(exp - 7))
    }

    /// float8 e5m2: bias 15, IEEE-style infinities/NaN.
    static let f8e5m2LUT: [Float] = (0..<256).map { i in
        let b = UInt8(i)
        let sign: Float = (b & 0x80) != 0 ? -1 : 1
        let exp = Int((b >> 2) & 0x1F)
        let mant = Float(b & 0x03)
        if exp == 0x1F { return mant == 0 ? sign * .infinity : .nan }
        if exp == 0 { return sign * (mant / 4) * pow(2, -14) }
        return sign * (1 + mant / 4) * pow(2, Float(exp - 15))
    }
}

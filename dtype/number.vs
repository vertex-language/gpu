// Number: the element types a kernel computes with, and what every
// function over them needs beyond arithmetic and order.
import "gpu"

/// Number is a type a kernel computes with: float32, float16, bfloat16,
/// int32 or uint32. A function written once
/// over a Number -- gpu/parallel's reductions, scans and sorts -- is built
/// for each type it is used at, on the device and on the host.
public protocol Number: Numeric, Comparable {
    /// Lowest is the least value: the identity of a maximum.
    static func Lowest() -> Self
    /// Highest is the greatest value: the identity of a minimum.
    static func Highest() -> Self
    /// Plus is a + b, wrapping around for an integer type rather than
    /// trapping, as a sum on a GPU does.
    static func Plus(_ a: Self, _ b: Self) -> Self
    /// OrderKey is the uint32 whose unsigned order is this type's total
    /// order: what a radix sort sorts. For a float, IEEE 754's total order
    /// (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN).
    static func OrderKey(_ x: Self) -> uint32
    /// FromOrderKey is the value OrderKey made k from.
    static func FromOrderKey(_ k: uint32) -> Self
    /// AtomicAdd adds v to *p as one indivisible step, and returns what
    /// *p was.
    static func AtomicAdd(_ p: UnsafeMutablePointer<Self>, _ v: Self) -> Self
    /// IsInteger is whether the type holds whole numbers only.
    static func IsInteger() -> bool
    /// ToFloat64 and FromFloat64 are conversions for the host: a device
    /// may have no float64.
    static func ToFloat64(_ x: Self) -> float64
    static func FromFloat64(_ x: float64) -> Self
    /// Accumulator is what a long sum of this type is taken in -- a dot
    /// product's, a matmul's: float32 for a half, which would lose the
    /// sum's low bits after a few hundred terms, and the type itself
    /// otherwise. Widen and Narrow go to it and back, Narrow rounding once.
    associatedtype Accumulator: Number
    static func Widen(_ x: Self) -> Accumulator
    static func Narrow(_ x: Accumulator) -> Self
}

extension float32: Number {
    @inlinable public static func Lowest() -> float32 { return -float32.infinity }
    @inlinable public static func Highest() -> float32 { return float32.infinity }
    @inlinable public static func Plus(_ a: float32, _ b: float32) -> float32 { return a + b }
    @inlinable public static func OrderKey(_ x: float32) -> uint32 {
        let b = x.bitPattern
        return (b & 0x80000000) != 0 ? ~b : (b | 0x80000000)
    }
    @inlinable public static func FromOrderKey(_ k: uint32) -> float32 {
        return float32(bitPattern: (k & 0x80000000) != 0 ? (k & 0x7FFFFFFF) : ~k)
    }
    @inlinable public static func AtomicAdd(_ p: UnsafeMutablePointer<float32>, _ v: float32) -> float32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(_ x: float32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(_ x: float64) -> float32 { return float32(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(_ x: float32) -> float32 { return x }
    @inlinable public static func Narrow(_ x: float32) -> float32 { return x }
}

extension int32: Number {
    @inlinable public static func Lowest() -> int32 { return int32.min }
    @inlinable public static func Highest() -> int32 { return int32.max }
    @inlinable public static func Plus(_ a: int32, _ b: int32) -> int32 { return a &+ b }
    @inlinable public static func OrderKey(_ x: int32) -> uint32 { return uint32(bitPattern: x) ^ 0x80000000 }
    @inlinable public static func FromOrderKey(_ k: uint32) -> int32 { return int32(bitPattern: k ^ 0x80000000) }
    @inlinable public static func AtomicAdd(_ p: UnsafeMutablePointer<int32>, _ v: int32) -> int32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return true }
    @inlinable public static func ToFloat64(_ x: int32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(_ x: float64) -> int32 { return int32(x) }
    public typealias Accumulator = int32
    @inlinable public static func Widen(_ x: int32) -> int32 { return x }
    @inlinable public static func Narrow(_ x: int32) -> int32 { return x }
}

extension uint32: Number {
    @inlinable public static func Lowest() -> uint32 { return 0 }
    @inlinable public static func Highest() -> uint32 { return uint32.max }
    @inlinable public static func Plus(_ a: uint32, _ b: uint32) -> uint32 { return a &+ b }
    @inlinable public static func OrderKey(_ x: uint32) -> uint32 { return x }
    @inlinable public static func FromOrderKey(_ k: uint32) -> uint32 { return k }
    @inlinable public static func AtomicAdd(_ p: UnsafeMutablePointer<uint32>, _ v: uint32) -> uint32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return true }
    @inlinable public static func ToFloat64(_ x: uint32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(_ x: float64) -> uint32 { return uint32(x) }
    public typealias Accumulator = uint32
    @inlinable public static func Widen(_ x: uint32) -> uint32 { return x }
    @inlinable public static func Narrow(_ x: uint32) -> uint32 { return x }
}

extension float16: Number {
    @inlinable public static func Lowest() -> float16 { return -float16.infinity }
    @inlinable public static func Highest() -> float16 { return float16.infinity }
    @inlinable public static func Plus(_ a: float16, _ b: float16) -> float16 { return a + b }
    /// A half's total order is float32's on its 16 bits.
    @inlinable public static func OrderKey(_ x: float16) -> uint32 {
        let b = uint32(x.bitPattern)
        return (b & 0x8000) != 0 ? (~b & 0xFFFF) : (b | 0x8000)
    }
    @inlinable public static func FromOrderKey(_ k: uint32) -> float16 {
        let b = (k & 0x8000) != 0 ? (k & 0x7FFF) : (~k & 0xFFFF)
        return float16(bitPattern: uint16(truncatingIfNeeded: b))
    }
    @inlinable public static func AtomicAdd(_ p: UnsafeMutablePointer<float16>, _ v: float16) -> float16 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(_ x: float16) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(_ x: float64) -> float16 { return float16(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(_ x: float16) -> float32 { return float32(x) }
    @inlinable public static func Narrow(_ x: float32) -> float16 { return float16(x) }
}

extension bfloat16: Number {
    @inlinable public static func Lowest() -> bfloat16 { return -bfloat16.infinity }
    @inlinable public static func Highest() -> bfloat16 { return bfloat16.infinity }
    @inlinable public static func Plus(_ a: bfloat16, _ b: bfloat16) -> bfloat16 { return a + b }
    /// A half's total order is float32's on its 16 bits.
    @inlinable public static func OrderKey(_ x: bfloat16) -> uint32 {
        let b = uint32(x.bitPattern)
        return (b & 0x8000) != 0 ? (~b & 0xFFFF) : (b | 0x8000)
    }
    @inlinable public static func FromOrderKey(_ k: uint32) -> bfloat16 {
        let b = (k & 0x8000) != 0 ? (k & 0x7FFF) : (~k & 0xFFFF)
        return bfloat16(bitPattern: uint16(truncatingIfNeeded: b))
    }
    @inlinable public static func AtomicAdd(_ p: UnsafeMutablePointer<bfloat16>, _ v: bfloat16) -> bfloat16 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(_ x: bfloat16) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(_ x: float64) -> bfloat16 { return bfloat16(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(_ x: bfloat16) -> float32 { return float32(x) }
    @inlinable public static func Narrow(_ x: float32) -> bfloat16 { return bfloat16(x) }
}

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
    static func Plus(a: Self, b: Self) -> Self
    /// OrderKey is the uint32 whose unsigned order is this type's total
    /// order: what a radix sort sorts. For a float, IEEE 754's total order
    /// (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN).
    static func OrderKey(x: Self) -> uint32
    /// FromOrderKey is the value OrderKey made k from.
    static func FromOrderKey(k: uint32) -> Self
    /// AtomicAdd adds v to *p as one indivisible step, and returns what
    /// *p was.
    static func AtomicAdd(p: UnsafeMutablePointer<Self>, v: Self) -> Self
    /// IsInteger is whether the type holds whole numbers only.
    static func IsInteger() -> bool
    /// ToFloat64 and FromFloat64 are conversions for the host: a device
    /// may have no float64.
    static func ToFloat64(x: Self) -> float64
    static func FromFloat64(x: float64) -> Self
    /// Accumulator is what a long sum of this type is taken in -- a dot
    /// product's, a matmul's: float32 for a half, which would lose the
    /// sum's low bits after a few hundred terms, and the type itself
    /// otherwise. Widen and Narrow go to it and back, Narrow rounding once.
    associatedtype Accumulator: Number
    static func Widen(x: Self) -> Accumulator
    static func Narrow(x: Accumulator) -> Self
}

extension float32: Number {
    @inlinable public static func Lowest() -> float32 { return -float32.infinity }
    @inlinable public static func Highest() -> float32 { return float32.infinity }
    @inlinable public static func Plus(a: float32, b: float32) -> float32 { return a + b }
    @inlinable public static func OrderKey(x: float32) -> uint32 {
        let b = x.bitPattern
        return (b & 0x80000000) != 0 ? ~b : (b | 0x80000000)
    }
    @inlinable public static func FromOrderKey(k: uint32) -> float32 {
        return float32(bitPattern: (k & 0x80000000) != 0 ? (k & 0x7FFFFFFF) : ~k)
    }
    @inlinable public static func AtomicAdd(p: UnsafeMutablePointer<float32>, v: float32) -> float32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(x: float32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(x: float64) -> float32 { return float32(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(x: float32) -> float32 { return x }
    @inlinable public static func Narrow(x: float32) -> float32 { return x }
}

extension int32: Number {
    @inlinable public static func Lowest() -> int32 { return int32.min }
    @inlinable public static func Highest() -> int32 { return int32.max }
    @inlinable public static func Plus(a: int32, b: int32) -> int32 { return a &+ b }
    @inlinable public static func OrderKey(x: int32) -> uint32 { return uint32(bitPattern: x) ^ 0x80000000 }
    @inlinable public static func FromOrderKey(k: uint32) -> int32 { return int32(bitPattern: k ^ 0x80000000) }
    @inlinable public static func AtomicAdd(p: UnsafeMutablePointer<int32>, v: int32) -> int32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return true }
    @inlinable public static func ToFloat64(x: int32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(x: float64) -> int32 { return int32(x) }
    public typealias Accumulator = int32
    @inlinable public static func Widen(x: int32) -> int32 { return x }
    @inlinable public static func Narrow(x: int32) -> int32 { return x }
}

extension uint32: Number {
    @inlinable public static func Lowest() -> uint32 { return 0 }
    @inlinable public static func Highest() -> uint32 { return uint32.max }
    @inlinable public static func Plus(a: uint32, b: uint32) -> uint32 { return a &+ b }
    @inlinable public static func OrderKey(x: uint32) -> uint32 { return x }
    @inlinable public static func FromOrderKey(k: uint32) -> uint32 { return k }
    @inlinable public static func AtomicAdd(p: UnsafeMutablePointer<uint32>, v: uint32) -> uint32 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return true }
    @inlinable public static func ToFloat64(x: uint32) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(x: float64) -> uint32 { return uint32(x) }
    public typealias Accumulator = uint32
    @inlinable public static func Widen(x: uint32) -> uint32 { return x }
    @inlinable public static func Narrow(x: uint32) -> uint32 { return x }
}

extension float16: Number {
    @inlinable public static func Lowest() -> float16 { return -float16.infinity }
    @inlinable public static func Highest() -> float16 { return float16.infinity }
    @inlinable public static func Plus(a: float16, b: float16) -> float16 { return a + b }
    /// A half's total order is float32's on its 16 bits.
    @inlinable public static func OrderKey(x: float16) -> uint32 {
        let b = uint32(x.bitPattern)
        return (b & 0x8000) != 0 ? (~b & 0xFFFF) : (b | 0x8000)
    }
    @inlinable public static func FromOrderKey(k: uint32) -> float16 {
        let b = (k & 0x8000) != 0 ? (k & 0x7FFF) : (~k & 0xFFFF)
        return float16(bitPattern: uint16(truncatingIfNeeded: b))
    }
    @inlinable public static func AtomicAdd(p: UnsafeMutablePointer<float16>, v: float16) -> float16 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(x: float16) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(x: float64) -> float16 { return float16(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(x: float16) -> float32 { return float32(x) }
    @inlinable public static func Narrow(x: float32) -> float16 { return float16(x) }
}

extension bfloat16: Number {
    @inlinable public static func Lowest() -> bfloat16 { return -bfloat16.infinity }
    @inlinable public static func Highest() -> bfloat16 { return bfloat16.infinity }
    @inlinable public static func Plus(a: bfloat16, b: bfloat16) -> bfloat16 { return a + b }
    /// A half's total order is float32's on its 16 bits.
    @inlinable public static func OrderKey(x: bfloat16) -> uint32 {
        let b = uint32(x.bitPattern)
        return (b & 0x8000) != 0 ? (~b & 0xFFFF) : (b | 0x8000)
    }
    @inlinable public static func FromOrderKey(k: uint32) -> bfloat16 {
        let b = (k & 0x8000) != 0 ? (k & 0x7FFF) : (~k & 0xFFFF)
        return bfloat16(bitPattern: uint16(truncatingIfNeeded: b))
    }
    @inlinable public static func AtomicAdd(p: UnsafeMutablePointer<bfloat16>, v: bfloat16) -> bfloat16 {
        return gpu.Atomic.Add(p, v)
    }
    @inlinable public static func IsInteger() -> bool { return false }
    @inlinable public static func ToFloat64(x: bfloat16) -> float64 { return float64(x) }
    @inlinable public static func FromFloat64(x: float64) -> bfloat16 { return bfloat16(x) }
    public typealias Accumulator = float32
    @inlinable public static func Widen(x: bfloat16) -> float32 { return float32(x) }
    @inlinable public static func Narrow(x: float32) -> bfloat16 { return bfloat16(x) }
}

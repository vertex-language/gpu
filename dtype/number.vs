// Number: the element types a kernel computes with, and what every
// function over them needs beyond arithmetic and order.
import "gpu"

/// Number is a type a kernel computes with: float32, int32 or uint32
/// today, and the half floats once vsc has them. A function written once
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
}

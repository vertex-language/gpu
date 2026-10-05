// Package raster turns triangles, lines and points into pixels: vertex
// fetch, vertex shading, clipping, rasterization, fragment shading,
// depth, stencil and blending. Today it draws on the CPU, shading with
// shader/interp; the Metal path comes behind the same API.
//
// A draw's shaders come compiled at run time (shader.Module, from GLSL
// through shader/glsl), which is what gles and WebGL hand it. Images are
// Textures; a window's surface is a Target whose rows are stored top row
// first (FlipY), as the window shows them.
package raster

import (
    "gpu"
)

/// How a texture's texels are stored.
public enum Format: Equatable {
    /// 8-bit unsigned normalized RGBA: 0…255 reads as 0…1.
    case rgba8
    /// 32-bit float RGBA.
    case rgba32f
    /// 32-bit float depth, and an 8-bit stencil beside it.
    case depthStencil
}

/// One mip level of one face: Width × Height texels, row 0 first. Its
/// storage is memory it owns, read and written in place (pointers, not
/// arrays: a draw touches every texel, and arrays would copy).
public final class Level {
    public let Width: int
    public let Height: int
    public let Format: Format
    /// rgba8: 4 bytes a texel (ByteCount of them).
    public let Bytes: UnsafeMutablePointer<uint8>
    public let ByteCount: int
    /// rgba32f: 4 floats a texel; depthStencil: 1 float a texel (FloatCount of them).
    public let Floats: UnsafeMutablePointer<float32>
    public let FloatCount: int
    /// depthStencil: 1 byte a texel.
    public let Stencil: UnsafeMutablePointer<uint8>
    public let StencilCount: int

    public init(width: int, height: int, format: Format) {
        Width = max(0, width)
        Height = max(0, height)
        Format = format
        let n = Width * Height
        ByteCount = format == .rgba8 ? n * 4 : 0
        FloatCount = format == .rgba32f ? n * 4 : (format == .depthStencil ? n : 0)
        StencilCount = format == .depthStencil ? n : 0
        Bytes = UnsafeMutablePointer<uint8>.allocate(capacity: max(1, ByteCount))
        Bytes.initialize(repeating: 0, count: max(1, ByteCount))
        Floats = UnsafeMutablePointer<float32>.allocate(capacity: max(1, FloatCount))
        Floats.initialize(repeating: format == .depthStencil ? 1 : 0, count: max(1, FloatCount))
        Stencil = UnsafeMutablePointer<uint8>.allocate(capacity: max(1, StencilCount))
        Stencil.initialize(repeating: 0, count: max(1, StencilCount))
    }

    deinit {
        Bytes.deallocate()
        Floats.deallocate()
        Stencil.deallocate()
    }

    /// The bytes, copied out (rgba8).
    public var ByteArray: [uint8] {
        var out = [uint8](repeating: 0, count: ByteCount)
        if ByteCount == 0 { return out }
        out.withUnsafeMutableBufferPointer { p in
            UnsafeMutableRawPointer(p.baseAddress!).copyMemory(from: UnsafeRawPointer(Bytes), byteCount: ByteCount)
        }
        return out
    }

    /// Copies `b` in from the start (rgba8).
    public func SetBytes(_ b: [uint8]) {
        let n = min(b.count, ByteCount)
        if n == 0 { return }
        b.withUnsafeBufferPointer { p in
            UnsafeMutableRawPointer(Bytes).copyMemory(from: UnsafeRawPointer(p.baseAddress!), byteCount: n)
        }
    }
}

/// A texture: 2D, or a cube's six faces, each with its mip levels.
///
/// On a GPU a texture also has a copy in device memory. The levels here
/// are the host's copy; whoever writes them calls Changed after, and
/// whoever reads them calls Synchronize before, so the two copies agree:
/// what the GPU drew is read back, what the host wrote is sent again.
public final class Texture {
    public let Format: Format
    public let Cube: bool
    /// Faces[face][level]; one face unless Cube.
    public var Faces: [[Level]] = []
    /// The device copy, and whether it or the host's is newer.
    var device: gpu.Texture? = nil
    var hostNewer = true
    var deviceNewer = false

    public init(format: Format, cube: bool = false) {
        Format = format
        Cube = cube
        // vsc_TODO #47: no [[Level]](repeating:count:).
        for _ in 0..<(cube ? 6 : 1) { Faces.append([]) }
    }

    /// A 2D texture of one level.
    public convenience init(width: int, height: int, format: Format) {
        self.init(format: format)
        Faces[0] = [Level(width: width, height: height, format: format)]
    }

    public var Width: int { Faces[0].first?.Width ?? 0 }
    public var Height: int { Faces[0].first?.Height ?? 0 }

    /// Brings the host's levels up to date with what the GPU drew. Call it
    /// before reading Level storage.
    public func Synchronize() {
        if !deviceNewer { return }
        deviceNewer = false
        guard let d = device, Format != .depthStencil else { return }
        metal?.endPass()
        metal?.stats.Readbacks += 1
        for (f, levels) in Faces.enumerated() {
            for (k, lv) in levels.enumerated() where k < d.Levels {
                let bpp = Format == .rgba8 ? 4 : 16
                let p = Format == .rgba8 ? UnsafeMutableRawPointer(lv.Bytes) : UnsafeMutableRawPointer(lv.Floats)
                d.Read(level: k, slice: f, x: 0, y: 0, width: lv.Width, height: lv.Height, into: p, bytesPerRow: lv.Width * bpp)
            }
        }
    }

    /// A level to read: brought up to date with what the GPU drew.
    public func Read(face: int = 0, level: int) -> Level? {
        Synchronize()
        return At(face: face, level: level)
    }

    /// A level to write: up to date, and sent to the GPU again before it's next drawn with.
    public func Write(face: int = 0, level: int) -> Level? {
        Synchronize()
        hostNewer = true
        return At(face: face, level: level)
    }

    deinit {
        // Releasing a device texture finishes the GPU's work: the open pass must end first.
        if device != nil { metal?.endPass() }
    }

    /// Says the host's levels changed: the GPU's copy is sent again before it's next used.
    public func Changed() {
        hostNewer = true
        deviceNewer = false
    }

    /// Level `level` of `face`, made (or remade) at this size.
    public func Define(face: int = 0, level: int, width: int, height: int) -> Level {
        Synchronize()
        hostNewer = true
        while Faces[face].count <= level {
            let k = Faces[face].count
            Faces[face].append(Level(width: max(1, Width >> k), height: max(1, Height >> k), format: Format))
        }
        let l = Level(width: width, height: height, format: Format)
        Faces[face][level] = l
        return l
    }

    public func At(face: int = 0, level: int) -> Level? {
        if face >= Faces.count || level >= Faces[face].count { return nil }
        return Faces[face][level]
    }

    /// Whether every level from 0 down to 1×1 is there, each half the one before.
    public var Complete: bool {
        guard let base = At(level: 0), base.Width > 0, base.Height > 0 else { return false }
        for f in Faces {
            if f.isEmpty || f[0].Width != base.Width || f[0].Height != base.Height { return false }
        }
        return true
    }

    /// Whether every level of a full mip chain is defined at the right size.
    public var MipmapComplete: bool {
        guard Complete else { return false }
        let levels = MipLevels(Width, Height)
        for f in Faces {
            if f.count < levels { return false }
            for k in 0..<levels where f[k].Width != max(1, Width >> k) || f[k].Height != max(1, Height >> k) { return false }
        }
        return true
    }

    /// Fills levels 1… from level 0 by averaging 2×2 blocks (glGenerateMipmap).
    public func GenerateMipmaps() {
        Synchronize()
        hostNewer = true
        let levels = MipLevels(Width, Height)
        for face in 0..<Faces.count {
            for k in 1..<max(1, levels) {
                guard let src = At(face: face, level: k - 1) else { break }
                let dst = Define(face: face, level: k, width: max(1, src.Width / 2), height: max(1, src.Height / 2))
                for y in 0..<dst.Height {
                    for x in 0..<dst.Width {
                        let x0 = min(src.Width - 1, x * 2)
                        let x1 = min(src.Width - 1, x * 2 + 1)
                        let y0 = min(src.Height - 1, y * 2)
                        let y1 = min(src.Height - 1, y * 2 + 1)
                        for c in 0..<4 {
                            if Format == .rgba8 {
                                let s = int(src.Bytes[(y0 * src.Width + x0) * 4 + c]) + int(src.Bytes[(y0 * src.Width + x1) * 4 + c]) +
                                    int(src.Bytes[(y1 * src.Width + x0) * 4 + c]) + int(src.Bytes[(y1 * src.Width + x1) * 4 + c])
                                dst.Bytes[(y * dst.Width + x) * 4 + c] = uint8((s + 2) / 4)
                            } else if Format == .rgba32f {
                                let s = src.Floats[(y0 * src.Width + x0) * 4 + c] + src.Floats[(y0 * src.Width + x1) * 4 + c] +
                                    src.Floats[(y1 * src.Width + x0) * 4 + c] + src.Floats[(y1 * src.Width + x1) * 4 + c]
                                dst.Floats[(y * dst.Width + x) * 4 + c] = s / 4
                            }
                        }
                    }
                }
            }
        }
    }
}

/// How many levels a full mip chain of this size has.
public func MipLevels(_ w: int, _ h: int) -> int {
    var n = 1
    var s = max(w, h)
    while s > 1 {
        s /= 2
        n += 1
    }
    return n
}

/// How texels between and beyond texel centers are read.
public enum Filter: Equatable {
    case nearest
    case linear
}

/// What coordinates outside 0…1 read.
public enum Wrap: Equatable {
    case repeating
    case clampToEdge
    case mirroredRepeat
}

/// Where a lookup's component comes from (GL's texture swizzle; how
/// luminance and alpha textures read).
public enum Swizzle: Equatable {
    case red, green, blue, alpha, zero, one
}

/// How a texture is read.
public struct Sampler {
    public var MinFilter: Filter = .nearest
    public var MagFilter: Filter = .linear
    /// Between levels: nil reads the base level only.
    public var MipFilter: Filter? = nil
    public var WrapS: Wrap = .repeating
    public var WrapT: Wrap = .repeating
    public var WrapR: Wrap = .repeating
    public var Swizzle: [Swizzle] = [.red, .green, .blue, .alpha]
    public var BaseLevel = 0
    public var MaxLevel = 1000
    public var MinLod: float32 = -1000
    public var MaxLod: float32 = 1000
    /// For shadow samplers: compare the reference with the stored depth.
    public var Compare: CompareFunc? = nil

    public init() {}
}

/// A comparison of a new value with a stored one: depth, stencil and shadow tests.
public enum CompareFunc: Equatable {
    case never, less, equal, lessEqual, greater, notEqual, greaterEqual, always

    public func Passes(_ a: float32, _ b: float32) -> bool {
        switch self {
        case .never: return false
        case .less: return a < b
        case .equal: return a == b
        case .lessEqual: return a <= b
        case .greater: return a > b
        case .notEqual: return a != b
        case .greaterEqual: return a >= b
        case .always: return true
        }
    }
}

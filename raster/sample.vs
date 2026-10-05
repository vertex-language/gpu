package raster

import (
    "math"
    "shader"
    "shader/interp"
)

/// A texture and how to read it, bound to a unit.
public struct Binding {
    public var Texture: Texture?
    public var Sampler: Sampler

    public init(texture: Texture?, sampler: Sampler) {
        Texture = texture
        Sampler = sampler
    }
}

/// The textures a draw reads, by unit: the shader's lookups come here.
final class Units: interp.Textures {
    var bindings: [Binding?]

    init() {
        bindings = []   // vsc_TODO #35
    }

    func binding(_ unit: int32) -> Binding? {
        let u = int(unit)
        if u < 0 || u >= bindings.count { return nil }
        return bindings[u]
    }

    func Sample(_ r: interp.SampleRequest) {
        // Lanes usually all read one unit: resolve it once, then each lane.
        var done: uint64 = 0
        let active = r.Mask
        for l0 in 0..<r.Lanes where active & (uint64(1) << uint64(l0)) != 0 && done & (uint64(1) << uint64(l0)) == 0 {
            let unit = r.Unit[l0]
            var lanes: uint64 = 0
            for l in l0..<r.Lanes where active & (uint64(1) << uint64(l)) != 0 && r.Unit[l] == unit {
                lanes |= uint64(1) << uint64(l)
            }
            done |= lanes
            guard let b = binding(unit), let t = b.Texture, t.Complete else {
                // An incomplete texture reads (0, 0, 0, 1).
                for l in 0..<r.Lanes where lanes & (uint64(1) << uint64(l)) != 0 {
                    r.Result[l * 4] = 0
                    r.Result[l * 4 + 1] = 0
                    r.Result[l * 4 + 2] = 0
                    r.Result[l * 4 + 3] = float32(1).bitPattern
                }
                continue
            }
            if !sampleFast(r, lanes, t, b.Sampler) {
                sampleGeneral(r, lanes, t, b.Sampler)
            }
        }
    }

    /// The common case, without levels of detail: an RGBA8 2D texture read
    /// at one level, one filter, no compare or swizzle. Returns false when
    /// the texture or sampler isn't that.
    func sampleFast(_ r: interp.SampleRequest, _ lanes: uint64, _ t: Texture, _ smp: Sampler) -> bool {
        if t.Format != .rgba8 || t.Cube || smp.Compare != nil || smp.MinFilter != smp.MagFilter { return false }
        let base = max(0, min(smp.BaseLevel, t.Faces[0].count - 1))
        let lastLevel = min(t.Faces[0].count - 1, smp.MaxLevel)
        if smp.MipFilter != nil && lastLevel > base { return false }
        if smp.Swizzle[0] != .red || smp.Swizzle[1] != .green || smp.Swizzle[2] != .blue || smp.Swizzle[3] != .alpha { return false }
        guard let lv = t.At(level: base) else { return false }
        let w = lv.Width
        let h = lv.Height
        let fw = float32(w)
        let fh = float32(h)
        let linear = smp.MagFilter == .linear
        let ws = smp.WrapS
        let wt = smp.WrapT
        let k: float32 = 1.0 / 255.0
        let coords = r.Coords
        let result = r.Result
        let count = r.Lanes
        let px = lv.Bytes
        for l in 0..<count where lanes & (uint64(1) << uint64(l)) != 0 {
            let u = coords[l * 4] * fw
            let v = coords[l * 4 + 1] * fh
            var c0: float32 = 0
            var c1: float32 = 0
            var c2: float32 = 0
            var c3: float32 = 0
            if !linear {
                let x = wrap(int(u.rounded(.down)), w, ws)
                let y = wrap(int(v.rounded(.down)), h, wt)
                let i = (y * w + x) * 4
                c0 = float32(px[i]) * k
                c1 = float32(px[i + 1]) * k
                c2 = float32(px[i + 2]) * k
                c3 = float32(px[i + 3]) * k
            } else {
                let uf = u - 0.5
                let vf = v - 0.5
                let x0f = uf.rounded(.down)
                let y0f = vf.rounded(.down)
                let a = uf - x0f
                let b = vf - y0f
                let x0 = wrap(int(x0f), w, ws)
                let x1 = wrap(int(x0f) + 1, w, ws)
                let y0 = wrap(int(y0f), h, wt)
                let y1 = wrap(int(y0f) + 1, h, wt)
                let i00 = (y0 * w + x0) * 4
                let i10 = (y0 * w + x1) * 4
                let i01 = (y1 * w + x0) * 4
                let i11 = (y1 * w + x1) * 4
                let w00 = (1 - a) * (1 - b) * k
                let w10 = a * (1 - b) * k
                let w01 = (1 - a) * b * k
                let w11 = a * b * k
                c0 = float32(px[i00]) * w00 + float32(px[i10]) * w10 + float32(px[i01]) * w01 + float32(px[i11]) * w11
                c1 = float32(px[i00 + 1]) * w00 + float32(px[i10 + 1]) * w10 + float32(px[i01 + 1]) * w01 + float32(px[i11 + 1]) * w11
                c2 = float32(px[i00 + 2]) * w00 + float32(px[i10 + 2]) * w10 + float32(px[i01 + 2]) * w01 + float32(px[i11 + 2]) * w11
                c3 = float32(px[i00 + 3]) * w00 + float32(px[i10 + 3]) * w10 + float32(px[i01 + 3]) * w01 + float32(px[i11 + 3]) * w11
            }
            result[l * 4] = c0.bitPattern
            result[l * 4 + 1] = c1.bitPattern
            result[l * 4 + 2] = c2.bitPattern
            result[l * 4 + 3] = c3.bitPattern
        }
        return true
    }

    /// Any texture: cube faces, levels of detail, compares and swizzles.
    func sampleGeneral(_ r: interp.SampleRequest, _ lanes: uint64, _ t: Texture, _ smp: Sampler) {
        for l in 0..<r.Lanes where lanes & (uint64(1) << uint64(l)) != 0 {
            var s = r.Coords[l * 4]
            var tc = r.Coords[l * 4 + 1]
            var face = 0
            var dsdx = r.Ddx[l * 4]
            var dtdx = r.Ddx[l * 4 + 1]
            var dsdy = r.Ddy[l * 4]
            var dtdy = r.Ddy[l * 4 + 1]
            if t.Cube {
                let c = cubeFace(r.Coords[l * 4], r.Coords[l * 4 + 1], r.Coords[l * 4 + 2])
                face = c.Face
                s = c.S
                tc = c.T
                // Derivatives on the face: approximated by the coordinate scale.
                dsdx = r.Ddx[l * 4] * c.Scale
                dtdx = r.Ddx[l * 4 + 1] * c.Scale
                dsdy = r.Ddy[l * 4] * c.Scale
                dtdy = r.Ddy[l * 4 + 1] * c.Scale
            }
            let base = max(0, min(smp.BaseLevel, t.Faces[face].count - 1))
            guard let l0 = t.At(face: face, level: base) else { continue }
            // The level of detail.
            var lambda: float32 = 0
            switch r.Lookup {
            case .level:
                lambda = r.LodOrBias[l]
            default:
                if r.HasDerivatives {
                    let w = float32(l0.Width)
                    let h = float32(l0.Height)
                    let px = ((dsdx * w) * (dsdx * w) + (dtdx * h) * (dtdx * h)).squareRoot()
                    let py = ((dsdy * w) * (dsdy * w) + (dtdy * h) * (dtdy * h)).squareRoot()
                    let rho = max(px, py)
                    lambda = rho > 0 ? math.Log2(rho) : -1000
                }
                if r.Lookup == .bias { lambda += r.LodOrBias[l] }
            }
            lambda = min(max(lambda, smp.MinLod), smp.MaxLod)
            var out = Color()
            let magnify = lambda <= 0
            let filter = magnify ? smp.MagFilter : smp.MinFilter
            let lastLevel = min(t.Faces[face].count - 1, smp.MaxLevel)
            if magnify || smp.MipFilter == nil || lastLevel <= base {
                out = filtered(t, face: face, level: base, s, tc, filter, smp)
            } else if smp.MipFilter == .nearest {
                let k = min(lastLevel, base + int((lambda + 0.5).rounded(.down)))
                out = filtered(t, face: face, level: max(base, k), s, tc, filter, smp)
            } else {
                let lo = min(lastLevel, base + int(lambda.rounded(.down)))
                let hi = min(lastLevel, lo + 1)
                let frac = lambda - lambda.rounded(.down)
                let a = filtered(t, face: face, level: lo, s, tc, filter, smp)
                let c = filtered(t, face: face, level: hi, s, tc, filter, smp)
                out = a.Mix(c, frac)
            }
            // A shadow lookup compares its reference (the coordinate after s and t) with the depth.
            if let cmp = smp.Compare, t.Format == .depthStencil {
                let ref = r.Coords[l * 4 + 2]
                let v: float32 = cmp.Passes(min(max(ref, 0), 1), out.R) ? 1 : 0
                out = Color(v, v, v, 1)
            }
            let sw = smp.Swizzle
            for k in 0..<4 {
                var v: float32 = 0
                switch sw[k] {
                case .red: v = out.R
                case .green: v = out.G
                case .blue: v = out.B
                case .alpha: v = out.A
                case .zero: v = 0
                case .one: v = 1
                }
                r.Result[l * 4 + k] = v.bitPattern
            }
            if r.Sampler == .shadow2D || r.Sampler == .shadowCube || r.Sampler == .shadowArray2D {
                r.Result[l * 4] = out.R.bitPattern
            }
        }
    }

    func Fetch(_ r: interp.SampleRequest) {
        for l in 0..<r.Lanes where r.Mask & (uint64(1) << uint64(l)) != 0 {
            guard let b = binding(r.Unit[l]), let t = b.Texture else { continue }
            let level = int(r.LodOrBias[l])
            let x = int(int32(bitPattern: r.Coords[l * 4].bitPattern))
            let y = int(int32(bitPattern: r.Coords[l * 4 + 1].bitPattern))
            guard let lv = t.At(level: level), x >= 0, y >= 0, x < lv.Width, y < lv.Height else {
                for k in 0..<4 { r.Result[l * 4 + k] = 0 }
                continue
            }
            let v = texel(t, lv, x, y)
            r.Result[l * 4] = v.R.bitPattern
            r.Result[l * 4 + 1] = v.G.bitPattern
            r.Result[l * 4 + 2] = v.B.bitPattern
            r.Result[l * 4 + 3] = v.A.bitPattern
        }
    }

    func Size(unit: int, sampler: shader.SamplerKind, level: int) -> [int32] {
        guard let b = binding(int32(unit)), let t = b.Texture, let lv = t.At(level: max(0, level)) else { return [0, 0, 0] }
        return [int32(lv.Width), int32(lv.Height), 1]
    }
}

/// Four float components.
public struct Color {
    public var R: float32 = 0
    public var G: float32 = 0
    public var B: float32 = 0
    public var A: float32 = 1

    public init() {}

    public init(_ r: float32, _ g: float32, _ b: float32, _ a: float32) {
        R = r
        G = g
        B = b
        A = a
    }

    /// This color moved `t` of the way to `o`.
    public func Mix(_ o: Color, _ t: float32) -> Color {
        Color(R + (o.R - R) * t, G + (o.G - G) * t, B + (o.B - B) * t, A + (o.A - A) * t)
    }

    public func Scaled(_ k: float32) -> Color { Color(R * k, G * k, B * k, A * k) }
    public func Plus(_ o: Color) -> Color { Color(R + o.R, G + o.G, B + o.B, A + o.A) }
}

/// A texel's components.
func texel(_ t: Texture, _ lv: Level, _ x: int, _ y: int) -> Color {
    let i = y * lv.Width + x
    switch t.Format {
    case .rgba8:
        let k: float32 = 1.0 / 255.0
        return Color(float32(lv.Bytes[i * 4]) * k, float32(lv.Bytes[i * 4 + 1]) * k,
                     float32(lv.Bytes[i * 4 + 2]) * k, float32(lv.Bytes[i * 4 + 3]) * k)
    case .rgba32f:
        return Color(lv.Floats[i * 4], lv.Floats[i * 4 + 1], lv.Floats[i * 4 + 2], lv.Floats[i * 4 + 3])
    case .depthStencil:
        return Color(lv.Floats[i], float32(lv.Stencil[i]), 0, 1)
    }
}

func wrap(_ i: int, _ size: int, _ w: Wrap) -> int {
    switch w {
    case .clampToEdge:
        return max(0, min(size - 1, i))
    case .repeating:
        let m = i % size
        return m < 0 ? m + size : m
    case .mirroredRepeat:
        let period = 2 * size
        var m = i % period
        if m < 0 { m += period }
        return m < size ? m : period - 1 - m
    }
}

/// A filtered read of one level at normalized (s, t).
func filtered(_ t: Texture, face: int, level: int, _ s: float32, _ tc: float32, _ f: Filter, _ smp: Sampler) -> Color {
    guard let lv = t.At(face: face, level: level), lv.Width > 0, lv.Height > 0 else { return Color(0, 0, 0, 1) }
    let u = s * float32(lv.Width)
    let v = tc * float32(lv.Height)
    if f == .nearest {
        let x = wrap(int(u.rounded(.down)), lv.Width, smp.WrapS)
        let y = wrap(int(v.rounded(.down)), lv.Height, smp.WrapT)
        return texel(t, lv, x, y)
    }
    let uf = u - 0.5
    let vf = v - 0.5
    let x0f = uf.rounded(.down)
    let y0f = vf.rounded(.down)
    let a = uf - x0f
    let b = vf - y0f
    let x0 = wrap(int(x0f), lv.Width, smp.WrapS)
    let x1 = wrap(int(x0f) + 1, lv.Width, smp.WrapS)
    let y0 = wrap(int(y0f), lv.Height, smp.WrapT)
    let y1 = wrap(int(y0f) + 1, lv.Height, smp.WrapT)
    let t00 = texel(t, lv, x0, y0)
    let t10 = texel(t, lv, x1, y0)
    let t01 = texel(t, lv, x0, y1)
    let t11 = texel(t, lv, x1, y1)
    return t00.Mix(t10, a).Mix(t01.Mix(t11, a), b)
}

/// A cube lookup's face and its 2D coordinates (OpenGL ES 3.0 table 3.21).
struct CubeHit {
    let Face: int
    let S: float32
    let T: float32
    /// How much a step in the direction moves on the face, for derivatives.
    let Scale: float32
}

func cubeFace(_ x: float32, _ y: float32, _ z: float32) -> CubeHit {
    let ax = abs(x)
    let ay = abs(y)
    let az = abs(z)
    var face = 0
    var sc: float32 = 0
    var tc: float32 = 0
    var ma: float32 = 1
    if ax >= ay && ax >= az {
        face = x >= 0 ? 0 : 1
        sc = x >= 0 ? -z : z
        tc = -y
        ma = ax
    } else if ay >= az {
        face = y >= 0 ? 2 : 3
        sc = x
        tc = y >= 0 ? z : -z
        ma = ay
    } else {
        face = z >= 0 ? 4 : 5
        sc = z >= 0 ? x : -x
        tc = -y
        ma = az
    }
    if ma == 0 { ma = 1 }
    return CubeHit(Face: face, S: (sc / ma + 1) / 2, T: (tc / ma + 1) / 2, Scale: 1 / (2 * ma))
}

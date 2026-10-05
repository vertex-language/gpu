package raster

import (
    "shader/interp"
)

extension Renderer {
    /// Shades the batch's quads and writes what passes the tests.
    func flush(_ p: Program, _ fm: interp.Machine, _ st: State, _ t: Target, front: bool) {
        if quads == 0 { return }
        let e = p.Fragment
        let lanes = quads * 4
        let A = triA
        let B = triB
        let C = triC
        // Slots and batch buffers as locals: these loops run per pixel.
        let sl = fm.Slots
        let L = fm.Lanes
        let lam = lambda
        let pxs = bx
        let pys = by
        let links = p.links
        let vb = vary
        let prov = triProvoking
        let fragCoord = e.FragCoord
        let frontFacing = e.FrontFacing
        let pointCoord = e.PointCoord
        let isPoint = batchPoint
        for l in 0..<lanes {
            let la = lam[l * 3]
            let lb = lam[l * 3 + 1]
            let lc = lam[l * 3 + 2]
            // Perspective-correct weights.
            let wa = la * A.IW
            let wb = lb * B.IW
            let wc = lc * C.IW
            let iw = wa + wb + wc
            let inv = iw != 0 ? 1 / iw : 0
            var at = 0
            for link in links {
                for k in 0..<link.Slots {
                    let slot = link.To + k
                    if link.Flat || link.Integer {
                        sl[slot * L + l] = vb[prov + at]
                    } else {
                        let va = float64(float32(bitPattern: vb[A.Vary + at]))
                        let vbb = float64(float32(bitPattern: vb[B.Vary + at]))
                        let vc = float64(float32(bitPattern: vb[C.Vary + at]))
                        sl[slot * L + l] = float32((va * wa + vbb * wb + vc * wc) * inv).bitPattern
                    }
                    at += 1
                }
            }
            if fragCoord >= 0 {
                sl[fragCoord * L + l] = (float32(pxs[l]) + 0.5).bitPattern
                sl[(fragCoord + 1) * L + l] = (float32(pys[l]) + 0.5).bitPattern
                sl[(fragCoord + 2) * L + l] = float32(la * A.Z + lb * B.Z + lc * C.Z).bitPattern
                sl[(fragCoord + 3) * L + l] = float32(iw).bitPattern
            }
            if frontFacing >= 0 { sl[frontFacing * L + l] = front ? 1 : 0 }
            if pointCoord >= 0 {
                sl[pointCoord * L + l] = (isPoint ? float32(la * A.PS + lb * B.PS + lc * C.PS) : 0).bitPattern
                sl[(pointCoord + 1) * L + l] = (isPoint ? float32(la * A.PT + lb * B.PT + lc * C.PT) : 0).bitPattern
            }
        }
        // Helper lanes run too, so derivatives see whole quads.
        let all: uint64 = lanes == 64 ? ~uint64(0) : (uint64(1) << uint64(lanes)) - 1
        let alive = fm.Run(all)
        FragmentsShaded += lanes
        let color = t.colorLevel
        let depth = t.depthLevel
        let colorOut = e.FragColor >= 0 ? e.FragColor : (e.FragData >= 0 ? e.FragData : (e.Outputs.first(where: { $0.Location <= 0 })?.Offset ?? -1))
        let stencilFace = front ? st.StencilFront : st.StencilBack
        let targetHeight = t.Height
        let allChannels = st.WriteRed && st.WriteGreen && st.WriteBlue && st.WriteAlpha
        let colorFloat = t.Color?.Format == .rgba32f
        for l in 0..<lanes where cover & alive & (uint64(1) << uint64(l)) != 0 {
            let x = pxs[l]
            let y = pys[l]
            let row = t.FlipY ? targetHeight - 1 - y : y
            var pass = true
            if let ds = depth {
                let di = row * ds.Width + x
                var z: float64 = 0
                if e.FragDepth >= 0 && e.Module.WritesDepth {
                    z = float64(fm.GetFloat(e.FragDepth, lane: l))
                } else {
                    z = lam[l * 3] * A.Z + lam[l * 3 + 1] * B.Z + lam[l * 3 + 2] * C.Z + triOffset
                }
                let zf = float32(min(max(z, 0), 1))
                if st.StencilTest {
                    let stored = uint32(ds.Stencil[di])
                    let ref = uint32(truncatingIfNeeded: max(0, min(255, stencilFace.Ref)))
                    let m = stencilFace.ReadMask & 0xFF
                    if !stencilFace.Func.Passes(float32(ref & m), float32(stored & m)) {
                        ds.Stencil[di] = stencilUpdate(stencilFace.Fail, ds.Stencil[di], ref, stencilFace.WriteMask)
                        continue
                    }
                }
                if let dt = st.DepthTest {
                    pass = dt.Passes(zf, ds.Floats[di])
                }
                if st.StencilTest {
                    let ref = uint32(truncatingIfNeeded: max(0, min(255, stencilFace.Ref)))
                    ds.Stencil[di] = stencilUpdate(pass ? stencilFace.Pass : stencilFace.DepthFail, ds.Stencil[di], ref, stencilFace.WriteMask)
                }
                if !pass { continue }
                if st.DepthTest != nil && st.DepthWrite { ds.Floats[di] = zf }
            }
            guard let cl = color, colorOut >= 0 else { continue }
            let r0 = float32(bitPattern: sl[colorOut * L + l])
            let g0 = float32(bitPattern: sl[(colorOut + 1) * L + l])
            let b0 = float32(bitPattern: sl[(colorOut + 2) * L + l])
            let a0 = float32(bitPattern: sl[(colorOut + 3) * L + l])
            let ci = (row * cl.Width + x) * 4
            if !colorFloat && !st.Blend && allChannels {
                // The common case: no blending into an RGBA8 target.
                let out = cl.Bytes
                out[ci] = r0 >= 1 ? 255 : (r0 > 0 ? uint8(r0 * 255 + 0.5) : 0)
                out[ci + 1] = g0 >= 1 ? 255 : (g0 > 0 ? uint8(g0 * 255 + 0.5) : 0)
                out[ci + 2] = b0 >= 1 ? 255 : (b0 > 0 ? uint8(b0 * 255 + 0.5) : 0)
                out[ci + 3] = a0 >= 1 ? 255 : (a0 > 0 ? uint8(a0 * 255 + 0.5) : 0)
                continue
            }
            var src = Color(r0, g0, b0, a0)
            let isFloat = colorFloat
            var dst = Color()
            if isFloat {
                dst = Color(cl.Floats[ci], cl.Floats[ci + 1], cl.Floats[ci + 2], cl.Floats[ci + 3])
            } else {
                let k: float32 = 1.0 / 255.0
                dst = Color(float32(cl.Bytes[ci]) * k, float32(cl.Bytes[ci + 1]) * k, float32(cl.Bytes[ci + 2]) * k, float32(cl.Bytes[ci + 3]) * k)
                // A fixed-point target stores what's in 0…1.
                src = Color(clamp01(src.R), clamp01(src.G), clamp01(src.B), clamp01(src.A))
            }
            if st.Blend { src = blend(src, dst, st) }
            if isFloat {
                if st.WriteRed { cl.Floats[ci] = src.R }
                if st.WriteGreen { cl.Floats[ci + 1] = src.G }
                if st.WriteBlue { cl.Floats[ci + 2] = src.B }
                if st.WriteAlpha { cl.Floats[ci + 3] = src.A }
            } else {
                if st.WriteRed { cl.Bytes[ci] = unorm8(src.R) }
                if st.WriteGreen { cl.Bytes[ci + 1] = unorm8(src.G) }
                if st.WriteBlue { cl.Bytes[ci + 2] = unorm8(src.B) }
                if st.WriteAlpha { cl.Bytes[ci + 3] = unorm8(src.A) }
            }
        }
        quads = 0
        cover = 0
    }
}

func clamp01(_ v: float32) -> float32 { v.isNaN ? 0 : min(max(v, 0), 1) }

/// A float in 0…1 as an 8-bit unsigned normalized value, rounded to nearest.
func unorm8(_ v: float32) -> uint8 {
    uint8(clamp01(v) * 255 + 0.5)
}

func stencilUpdate(_ op: StencilOp, _ s: uint8, _ ref: uint32, _ writeMask: uint32) -> uint8 {
    var v: uint32 = uint32(s)
    switch op {
    case .keep: return s
    case .zero: v = 0
    case .replace: v = ref
    case .increment: v = min(255, v + 1)
    case .incrementWrap: v = (v + 1) & 0xFF
    case .decrement: v = v == 0 ? 0 : v - 1
    case .decrementWrap: v = (v &- 1) & 0xFF
    case .invert: v = ~v & 0xFF
    }
    let m = writeMask & 0xFF
    return uint8((uint32(s) & ~m) | (v & m))
}

/// The factor's four components.
func factors(_ f: BlendFactor, _ s: Color, _ d: Color, _ k: Color) -> Color {
    switch f {
    case .zero: return Color(0, 0, 0, 0)
    case .one: return Color(1, 1, 1, 1)
    case .srcColor: return s
    case .oneMinusSrcColor: return Color(1 - s.R, 1 - s.G, 1 - s.B, 1 - s.A)
    case .dstColor: return d
    case .oneMinusDstColor: return Color(1 - d.R, 1 - d.G, 1 - d.B, 1 - d.A)
    case .srcAlpha: return Color(s.A, s.A, s.A, s.A)
    case .oneMinusSrcAlpha: return Color(1 - s.A, 1 - s.A, 1 - s.A, 1 - s.A)
    case .dstAlpha: return Color(d.A, d.A, d.A, d.A)
    case .oneMinusDstAlpha: return Color(1 - d.A, 1 - d.A, 1 - d.A, 1 - d.A)
    case .constantColor: return k
    case .oneMinusConstantColor: return Color(1 - k.R, 1 - k.G, 1 - k.B, 1 - k.A)
    case .constantAlpha: return Color(k.A, k.A, k.A, k.A)
    case .oneMinusConstantAlpha: return Color(1 - k.A, 1 - k.A, 1 - k.A, 1 - k.A)
    case .srcAlphaSaturate:
        let f = min(s.A, 1 - d.A)
        return Color(f, f, f, 1)
    }
}

func combine(_ eq: BlendEquation, _ s: float32, _ fs: float32, _ d: float32, _ fd: float32) -> float32 {
    switch eq {
    case .add: return s * fs + d * fd
    case .subtract: return s * fs - d * fd
    case .reverseSubtract: return d * fd - s * fs
    case .min: return min(s, d)
    case .max: return max(s, d)
    }
}

func blend(_ s: Color, _ d: Color, _ st: State) -> Color {
    let k = st.BlendColor
    let fsRGB = factors(st.SrcRGB, s, d, k)
    let fdRGB = factors(st.DstRGB, s, d, k)
    let fsA = factors(st.SrcAlpha, s, d, k)
    let fdA = factors(st.DstAlpha, s, d, k)
    return Color(combine(st.BlendRGB, s.R, fsRGB.R, d.R, fdRGB.R),
                 combine(st.BlendRGB, s.G, fsRGB.G, d.G, fdRGB.G),
                 combine(st.BlendRGB, s.B, fsRGB.B, d.B, fdRGB.B),
                 combine(st.BlendAlpha, s.A, fsA.A, d.A, fdA.A))
}

/// One attribute of one vertex, as four components' bits (missing ones
/// from (0, 0, 0, 1)); floats unless the shader input is an integer.
func readAttribute(_ a: Attribute, _ b: [uint8], _ index: int, integerInput: bool) -> [uint32] {
    let size: int
    switch a.Type {
    case .float32, .int32, .uint32, .fixed: size = 4
    case .float16, .int16, .uint16: size = 2
    case .int8, .uint8: size = 1
    }
    let stride = a.Stride > 0 ? a.Stride : size * a.Count
    let base = a.Offset + index * stride
    var out: [uint32] = [0, 0, 0, integerInput ? 1 : float32(1).bitPattern]
    for k in 0..<min(4, a.Count) {
        let at = base + k * size
        if at < 0 || at + size > b.count { continue }
        var raw: uint32 = 0
        for j in 0..<size { raw |= uint32(b[at + j]) << uint32(8 * j) }
        if integerInput || a.Integer {
            switch a.Type {
            case .int8: out[k] = uint32(bitPattern: int32(int8(truncatingIfNeeded: raw)))
            case .int16: out[k] = uint32(bitPattern: int32(int16(truncatingIfNeeded: raw)))
            default: out[k] = raw
            }
            continue
        }
        var v: float32 = 0
        switch a.Type {
        case .float32: v = float32(bitPattern: raw)
        case .float16: v = float32(float16(bitPattern: uint16(raw)))
        case .fixed: v = float32(int32(bitPattern: raw)) / 65536
        case .int8:
            let x = float32(int8(truncatingIfNeeded: raw))
            v = a.Normalized ? max(x / 127, -1) : x
        case .uint8:
            v = a.Normalized ? float32(raw) / 255 : float32(raw)
        case .int16:
            let x = float32(int16(truncatingIfNeeded: raw))
            v = a.Normalized ? max(x / 32767, -1) : x
        case .uint16:
            v = a.Normalized ? float32(raw) / 65535 : float32(raw)
        case .int32:
            let x = float32(int32(bitPattern: raw))
            v = a.Normalized ? max(x / 2147483647, -1) : x
        case .uint32:
            v = a.Normalized ? float32(raw) / 4294967295 : float32(raw)
        }
        out[k] = v.bitPattern
    }
    return out
}

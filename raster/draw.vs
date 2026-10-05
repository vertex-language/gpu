package raster

import (
    "shader"
    "shader/interp"
)

/// A vertex in window coordinates (y up), ready to rasterize.
struct WinVert {
    var X: float64
    var Y: float64
    var Z: float64
    /// 1/w of the clip position, for perspective-correct interpolation.
    var IW: float64
    /// Where its varyings start in the renderer's varying store.
    var Vary: int
    /// gl_PointCoord, for points.
    var PS: float64 = 0
    var PT: float64 = 0
}

/// A vertex in clip coordinates.
struct ClipVert {
    var X: float32
    var Y: float32
    var Z: float32
    var W: float32
    var Vary: int
}

/// Draws into Targets on the CPU, shading with shader/interp. One
/// renderer serves one thread; it keeps its scratch space between draws.
public final class Renderer {
    /// Lanes a fragment batch shades together (four per 2×2 quad).
    public let Lanes = 16
    let units = Units()
    /// Varyings: shaded vertices' first, then vertices clipping makes.
    var vary: [uint32] = []
    var clip: [ClipVert] = []
    var pointSize: [float32] = []
    var varyingSlots = 0
    var shadedCount = 0
    // The fragment batch being filled (pointers: written per pixel).
    let bx: UnsafeMutablePointer<int>
    let by: UnsafeMutablePointer<int>
    var cover: uint64 = 0
    let lambda: UnsafeMutablePointer<float64>
    var quads = 0
    /// Counters, for tests and profiling.
    public private(set) var FragmentsShaded = 0

    public init() {
        bx = UnsafeMutablePointer<int>.allocate(capacity: 64)
        bx.initialize(repeating: 0, count: 64)
        by = UnsafeMutablePointer<int>.allocate(capacity: 64)
        by.initialize(repeating: 0, count: 64)
        lambda = UnsafeMutablePointer<float64>.allocate(capacity: 64 * 3)
        lambda.initialize(repeating: 0, count: 64 * 3)
    }

    deinit {
        bx.deallocate()
        by.deallocate()
        lambda.deallocate()
    }

    // MARK: clearing

    /// Clears a target's color, depth and stencil inside `scissor`,
    /// honoring the state's color and stencil write masks.
    public func Clear(_ t: Target, color: Color?, depth: float32?, stencil: int?, state: State) {
        if let m = metal {
            if m.Clear(t, color: color, depth: depth, stencil: stencil, state: state) { return }
            m.toHost(t, [])
        }
        let w = t.Width
        let h = t.Height
        var x0 = 0
        var y0 = 0
        var x1 = w
        var y1 = h
        if let s = state.Scissor {
            x0 = max(x0, s.X)
            y0 = max(y0, s.Y)
            x1 = min(x1, s.X + s.Width)
            y1 = min(y1, s.Y + s.Height)
        }
        if x0 >= x1 || y0 >= y1 { return }
        if let c = color, let lv = t.colorLevel {
            let fmt = t.Color!.Format
            let rgba = [c.R, c.G, c.B, c.A]
            let mask = [state.WriteRed, state.WriteGreen, state.WriteBlue, state.WriteAlpha]
            var bytes = [uint8](repeating: 0, count: 4)
            for k in 0..<4 { bytes[k] = unorm8(rgba[k]) }
            for y in y0..<y1 {
                let row = t.FlipY ? lv.Height - 1 - y : y
                for x in x0..<x1 {
                    let i = (row * lv.Width + x) * 4
                    for k in 0..<4 where mask[k] {
                        if fmt == .rgba8 { lv.Bytes[i + k] = bytes[k] } else { lv.Floats[i + k] = rgba[k] }
                    }
                }
            }
        }
        if let ds = t.depthLevel {
            for y in y0..<y1 {
                let row = t.FlipY ? ds.Height - 1 - y : y
                for x in x0..<x1 {
                    let i = row * ds.Width + x
                    if let d = depth { ds.Floats[i] = min(max(d, 0), 1) }
                    if let s = stencil {
                        let wm = uint8(truncatingIfNeeded: state.StencilFront.WriteMask)
                        ds.Stencil[i] = (ds.Stencil[i] & ~wm) | (uint8(truncatingIfNeeded: s) & wm)
                    }
                }
            }
        }
    }

    // MARK: drawing

    /// Draws `d` with `program` into `target`.
    public func Draw(_ d: Draw, program: Program, state: State, target: Target, textures: [Binding?]) {
        if d.Count <= 0 || target.Width <= 0 || target.Height <= 0 { return }
        if let m = metal {
            if m.Draw(d, program: program, state: state, target: target, textures: textures) { return }
            m.toHost(target, textures)
        }
        units.bindings = textures
        let vm = program.vertexMachine ?? interp.Machine(program.Vertex, lanes: Lanes)
        let fm = program.fragmentMachine ?? interp.Machine(program.Fragment, lanes: Lanes)
        program.vertexMachine = vm
        program.fragmentMachine = fm
        vm.Textures = units
        fm.Textures = units
        loadUniforms(vm, program.Vertex, program.VertexUniforms)
        loadUniforms(fm, program.Fragment, program.FragmentUniforms)
        varyingSlots = program.varyingSlots

        // The vertices in draw order (indices resolved); -1 restarts a strip.
        var order: [int] = []
        order.reserveCapacity(d.Count)
        if let ib = d.Indices {
            let b = ib.Bytes
            for k in 0..<d.Count {
                let at = d.IndexOffset + (d.First + k) * d.IndexSize
                if at + d.IndexSize > b.count { break }
                var v: uint32 = 0
                for j in 0..<d.IndexSize { v |= uint32(b[at + j]) << uint32(8 * j) }
                if let r = d.RestartIndex, v == r { order.append(-1) } else { order.append(int(v)) }
            }
        } else {
            for k in 0..<d.Count { order.append(d.First + k) }
        }

        for instance in 0..<max(1, d.Instances) {
            // Shade each distinct vertex once.
            var slotOf: [int: int] = [:]
            var unique: [int] = []
            var shaded: [int] = []
            shaded.reserveCapacity(order.count)
            for v in order {
                if v < 0 { shaded.append(-1); continue }
                if let s = slotOf[v] {
                    shaded.append(s)
                } else {
                    slotOf[v] = unique.count
                    shaded.append(unique.count)
                    unique.append(v)
                }
            }
            shadeVertices(unique, instance: instance, d, program, vm)
            assemble(shaded, d.Primitive, program, fm, state, target)
        }
    }

    func loadUniforms(_ m: interp.Machine, _ e: interp.Executable, _ values: [uint32]) {
        for u in e.Uniforms {
            let n = max(1, e.Module.Slots(u.Type))
            for k in 0..<n where u.Offset + k < values.count {
                m.Set(u.Offset + k, all: values[u.Offset + k])
            }
        }
    }

    // MARK: vertices

    func shadeVertices(_ ids: [int], instance: int, _ d: Draw, _ p: Program, _ vm: interp.Machine) {
        let e = p.Vertex
        let L = Lanes
        let V = varyingSlots
        vary = [uint32](repeating: 0, count: ids.count * V)
        clip = []
        clip.reserveCapacity(ids.count)
        pointSize = [float32](repeating: 1, count: ids.count)
        shadedCount = ids.count
        var start = 0
        while start < ids.count {
            let n = min(L, ids.count - start)
            for lane in 0..<n {
                let id = ids[start + lane]
                for (k, input) in e.Inputs.enumerated() {
                    fetch(input, location: p.InputLocations[k], vertex: id, instance: instance, d, vm, lane)
                }
                if e.VertexId >= 0 { vm.Set(e.VertexId, lane: lane, uint32(bitPattern: int32(id))) }
                if e.InstanceId >= 0 { vm.Set(e.InstanceId, lane: lane, uint32(bitPattern: int32(instance))) }
            }
            let mask: uint64 = n == 64 ? ~uint64(0) : (uint64(1) << uint64(n)) - 1
            _ = vm.Run(mask)
            for lane in 0..<n {
                let v = start + lane
                if e.Position >= 0 {
                    clip.append(ClipVert(X: vm.GetFloat(e.Position, lane: lane), Y: vm.GetFloat(e.Position + 1, lane: lane),
                                         Z: vm.GetFloat(e.Position + 2, lane: lane), W: vm.GetFloat(e.Position + 3, lane: lane), Vary: v * V))
                } else {
                    clip.append(ClipVert(X: 0, Y: 0, Z: 0, W: 1, Vary: v * V))
                }
                if e.PointSize >= 0 { pointSize[v] = vm.GetFloat(e.PointSize, lane: lane) }
                var at = v * V
                for link in p.links {
                    for k in 0..<link.Slots {
                        vary[at] = vm.Get(link.From + k, lane: lane)
                        at += 1
                    }
                }
            }
            start += n
        }
    }

    /// Sets one vertex input of one lane from its attribute.
    func fetch(_ input: interp.Interface, location: int, vertex: int, instance: int, _ d: Draw, _ m: interp.Machine, _ lane: int) {
        let t = input.Type
        let columns = t.IsMatrix ? t.Columns : 1
        let rows = t.IsMatrix ? t.Rows : max(1, t.Rows)
        for c in 0..<columns {
            let loc = location + c
            var comps: [uint32] = [0, 0, 0, float32(1).bitPattern]
            if loc < d.Attributes.count {
                let a = d.Attributes[loc]
                if a.Enabled, let buf = a.Buffer {
                    let index = a.Divisor > 0 ? instance / a.Divisor : vertex
                    comps = readAttribute(a, buf.Bytes, index, integerInput: t.Kind == .int || t.Kind == .uint)
                } else {
                    comps = a.Constant
                }
            }
            for r in 0..<rows {
                m.Set(input.Offset + c * rows + r, lane: lane, comps[r])
            }
        }
    }

    // MARK: primitives

    func assemble(_ s: [int], _ prim: Primitive, _ p: Program, _ fm: interp.Machine, _ st: State, _ t: Target) {
        // Split at restarts.
        var runs: [[int]] = [[]]
        for v in s {
            if v < 0 { runs.append([]) } else { runs[runs.count - 1].append(v) }
        }
        for r in runs {
            let n = r.count
            switch prim {
            case .triangles:
                var k = 0
                while k + 2 < n {
                    triangle(r[k], r[k + 1], r[k + 2], provoking: r[k + 2], p, fm, st, t)
                    k += 3
                }
            case .triangleStrip:
                if n >= 3 {
                    for k in 0..<(n - 2) {
                        if k % 2 == 0 {
                            triangle(r[k], r[k + 1], r[k + 2], provoking: r[k + 2], p, fm, st, t)
                        } else {
                            triangle(r[k + 1], r[k], r[k + 2], provoking: r[k + 2], p, fm, st, t)
                        }
                    }
                }
            case .triangleFan:
                if n >= 3 {
                    for k in 1..<(n - 1) {
                        triangle(r[0], r[k], r[k + 1], provoking: r[k + 1], p, fm, st, t)
                    }
                }
            case .lines:
                var k = 0
                while k + 1 < n {
                    line(r[k], r[k + 1], p, fm, st, t)
                    k += 2
                }
            case .lineStrip, .lineLoop:
                if n >= 2 {
                    for k in 0..<(n - 1) { line(r[k], r[k + 1], p, fm, st, t) }
                    if prim == .lineLoop && n > 2 { line(r[n - 1], r[0], p, fm, st, t) }
                }
            case .points:
                for v in r { point(v, p, fm, st, t) }
            }
        }
        flush(p, fm, st, t, front: true)
    }

    /// The window position of a clip-space vertex.
    func window(_ c: ClipVert, _ st: State) -> WinVert {
        let iw = c.W != 0 ? 1.0 / float64(c.W) : 0
        let nx = float64(c.X) * iw
        let ny = float64(c.Y) * iw
        let nz = float64(c.Z) * iw
        let vp = st.Viewport
        let x = (nx + 1) * float64(vp.Width) * 0.5 + float64(vp.X)
        let y = (ny + 1) * float64(vp.Height) * 0.5 + float64(vp.Y)
        let z = nz * float64(st.DepthFar - st.DepthNear) * 0.5 + float64(st.DepthFar + st.DepthNear) * 0.5
        return WinVert(X: x, Y: y, Z: z, IW: iw, Vary: c.Vary)
    }

    // MARK: clipping

    /// A vertex part of the way from a to b, its varyings interpolated.
    func between(_ a: ClipVert, _ b: ClipVert, _ t: float32) -> ClipVert {
        let at = vary.count
        for k in 0..<varyingSlots {
            let x = float32(bitPattern: vary[a.Vary + k])
            let y = float32(bitPattern: vary[b.Vary + k])
            vary.append((x + (y - x) * t).bitPattern)
        }
        return ClipVert(X: a.X + (b.X - a.X) * t, Y: a.Y + (b.Y - a.Y) * t, Z: a.Z + (b.Z - a.Z) * t,
                        W: a.W + (b.W - a.W) * t, Vary: at)
    }

    /// The polygon left after clipping against near (z ≥ -w) and far (z ≤ w).
    func clipPolygon(_ poly: [ClipVert]) -> [ClipVert] {
        var p = poly
        for plane in 0..<2 {
            if p.isEmpty { return p }
            var out: [ClipVert] = []
            for k in 0..<p.count {
                let a = p[k]
                let b = p[(k + 1) % p.count]
                let da = plane == 0 ? a.Z + a.W : a.W - a.Z
                let db = plane == 0 ? b.Z + b.W : b.W - b.Z
                if da >= 0 { out.append(a) }
                if (da >= 0) != (db >= 0) {
                    out.append(between(a, b, da / (da - db)))
                }
            }
            p = out
        }
        return p
    }

    func inside(_ c: ClipVert) -> bool { c.Z + c.W >= 0 && c.W - c.Z >= 0 }

    func triangle(_ i0: int, _ i1: int, _ i2: int, provoking: int, _ p: Program, _ fm: interp.Machine, _ st: State, _ t: Target) {
        let a = clip[i0]
        let b = clip[i1]
        let c = clip[i2]
        let prov = clip[provoking].Vary
        if inside(a) && inside(b) && inside(c) {
            fill(window(a, st), window(b, st), window(c, st), prov, p, fm, st, t, point: false)
            return
        }
        let poly = clipPolygon([a, b, c])
        if poly.count < 3 { return }
        let w0 = window(poly[0], st)
        for k in 1..<(poly.count - 1) {
            fill(w0, window(poly[k], st), window(poly[k + 1], st), prov, p, fm, st, t, point: false)
        }
    }

    func line(_ i0: int, _ i1: int, _ p: Program, _ fm: interp.Machine, _ st: State, _ t: Target) {
        var a = clip[i0]
        var b = clip[i1]
        // Clip the segment against near and far.
        for plane in 0..<2 {
            let da = plane == 0 ? a.Z + a.W : a.W - a.Z
            let db = plane == 0 ? b.Z + b.W : b.W - b.Z
            if da < 0 && db < 0 { return }
            if da < 0 { a = between(a, b, da / (da - db)) }
            else if db < 0 { b = between(b, a, db / (db - da)) }
        }
        let wa = window(a, st)
        let wb = window(b, st)
        // A parallelogram one pixel wide (LineWidth) across the minor axis.
        let half = float64(max(1, st.LineWidth)) * 0.5
        let dx = wb.X - wa.X
        let dy = wb.Y - wa.Y
        var ox: float64 = 0
        var oy: float64 = 0
        if abs(dx) >= abs(dy) { oy = half } else { ox = half }
        var a0 = wa; a0.X -= ox; a0.Y -= oy
        var a1 = wa; a1.X += ox; a1.Y += oy
        var b0 = wb; b0.X -= ox; b0.Y -= oy
        var b1 = wb; b1.X += ox; b1.Y += oy
        let prov = clip[i1].Vary
        fill(a0, b0, b1, prov, p, fm, st, t, point: false, cullable: false)
        fill(a0, b1, a1, prov, p, fm, st, t, point: false, cullable: false)
    }

    func point(_ i: int, _ p: Program, _ fm: interp.Machine, _ st: State, _ t: Target) {
        let c = clip[i]
        if !inside(c) { return }
        let w = window(c, st)
        let size = float64(min(max(pointSize[i], 1), 1024))
        let h = size * 0.5
        var bl = w; bl.X -= h; bl.Y -= h; bl.PS = 0; bl.PT = 1
        var br = w; br.X += h; br.Y -= h; br.PS = 1; br.PT = 1
        var tl = w; tl.X -= h; tl.Y += h; tl.PS = 0; tl.PT = 0
        var tr = w; tr.X += h; tr.Y += h; tr.PS = 1; tr.PT = 0
        fill(bl, br, tr, c.Vary, p, fm, st, t, point: true, cullable: false)
        fill(bl, tr, tl, c.Vary, p, fm, st, t, point: true, cullable: false)
    }

    // MARK: rasterizing

    /// Covers a triangle's pixels: 2×2 quads into fragment batches.
    func fill(_ va: WinVert, _ vb: WinVert, _ vc: WinVert, _ prov: int, _ p: Program, _ fm: interp.Machine,
              _ st: State, _ t: Target, point: bool, cullable: bool = true) {
        // Fixed point, 8 bits of subpixel, for exact edge tests.
        let limit: float64 = 1048576
        func fx(_ v: float64) -> int64 { int64((max(-limit, min(limit, v)) * 256).rounded()) }
        var A = va
        var B = vb
        var C = vc
        let ax = fx(A.X)
        let ay = fx(A.Y)
        var bxp = fx(B.X)
        var byp = fx(B.Y)
        var cx = fx(C.X)
        var cy = fx(C.Y)
        var area = (bxp - ax) * (cy - ay) - (cx - ax) * (byp - ay)
        if area == 0 { return }
        let ccw = area > 0
        let front = st.FrontCCW ? ccw : !ccw
        if cullable, let cull = st.Cull {
            if cull == .frontAndBack || (cull == .front && front) || (cull == .back && !front) { return }
        }
        if !ccw {
            // Make it counter-clockwise: swap b and c.
            let tmp = B; B = C; C = tmp
            let tx = bxp; bxp = cx; cx = tx
            let ty = byp; byp = cy; cy = ty
            area = -area
        }
        // The batch shades one facing at a time.
        if quads > 0 && batchFront != front { flush(p, fm, st, t, front: batchFront) }
        batchFront = front
        batchPoint = point
        triA = A
        triB = B
        triC = C
        triProvoking = prov
        // Depth slope, for polygon offset.
        var offset: float64 = 0
        if st.PolygonOffset && !point {
            let fa = float64(area) / 65536
            let dzdx = ((B.Z - A.Z) * (C.Y - A.Y) - (C.Z - A.Z) * (B.Y - A.Y)) / fa
            let dzdy = ((C.Z - A.Z) * (B.X - A.X) - (B.Z - A.Z) * (C.X - A.X)) / fa
            offset = float64(st.OffsetFactor) * max(abs(dzdx), abs(dzdy)) + float64(st.OffsetUnits) / 16777216
        }
        triOffset = offset
        // The bounding box, inside the target and scissor.
        var x0 = int(min(A.X, min(B.X, C.X)).rounded(.down))
        var x1 = int(max(A.X, max(B.X, C.X)).rounded(.up))
        var y0 = int(min(A.Y, min(B.Y, C.Y)).rounded(.down))
        var y1 = int(max(A.Y, max(B.Y, C.Y)).rounded(.up))
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, t.Width)
        y1 = min(y1, t.Height)
        if let s = st.Scissor {
            x0 = max(x0, s.X)
            y0 = max(y0, s.Y)
            x1 = min(x1, s.X + s.Width)
            y1 = min(y1, s.Y + s.Height)
        }
        if x0 >= x1 || y0 >= y1 { return }
        // Quads start on even coordinates.
        x0 &= ~1
        y0 &= ~1
        // Edge functions at pixel centers: E_ab is ≥ 0 inside for a ccw triangle.
        // The fill rule: a center exactly on an edge belongs to one side only.
        func owns(_ dx: int64, _ dy: int64) -> bool { dy > 0 || (dy == 0 && dx > 0) }
        let ownBC = owns(cx - bxp, cy - byp)
        let ownCA = owns(ax - cx, ay - cy)
        let ownAB = owns(bxp - ax, byp - ay)
        let fa = float64(area)
        var qy = y0
        while qy < y1 {
            var qx = x0
            while qx < x1 {
                var m: uint64 = 0
                // Written into the next batch slots; kept only if the quad is.
                let base = quads * 4
                for k in 0..<4 {
                    let px = qx + (k & 1)
                    let py = qy + (k >> 1)
                    let cxp = int64(px) * 256 + 128
                    let cyp = int64(py) * 256 + 128
                    let ebc = (cx - bxp) * (cyp - byp) - (cy - byp) * (cxp - bxp)
                    let eca = (ax - cx) * (cyp - cy) - (ay - cy) * (cxp - cx)
                    let eab = (bxp - ax) * (cyp - ay) - (byp - ay) * (cxp - ax)
                    bx[base + k] = px
                    by[base + k] = py
                    lambda[(base + k) * 3] = float64(ebc) / fa
                    lambda[(base + k) * 3 + 1] = float64(eca) / fa
                    lambda[(base + k) * 3 + 2] = float64(eab) / fa
                    if px < x1 && py < y1 && px >= 0 && py >= 0 &&
                        (ebc > 0 || (ebc == 0 && ownBC)) && (eca > 0 || (eca == 0 && ownCA)) && (eab > 0 || (eab == 0 && ownAB)) {
                        if let s = st.Scissor, px < s.X || py < s.Y || px >= s.X + s.Width || py >= s.Y + s.Height { continue }
                        m |= uint64(1) << uint64(k)
                    }
                }
                if m != 0 {
                    cover |= m << uint64(base)
                    quads += 1
                    if quads * 4 >= Lanes { flush(p, fm, st, t, front: front) }
                }
                qx += 2
            }
            qy += 2
        }
        // The batch refers to this triangle's vertices: shade it before the next.
        flush(p, fm, st, t, front: front)
    }

    var batchFront = true
    var batchPoint = false
    var triA = WinVert(X: 0, Y: 0, Z: 0, IW: 0, Vary: 0)
    var triB = WinVert(X: 0, Y: 0, Z: 0, IW: 0, Vary: 0)
    var triC = WinVert(X: 0, Y: 0, Z: 0, IW: 0, Vary: 0)
    var triProvoking = 0
    var triOffset: float64 = 0
}

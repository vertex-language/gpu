package raster

// Drawing on the GPU. Once EnableGPU finds a Metal device, every
// Renderer draws there: a Program's shaders are translated to Metal
// (shader/msl) and compiled once, pipelines and depth-stencil and sampler
// states are made once per state they're asked for, and draws into one
// target go in one render pass. What Metal can't take (a shader msl
// can't translate, a swizzled texture) is drawn on the CPU as before,
// after the textures it touches are read back.

import (
    "gpu"
    "shader"
    "shader/interp"
    "shader/msl"
)

/// The GPU renderers draw with; nil: the CPU.
var metal: Metal? = nil

/// Draws on the GPU from now on, when this machine has a Metal device
/// (VERTEX_GPU=cpu says not to). Returns whether it does.
public func EnableGPU() -> bool {
    if metal != nil { return true }
    let d = gpu.Default()
    if d.Name != "metal" { return false }
    metal = Metal(d)
    return true
}

/// Whether draws go to the GPU.
public var GPUEnabled: bool { metal != nil }

/// What the GPU has done: draws there, and draws that went to the CPU instead.
public struct GPUStats {
    public var Draws = 0
    public var Clears = 0
    public var Fallbacks = 0
    public var Uploads = 0
    public var Readbacks = 0
    /// Why the last program that couldn't go to the GPU couldn't.
    public var LastFailure = ""

    public init() {}
}

public var GPUCounters: GPUStats { metal?.stats ?? GPUStats() }

/// Ends the open render pass and starts the GPU on what was drawn.
public func FlushGPU() {
    metal?.endPass()
}

/// A program's Metal side.
final class GPUProgram {
    let libraries: [gpu.Library]
    let vertexFunction: gpu.ShaderFunction
    let fragmentFunction: gpu.ShaderFunction
    let vertexSamplers: [msl.SamplerSlot]
    let fragmentSamplers: [msl.SamplerSlot]
    /// The attribute locations the vertex shader reads, and whether each is
    /// float (0), int (1) or uint (2). A matrix reads one per column.
    let locations: [int]
    let kinds: [int]
    /// Uniform slots each stage reads (the rest are its temporaries).
    let vertexSlots: int
    let fragmentSlots: int
    var pipelines: [string: gpu.RenderPipeline] = [:]

    init(libraries: [gpu.Library], vertex: gpu.ShaderFunction, fragment: gpu.ShaderFunction,
         vertexSamplers: [msl.SamplerSlot], fragmentSamplers: [msl.SamplerSlot],
         locations: [int], kinds: [int], vertexSlots: int, fragmentSlots: int) {
        self.libraries = libraries
        vertexFunction = vertex
        fragmentFunction = fragment
        self.vertexSamplers = vertexSamplers
        self.fragmentSamplers = fragmentSamplers
        self.locations = locations
        self.kinds = kinds
        self.vertexSlots = vertexSlots
        self.fragmentSlots = fragmentSlots
    }
}

/// One vertex buffer slot of a draw: its bytes and how they're read.
struct VertexSlot {
    var Format: int
    var Stride: int
    var Step: int
    var Rate: int
}

final class Metal {
    let device: gpu.Device
    var stats = GPUStats()
    // The open pass, and the target it draws into.
    var pass: gpu.RenderPass? = nil
    var passColor: Texture? = nil
    var passLevel = 0
    var passFace = 0
    var passDepth: Texture? = nil
    var depthStates: [string: gpu.DepthStencilState] = [:]
    var samplers: [string: gpu.SamplerState] = [:]
    // What an unbound or incomplete texture reads: (0, 0, 0, 1).
    var blank2D: gpu.Texture? = nil
    var blankCube: gpu.Texture? = nil
    // Clearing part of a target: a rectangle drawn with these.
    var clearLibrary: gpu.Library? = nil
    var clearPipelines: [string: gpu.RenderPipeline] = [:]

    init(_ device: gpu.Device) {
        self.device = device
    }

    // MARK: passes

    func endPass() {
        guard let p = pass else { return }
        p.End()
        pass = nil
        passColor = nil
        passDepth = nil
        try? device.Commit(wait: false)
    }

    /// A pass into t: the open one when it draws there, else a new one.
    func passFor(_ t: Target, color: gpu.Texture?, depth: gpu.Texture?) -> gpu.RenderPass? {
        if let p = pass, passColor === t.Color, passLevel == t.ColorLevel, passFace == t.ColorFace, passDepth === t.Depth {
            return p
        }
        endPass()
        return begin(t, color: color, depth: depth, clearColor: nil, clearDepth: nil, clearStencil: nil)
    }

    func begin(_ t: Target, color: gpu.Texture?, depth: gpu.Texture?, clearColor: [float32]?, clearDepth: float32?, clearStencil: int?) -> gpu.RenderPass? {
        guard let p = gpu.RenderPass.Begin(device: device, color: color, level: t.ColorLevel, slice: t.ColorFace, clearColor: clearColor,
                                           depth: depth, clearDepth: clearDepth, clearStencil: clearStencil) else { return nil }
        pass = p
        passColor = t.Color
        passLevel = t.ColorLevel
        passFace = t.ColorFace
        passDepth = t.Depth
        return p
    }

    // MARK: textures

    /// t's device copy, made at its size and sent the host's levels when they're newer.
    func resident(_ t: Texture) -> gpu.Texture? {
        let w = t.Width
        let h = t.Height
        if w <= 0 || h <= 0 { return nil }
        if !t.Complete { return nil }
        // The levels: as many of the chain as are there at their sizes.
        var levels = 1
        let full = MipLevels(w, h)
        while levels < full {
            var ok = true
            for f in t.Faces where f.count <= levels || f[levels].Width != max(1, w >> levels) || f[levels].Height != max(1, h >> levels) {
                ok = false
            }
            if !ok { break }
            levels += 1
        }
        if let d = t.device, d.Width == w, d.Height == h, d.Levels == levels {
            if !t.hostNewer { return d }
        } else {
            endPass()
            let format: gpu.TextureFormat = t.Format == .rgba8 ? .rgba8 : (t.Format == .rgba32f ? .rgba32f : .depthStencil)
            t.device = gpu.Texture.Create(device: device, width: w, height: h, format: format, levels: levels, cube: t.Cube)
            t.hostNewer = true
        }
        guard let d = t.device else { return nil }
        if t.hostNewer && t.Format != .depthStencil {
            endPass()
            stats.Uploads += 1
            let bpp = t.Format == .rgba8 ? 4 : 16
            for (f, lvs) in t.Faces.enumerated() {
                for k in 0..<min(levels, lvs.count) {
                    let lv = lvs[k]
                    let p = t.Format == .rgba8 ? UnsafeRawPointer(lv.Bytes) : UnsafeRawPointer(lv.Floats)
                    d.Write(level: k, slice: f, x: 0, y: 0, width: lv.Width, height: lv.Height, bytes: p, bytesPerRow: lv.Width * bpp)
                }
            }
        }
        t.hostNewer = false
        return d
    }

    func blank(cube: bool) -> gpu.Texture? {
        if cube, let b = blankCube { return b }
        if !cube, let b = blank2D { return b }
        endPass()
        guard let b = gpu.Texture.Create(device: device, width: 1, height: 1, format: .rgba8, cube: cube) else { return nil }
        let px: [uint8] = [0, 0, 0, 255]
        px.withUnsafeBufferPointer { p in
            for f in 0..<(cube ? 6 : 1) { b.Write(slice: f, x: 0, y: 0, width: 1, height: 1, bytes: UnsafeRawPointer(p.baseAddress!), bytesPerRow: 4) }
        }
        if cube { blankCube = b } else { blank2D = b }
        return b
    }

    func sampler(_ s: Sampler) -> gpu.SamplerState? {
        let mip = s.MipFilter == nil ? 0 : (s.MipFilter! == .nearest ? 1 : 2)
        let compare = s.Compare == nil ? -1 : compareCode(s.Compare!)
        let maxLod = min(s.MaxLod, float32(s.MaxLevel - s.BaseLevel))
        let key = "\(s.MinFilter)\(s.MagFilter)\(mip)\(s.WrapS)\(s.WrapT)\(s.WrapR)\(compare)\(s.MinLod)\(maxLod)"
        if let st = samplers[key] { return st }
        let st = gpu.SamplerState.Create(device: device, min: s.MinFilter == .linear ? 1 : 0, mag: s.MagFilter == .linear ? 1 : 0, mip: mip,
                                         s: wrapCode(s.WrapS), t: wrapCode(s.WrapT), r: wrapCode(s.WrapR), compare: compare,
                                         minLod: max(0, s.MinLod), maxLod: max(0, maxLod))
        if let st = st { samplers[key] = st }
        return st
    }

    // MARK: programs

    func prepare(_ p: Program) -> GPUProgram? {
        if let g = p.gpuState { return g }
        if p.gpuFailed { return nil }
        do {
            let g = try translate(p)
            p.gpuState = g
            return g
        } catch {
            p.gpuFailed = true
            stats.LastFailure = "\(error)"
            return nil
        }
    }

    func translate(_ p: Program) throws -> GPUProgram {
        let vm = p.Vertex.Module
        var attributes: [int: int] = [:]
        var locations: [int] = []
        var kinds: [int] = []
        for (i, input) in p.Vertex.Inputs.enumerated() {
            guard let v = vm.Variables.first(where: { $0.Storage == .input && $0.Name == input.Name }) else {
                throw msl.TranslateError("no variable for input \(input.Name)")
            }
            let loc = p.InputLocations[i]
            attributes[v.Index] = loc
            let kind = input.Type.Kind == .int ? 1 : (input.Type.Kind == .uint ? 2 : 0)
            for c in 0..<(v.Type.IsMatrix ? v.Type.Columns : 1) {
                locations.append(loc + c)
                kinds.append(kind)
            }
        }
        if locations.count > 28 { throw msl.TranslateError("more than 28 attributes") }
        let vs = try msl.Translate(vm, offsets: p.Vertex.Offsets, attributes: attributes)
        let fs = try msl.Translate(p.Fragment.Module, offsets: p.Fragment.Offsets, attributes: [:])
        for slot in vs.Samplers + fs.Samplers {
            switch slot.Kind {
            case .texture2D, .external, .cube: break
            default: throw msl.TranslateError("sampler kind \(slot.Kind) isn't drawn on the GPU yet")
            }
        }
        let vl = try gpu.Library(device: device, source: vs.Text)
        let fl = try gpu.Library(device: device, source: fs.Text)
        return GPUProgram(libraries: [vl, fl], vertex: try vl.Function(vs.Entry), fragment: try fl.Function(fs.Entry),
                          vertexSamplers: vs.Samplers, fragmentSamplers: fs.Samplers, locations: locations, kinds: kinds,
                          vertexSlots: uniformSlots(p.Vertex), fragmentSlots: uniformSlots(p.Fragment))
    }

    func uniformSlots(_ e: interp.Executable) -> int {
        var n = 1
        for u in e.Uniforms { n = max(n, u.Offset + max(1, e.Module.Slots(u.Type))) }
        return min(n, e.SlotCount)
    }

    // MARK: drawing

    /// Draws d on the GPU; false when it can't, and the CPU should.
    func Draw(_ d: Draw, program: Program, state: State, target: Target, textures: [Binding?]) -> bool {
        guard let g = prepare(program) else { return false }
        let triangles = d.Primitive == .triangles || d.Primitive == .triangleStrip || d.Primitive == .triangleFan
        if triangles && state.Cull == .frontAndBack { return true }

        // Textures first: sending one ends the pass.
        var stageTextures: [[gpu.Texture?]] = [[], []]
        var stageSamplers: [[gpu.SamplerState?]] = [[], []]
        for stage in 0..<2 {
            let slots = stage == 0 ? g.vertexSamplers : g.fragmentSamplers
            let e = stage == 0 ? program.Vertex : program.Fragment
            let values = stage == 0 ? program.VertexUniforms : program.FragmentUniforms
            for slot in slots {
                let off = slot.Variable < e.Offsets.count ? e.Offsets[slot.Variable] : -1
                let unit = off >= 0 && off < values.count ? int(int32(bitPattern: values[off])) : -1
                let cube = slot.Kind == .cube
                var tex: gpu.Texture? = nil
                var smp: gpu.SamplerState? = nil
                if unit >= 0 && unit < textures.count, let b = textures[unit], let t = b.Texture, t.Cube == cube {
                    if !identity(b.Sampler.Swizzle) { return false }
                    if b.Sampler.MipFilter != nil && !t.MipmapComplete {
                        tex = nil   // incomplete: reads (0, 0, 0, 1)
                    } else {
                        tex = resident(t)
                    }
                    smp = sampler(b.Sampler)
                }
                if tex == nil {
                    tex = blank(cube: cube)
                    smp = sampler(Sampler())
                }
                stageTextures[stage].append(tex)
                stageSamplers[stage].append(smp)
            }
        }
        var ct: gpu.Texture? = nil
        var dt: gpu.Texture? = nil
        if !targetTextures(target, &ct, &dt) { return false }

        // Vertices: one buffer slot for each location the shader reads.
        var last = 0           // the highest vertex index read
        var order: [uint32] = []
        var indexed = d.Indices != nil
        let restart = d.RestartIndex
        if let ib = d.Indices {
            let n = d.Count
            let size = d.IndexSize
            order.reserveCapacity(n)
            let bytes = ib.Bytes
            for i in 0..<n {
                let at = d.IndexOffset + i * size
                if at + size > bytes.count { break }
                var v: uint32 = uint32(bytes[at])
                if size >= 2 { v |= uint32(bytes[at + 1]) << 8 }
                if size == 4 { v |= uint32(bytes[at + 2]) << 16 | uint32(bytes[at + 3]) << 24 }
                if let r = restart, v == r {
                    order.append(0xFFFF_FFFF)
                    continue
                }
                last = max(last, int(v))
                order.append(v)
            }
        } else {
            last = d.First + d.Count - 1
        }
        // Loops and fans Metal doesn't draw: lists of indices instead.
        var primitive = 3
        switch d.Primitive {
        case .points: primitive = 0
        case .lines: primitive = 1
        case .lineStrip: primitive = 2
        case .triangles: primitive = 3
        case .triangleStrip: primitive = 4
        case .lineLoop, .triangleFan:
            if !indexed {
                for i in 0..<d.Count { order.append(uint32(d.First + i)) }
                indexed = true
            }
            order = d.Primitive == .lineLoop ? loopToStrip(order) : fanToList(order)
            primitive = d.Primitive == .lineLoop ? 2 : 3
        }
        if indexed && order.isEmpty { return true }

        guard let pass = passFor(target, color: ct, depth: dt) else { return false }
        var slots: [VertexSlot] = []
        for (k, loc) in g.locations.enumerated() {
            let a = loc < d.Attributes.count ? d.Attributes[loc] : Attribute()
            slots.append(bindAttribute(pass, a, kind: g.kinds[k], slot: k, last: last, instances: d.Instances))
        }

        // The pipeline for these formats and this blending.
        var key = "\(target.Color == nil ? -1 : (target.Color!.Format == .rgba8 ? 0 : 1)),\(target.Depth != nil)"
        if state.Blend {
            key += ",b\(state.BlendRGB)\(state.BlendAlpha)\(state.SrcRGB)\(state.DstRGB)\(state.SrcAlpha)\(state.DstAlpha)"
        }
        let mask = (state.WriteRed ? 8 : 0) | (state.WriteGreen ? 4 : 0) | (state.WriteBlue ? 2 : 0) | (state.WriteAlpha ? 1 : 0)
        key += ",m\(mask)"
        for s in slots { key += ";\(s.Format),\(s.Stride),\(s.Step),\(s.Rate)" }
        var pipeline = g.pipelines[key]
        if pipeline == nil {
            var pd = gpu.PipelineDescriptor()
            pd.Color = target.Color == nil ? nil : (target.Color!.Format == .rgba8 ? .rgba8 : .rgba32f)
            pd.DepthStencil = target.Depth != nil
            pd.Blending = state.Blend
            pd.RGBOperation = blendOp(state.BlendRGB)
            pd.AlphaOperation = blendOp(state.BlendAlpha)
            pd.SourceRGB = blendFactor(state.SrcRGB)
            pd.DestinationRGB = blendFactor(state.DstRGB)
            pd.SourceAlpha = blendFactor(state.SrcAlpha)
            pd.DestinationAlpha = blendFactor(state.DstAlpha)
            pd.WriteMask = mask
            for (k, s) in slots.enumerated() {
                pd.Attributes.append(gpu.VertexAttribute(index: g.locations[k], format: s.Format, offset: 0, buffer: k))
                pd.Layouts.append(gpu.VertexLayout(buffer: k, stride: s.Stride, step: s.Step, rate: s.Rate))
            }
            do {
                pipeline = try gpu.RenderPipeline(device: device, vertex: g.vertexFunction, fragment: g.fragmentFunction, pd)
            } catch {
                stats.LastFailure = "\(error)"
                return false
            }
            g.pipelines[key] = pipeline!
        }
        pass.SetPipeline(pipeline!)

        // Fixed state.
        let w = target.Width
        let h = target.Height
        let flip: float32 = target.FlipY ? 1 : -1
        let vp = state.Viewport
        let vy = target.FlipY ? h - (vp.Y + vp.Height) : vp.Y
        pass.SetViewport(x: float32(vp.X), y: float32(vy), width: float32(vp.Width), height: float32(vp.Height),
                         near: state.DepthNear, far: state.DepthFar)
        var sx0 = 0
        var sy0 = 0
        var sx1 = w
        var sy1 = h
        if let s = state.Scissor {
            sx0 = max(0, s.X)
            sx1 = min(w, s.X + s.Width)
            let top = target.FlipY ? h - (s.Y + s.Height) : s.Y
            sy0 = max(0, top)
            sy1 = min(h, top + s.Height)
        }
        if sx0 >= sx1 || sy0 >= sy1 { return true }
        pass.SetScissor(x: sx0, y: sy0, width: sx1 - sx0, height: sy1 - sy0)
        let cull = !triangles || state.Cull == nil ? 0 : (state.Cull! == .front ? 1 : 2)
        // A target stored bottom row first is drawn upside down: its windings turn over.
        pass.SetRaster(cull: cull, counterClockwise: state.FrontCCW == target.FlipY,
                       bias: state.PolygonOffset ? state.OffsetUnits : 0, slope: state.PolygonOffset ? state.OffsetFactor : 0)
        if let ds = depthStencil(state, hasDepth: target.Depth != nil) {
            pass.SetDepthStencil(ds, front: state.StencilFront.Ref & 0xFF, back: state.StencilBack.Ref & 0xFF)
        }
        let bc = state.BlendColor
        pass.SetBlendColor(bc.R, bc.G, bc.B, bc.A)

        // Uniforms, the per-draw constants, textures.
        let k: [float32] = [flip, float32(h), -flip, 0]
        k.withUnsafeBufferPointer { p in
            pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress!), count: 16, index: msl.ConstantsBuffer)
            pass.SetBytes(fragment: true, UnsafeRawPointer(p.baseAddress!), count: 16, index: msl.ConstantsBuffer)
        }
        program.VertexUniforms.withUnsafeBufferPointer { p in
            pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress!), count: min(g.vertexSlots, p.count) * 4, index: msl.UniformBuffer)
        }
        program.FragmentUniforms.withUnsafeBufferPointer { p in
            pass.SetBytes(fragment: true, UnsafeRawPointer(p.baseAddress!), count: min(g.fragmentSlots, p.count) * 4, index: msl.UniformBuffer)
        }
        for stage in 0..<2 {
            let slots = stage == 0 ? g.vertexSamplers : g.fragmentSamplers
            for (i, slot) in slots.enumerated() {
                pass.SetTexture(fragment: stage == 1, stageTextures[stage][i], index: slot.Slot)
                if let s = stageSamplers[stage][i] { pass.SetSampler(fragment: stage == 1, s, index: slot.Slot) }
            }
        }

        if indexed {
            order.withUnsafeBufferPointer { p in
                pass.DrawIndexed(primitive: primitive, count: p.count, indexSize: 4, indices: UnsafeRawPointer(p.baseAddress!),
                                 bytes: p.count * 4, instances: max(1, d.Instances))
            }
        } else {
            pass.Draw(primitive: primitive, start: d.First, count: d.Count, instances: max(1, d.Instances))
        }
        drewInto(target)
        stats.Draws += 1
        return true
    }

    /// The target's device textures; false when one it has can't be made.
    func targetTextures(_ t: Target, _ color: inout gpu.Texture?, _ depth: inout gpu.Texture?) -> bool {
        if let c = t.Color {
            color = resident(c)
            if color == nil { return false }
        }
        if let d = t.Depth {
            depth = resident(d)
            if depth == nil { return false }
        }
        return true
    }

    func drewInto(_ t: Target) {
        if let c = t.Color {
            c.deviceNewer = true
            c.hostNewer = false
        }
        if let d = t.Depth {
            d.deviceNewer = true
            d.hostNewer = false
        }
    }

    /// Binds attribute a's bytes to buffer slot `slot`, and says how they're read.
    func bindAttribute(_ pass: gpu.RenderPass, _ a: Attribute, kind: int, slot: int, last: int, instances: int) -> VertexSlot {
        if !a.Enabled || a.Buffer == nil {
            a.Constant.withUnsafeBufferPointer { p in
                pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress!), count: 16, index: slot)
            }
            return VertexSlot(Format: wideFormat(kind), Stride: 16, Step: 0, Rate: 0)
        }
        let bytes = a.Buffer!.Bytes
        let size = componentSize(a.Type)
        let stride = a.Stride > 0 ? a.Stride : size * a.Count
        let step = a.Divisor > 0 ? 2 : 1
        let lastElement = a.Divisor > 0 ? max(0, instances - 1) / a.Divisor : last
        let need = lastElement * stride + size * a.Count
        let format = directFormat(a, kind: kind)
        if format >= 0 && a.Offset % 4 == 0 && stride % 4 == 0 && a.Offset + need <= bytes.count {
            bytes.withUnsafeBufferPointer { p in
                pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress! + a.Offset), count: need, index: slot)
            }
            return VertexSlot(Format: format, Stride: stride, Step: step, Rate: a.Divisor > 0 ? a.Divisor : 1)
        }
        // Anything else is read as the CPU reads it, into four 32-bit words a vertex.
        var words: [uint32] = []
        words.reserveCapacity((lastElement + 1) * 4)
        for i in 0...lastElement { words += readAttribute(a, bytes, i, integerInput: kind != 0) }
        words.withUnsafeBufferPointer { p in
            pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress!), count: p.count * 4, index: slot)
        }
        return VertexSlot(Format: wideFormat(kind), Stride: 16, Step: step, Rate: a.Divisor > 0 ? a.Divisor : 1)
    }

    func depthStencil(_ s: State, hasDepth: bool) -> gpu.DepthStencilState? {
        let compare = hasDepth && s.DepthTest != nil ? compareCode(s.DepthTest!) : 7
        let write = hasDepth && s.DepthTest != nil && s.DepthWrite
        let stencil = hasDepth && s.StencilTest
        var key = "\(compare),\(write),\(stencil)"
        var front = gpu.StencilDescriptor()
        var back = gpu.StencilDescriptor()
        if stencil {
            front = stencilDescriptor(s.StencilFront)
            back = stencilDescriptor(s.StencilBack)
            for f in [front, back] { key += ";\(f.Compare),\(f.Fail),\(f.DepthFail),\(f.Pass),\(f.ReadMask),\(f.WriteMask)" }
        }
        if let ds = depthStates[key] { return ds }
        let ds = gpu.DepthStencilState.Create(device: device, compare: compare, write: write, stencil: stencil, front: front, back: back)
        if let ds = ds { depthStates[key] = ds }
        return ds
    }

    // MARK: clearing

    /// Clears on the GPU: the whole target as a pass starts, or a rectangle drawn.
    func Clear(_ t: Target, color: Color?, depth: float32?, stencil: int?, state: State) -> bool {
        let w = t.Width
        let h = t.Height
        if w <= 0 || h <= 0 { return true }
        var ct: gpu.Texture? = nil
        var dt: gpu.Texture? = nil
        if !targetTextures(t, &ct, &dt) { return false }
        let wantColor = color != nil && t.Color != nil
        let wantDepth = (depth != nil || stencil != nil) && t.Depth != nil
        if !wantColor && !wantDepth { return true }
        var x0 = 0
        var y0 = 0
        var x1 = w
        var y1 = h
        if let s = state.Scissor {
            x0 = max(0, s.X)
            x1 = min(w, s.X + s.Width)
            let top = t.FlipY ? h - (s.Y + s.Height) : s.Y
            y0 = max(0, top)
            y1 = min(h, top + s.Height)
        }
        if x0 >= x1 || y0 >= y1 { return true }
        let whole = x0 == 0 && y0 == 0 && x1 == w && y1 == h
        let allColor = state.WriteRed && state.WriteGreen && state.WriteBlue && state.WriteAlpha
        let allStencil = state.StencilFront.WriteMask & 0xFF == 0xFF
        // Cleared whole: a new pass that starts so (what it doesn't clear, it keeps).
        if whole && (!wantColor || allColor) && (!wantDepth || stencil == nil || allStencil) {
            endPass()
            let c = color ?? Color()
            guard begin(t, color: ct, depth: dt, clearColor: wantColor ? [c.R, c.G, c.B, c.A] : nil,
                        clearDepth: wantDepth ? depth : nil, clearStencil: wantDepth ? stencil : nil) != nil else { return false }
            drewInto(t)
            stats.Clears += 1
            return true
        }
        // Else a rectangle, drawn with the masks.
        guard let pass = passFor(t, color: ct, depth: dt) else { return false }
        let mask = wantColor ? ((state.WriteRed ? 8 : 0) | (state.WriteGreen ? 4 : 0) | (state.WriteBlue ? 2 : 0) | (state.WriteAlpha ? 1 : 0)) : 0
        guard let pipeline = clearPipeline(t, mask: mask) else { return false }
        pass.SetPipeline(pipeline)
        pass.SetViewport(x: 0, y: 0, width: float32(w), height: float32(h), near: 0, far: 1)
        pass.SetScissor(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
        pass.SetRaster(cull: 0, counterClockwise: true, bias: 0, slope: 0)
        var front = gpu.StencilDescriptor()
        let clearsStencil = wantDepth && stencil != nil
        if clearsStencil {
            front.Compare = 7
            front.Pass = 2   // replace
            front.WriteMask = int(state.StencilFront.WriteMask & 0xFF)
        }
        let key = "clear,\(wantDepth && depth != nil),\(clearsStencil),\(front.WriteMask)"
        var ds = depthStates[key]
        if ds == nil {
            ds = gpu.DepthStencilState.Create(device: device, compare: 7, write: wantDepth && depth != nil, stencil: clearsStencil, front: front, back: front)
            if ds == nil { return false }
            depthStates[key] = ds!
        }
        pass.SetDepthStencil(ds!, front: (stencil ?? 0) & 0xFF, back: (stencil ?? 0) & 0xFF)
        let c = color ?? Color()
        let values: [float32] = [c.R, c.G, c.B, c.A, min(max(depth ?? 0, 0), 1), 0, 0, 0]
        values.withUnsafeBufferPointer { p in
            pass.SetBytes(fragment: false, UnsafeRawPointer(p.baseAddress!), count: 32, index: 0)
            pass.SetBytes(fragment: true, UnsafeRawPointer(p.baseAddress!), count: 32, index: 0)
        }
        pass.Draw(primitive: 4, start: 0, count: 4, instances: 1)
        drewInto(t)
        stats.Clears += 1
        return true
    }

    func clearPipeline(_ t: Target, mask: int) -> gpu.RenderPipeline? {
        let format = t.Color == nil ? -1 : (t.Color!.Format == .rgba8 ? 0 : 1)
        let key = "\(format),\(t.Depth != nil),\(mask)"
        if let p = clearPipelines[key] { return p }
        do {
            if clearLibrary == nil { clearLibrary = try gpu.Library(device: device, source: clearSource) }
            var pd = gpu.PipelineDescriptor()
            pd.Color = format < 0 ? nil : (format == 0 ? .rgba8 : .rgba32f)
            pd.DepthStencil = t.Depth != nil
            pd.WriteMask = mask
            let p = try gpu.RenderPipeline(device: device, vertex: try clearLibrary!.Function("clearVertex"),
                                           fragment: try clearLibrary!.Function("clearFragment"), pd)
            clearPipelines[key] = p
            return p
        } catch {
            stats.LastFailure = "\(error)"
            return nil
        }
    }

    /// Before the CPU draws: what it reads and writes, brought back from the GPU.
    func toHost(_ t: Target, _ textures: [Binding?]) {
        stats.Fallbacks += 1
        if let c = t.Color {
            c.Synchronize()
            c.hostNewer = true
        }
        if let d = t.Depth {
            d.Synchronize()
            d.hostNewer = true
        }
        for b in textures {
            if let tex = b?.Texture { tex.Synchronize() }
        }
    }
}

/// A rectangle covering the viewport: the color and depth in buffer 0.
let clearSource = """
#include <metal_stdlib>
using namespace metal;
struct ClearOut { float4 position [[position]]; };
vertex ClearOut clearVertex(uint vid [[vertex_id]], constant float4* v [[buffer(0)]]) {
    ClearOut o;
    float2 xy = float2((vid & 1) != 0 ? 1.0 : -1.0, (vid & 2) != 0 ? 1.0 : -1.0);
    o.position = float4(xy, v[1].x, 1.0);
    return o;
}
fragment float4 clearFragment(ClearOut in [[stage_in]], constant float4* v [[buffer(0)]]) {
    return v[0];
}
"""

func identity(_ s: [Swizzle]) -> bool {
    return s.count == 4 && s[0] == .red && s[1] == .green && s[2] == .blue && s[3] == .alpha
}

func loopToStrip(_ order: [uint32]) -> [uint32] {
    if order.isEmpty { return order }
    var out = order
    out.append(order[0])
    return out
}

func fanToList(_ order: [uint32]) -> [uint32] {
    var out: [uint32] = []
    if order.count < 3 { return out }
    out.reserveCapacity((order.count - 2) * 3)
    for i in 1..<(order.count - 1) {
        out.append(order[0])
        out.append(order[i])
        out.append(order[i + 1])
    }
    return out
}

func componentSize(_ t: ComponentType) -> int {
    switch t {
    case .float32, .int32, .uint32, .fixed: return 4
    case .float16, .int16, .uint16: return 2
    case .int8, .uint8: return 1
    }
}

/// Four 32-bit components: float4, int4 or uint4.
func wideFormat(_ kind: int) -> int {
    return kind == 1 ? 35 : (kind == 2 ? 39 : 31)
}

/// The Metal vertex format that reads a's bytes as they are; -1: none does.
func directFormat(_ a: Attribute, kind: int) -> int {
    let n = a.Count
    if n < 1 || n > 4 { return -1 }
    if kind != 0 {
        if !a.Integer { return -1 }
        switch a.Type {
        case .int32: return kind == 1 ? [32, 33, 34, 35][n - 1] : -1
        case .uint32: return kind == 2 ? [36, 37, 38, 39][n - 1] : -1
        default: return -1
        }
    }
    if a.Integer { return -1 }
    switch a.Type {
    case .float32: return [28, 29, 30, 31][n - 1]
    case .float16: return [53, 25, 26, 27][n - 1]
    case .uint8: return a.Normalized ? [47, 7, 8, 9][n - 1] : -1
    case .int8: return a.Normalized ? [48, 10, 11, 12][n - 1] : -1
    case .uint16: return a.Normalized ? [51, 19, 20, 21][n - 1] : -1
    case .int16: return a.Normalized ? [52, 22, 23, 24][n - 1] : -1
    default: return -1
    }
}

func compareCode(_ f: CompareFunc) -> int {
    switch f {
    case .never: return 0
    case .less: return 1
    case .equal: return 2
    case .lessEqual: return 3
    case .greater: return 4
    case .notEqual: return 5
    case .greaterEqual: return 6
    case .always: return 7
    }
}

func wrapCode(_ w: Wrap) -> int {
    switch w {
    case .clampToEdge: return 0
    case .repeating: return 2
    case .mirroredRepeat: return 3
    }
}

func blendOp(_ e: BlendEquation) -> int {
    switch e {
    case .add: return 0
    case .subtract: return 1
    case .reverseSubtract: return 2
    case .min: return 3
    case .max: return 4
    }
}

func blendFactor(_ f: BlendFactor) -> int {
    switch f {
    case .zero: return 0
    case .one: return 1
    case .srcColor: return 2
    case .oneMinusSrcColor: return 3
    case .srcAlpha: return 4
    case .oneMinusSrcAlpha: return 5
    case .dstColor: return 6
    case .oneMinusDstColor: return 7
    case .dstAlpha: return 8
    case .oneMinusDstAlpha: return 9
    case .srcAlphaSaturate: return 10
    case .constantColor: return 11
    case .oneMinusConstantColor: return 12
    case .constantAlpha: return 13
    case .oneMinusConstantAlpha: return 14
    }
}

func stencilOp(_ o: StencilOp) -> int {
    switch o {
    case .keep: return 0
    case .zero: return 1
    case .replace: return 2
    case .increment: return 3
    case .decrement: return 4
    case .invert: return 5
    case .incrementWrap: return 6
    case .decrementWrap: return 7
    }
}

func stencilDescriptor(_ f: StencilFace) -> gpu.StencilDescriptor {
    var s = gpu.StencilDescriptor()
    s.Compare = compareCode(f.Func)
    s.Fail = stencilOp(f.Fail)
    s.DepthFail = stencilOp(f.DepthFail)
    s.Pass = stencilOp(f.Pass)
    s.ReadMask = int(f.ReadMask & 0xFF)
    s.WriteMask = int(f.WriteMask & 0xFF)
    return s
}

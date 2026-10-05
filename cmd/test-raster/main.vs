// gpu/raster on the CPU: coverage by the fill rule, interpolation,
// textures, depth, stencil and blending, against values worked out by hand.
package main

import (
    "gpu/raster"
    "shader"
    "shader/glsl"
    "shader/interp"
)

var failures = 0

var prefix = ""

func check(_ ok: bool, _ what: string) {
    let what = prefix + what
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

func program(_ vs: string, _ fs: string) -> raster.Program? {
    let v = glsl.Compile(vs, stage: .vertex)
    let f = glsl.Compile(fs, stage: .fragment)
    guard let vm = v.Module, let fm = f.Module else {
        print("      \(v.Log)\(f.Log)")
        return nil
    }
    do {
        return try raster.Program(vertex: try interp.Compile(vm), fragment: try interp.Compile(fm))
    } catch {
        print("      \(error)")
        return nil
    }
}

func floats(_ xs: [float32]) -> raster.Buffer {
    var b: [uint8] = []
    for x in xs {
        let u = x.bitPattern
        b += [uint8(u & 0xff), uint8((u >> 8) & 0xff), uint8((u >> 16) & 0xff), uint8(u >> 24)]
    }
    return raster.Buffer(b)
}

func positions(_ xs: [float32]) -> raster.Attribute {
    var a = raster.Attribute()
    a.Enabled = true
    a.Buffer = floats(xs)
    a.Count = 2
    return a
}

func setUniform(_ p: raster.Program, _ name: string, _ v: [float32]) {
    if let u = p.Vertex.FindUniform(name) { for (k, x) in v.enumerated() { p.VertexUniforms[u.Offset + k] = x.bitPattern } }
    if let u = p.Fragment.FindUniform(name) { for (k, x) in v.enumerated() { p.FragmentUniforms[u.Offset + k] = x.bitPattern } }
}

func pixel(_ t: raster.Texture, _ x: int, _ y: int) -> [uint8] {
    let l = t.Read(level: 0)!
    let i = (y * l.Width + x) * 4
    return [l.Bytes[i], l.Bytes[i + 1], l.Bytes[i + 2], l.Bytes[i + 3]]
}

func main() -> int32 {
    suite(gpu: false)
    if raster.EnableGPU() {
        prefix = "gpu: "
        suite(gpu: true)
        let c = raster.GPUCounters
        check(c.Fallbacks == 0, "every draw went to the GPU (\(c.Draws) draws, \(c.Clears) clears, \(c.Fallbacks) on the CPU) \(c.LastFailure)")
    } else {
        print("skip  no Metal device: the GPU path isn't tested")
    }
    print(failures == 0 ? "all passed" : "\(failures) failed")
    return failures == 0 ? 0 : 1
}

func suite(gpu: bool) {
    let vs = "attribute vec2 pos; varying vec2 uv; void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }"
    let flat = "precision mediump float; uniform vec4 color; void main() { gl_FragColor = color; }"
    guard let p = program(vs, flat) else { check(false, "the flat program compiles"); return }
    let r = raster.Renderer()
    let tex = raster.Texture(width: 16, height: 16, format: .rgba8)
    let target = raster.Target(color: tex, depth: nil)
    var st = raster.State()
    st.Viewport = raster.Rect(x: 0, y: 0, width: 16, height: 16)
    r.Clear(target, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: st)

    // A full-screen quad as two triangles, blended additively at 1/4:
    // a pixel drawn twice would read 128, one missed 0.
    st.Blend = true
    st.SrcRGB = .one
    st.DstRGB = .one
    st.SrcAlpha = .one
    st.DstAlpha = .one
    setUniform(p, "color", [0.25, 0.25, 0.25, 0.25])
    var d = raster.Draw()
    d.Primitive = .triangleFan
    d.Count = 4
    d.Attributes = [positions([-1, -1, 1, -1, 1, 1, -1, 1])]
    r.Draw(d, program: p, state: st, target: target, textures: [])
    var once = true
    for y in 0..<16 { for x in 0..<16 where pixel(tex, x, y)[0] != 64 { once = false } }
    check(once, "a fan's two triangles cover every pixel exactly once (\(pixel(tex, 0, 0)), \(pixel(tex, 7, 7)))")
    // An odd diagonal through pixel centers: still each pixel once.
    r.Clear(target, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: raster.State())
    d.Primitive = .triangles
    d.Count = 6
    d.Attributes = [positions([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1])]
    r.Draw(d, program: p, state: st, target: target, textures: [])
    once = true
    for y in 0..<16 { for x in 0..<16 where pixel(tex, x, y)[0] != 64 { once = false } }
    check(once, "two triangles sharing a diagonal through pixel centers: each pixel once")

    // Half the screen: the left 8 columns.
    st.Blend = false
    r.Clear(target, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: st)
    setUniform(p, "color", [1, 0, 0, 1])
    d.Primitive = .triangleStrip
    d.Count = 4
    d.Attributes = [positions([-1, -1, 0, -1, -1, 1, 0, 1])]
    r.Draw(d, program: p, state: st, target: target, textures: [])
    check(pixel(tex, 7, 3) == [255, 0, 0, 255] && pixel(tex, 8, 3) == [0, 0, 0, 0], "a strip covers the left half and stops at x = 8")

    // Interpolation and a texture: uv mapped onto a 2×2 checker, nearest.
    let texfs = "precision mediump float; varying vec2 uv; uniform sampler2D s; void main() { gl_FragColor = texture2D(s, uv); }"
    guard let tp = program(vs, texfs) else { check(false, "the texture program compiles"); return }
    let checker = raster.Texture(width: 2, height: 2, format: .rgba8)
    checker.Write(level: 0)!.SetBytes([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255])
    var smp = raster.Sampler()
    smp.MagFilter = .nearest
    d.Primitive = .triangleFan
    d.Attributes = [positions([-1, -1, 1, -1, 1, 1, -1, 1])]
    r.Draw(d, program: tp, state: st, target: target, textures: [raster.Binding(texture: checker, sampler: smp)])
    check(pixel(tex, 2, 2) == [255, 0, 0, 255] && pixel(tex, 13, 2) == [0, 255, 0, 255] &&
          pixel(tex, 2, 13) == [0, 0, 255, 255] && pixel(tex, 13, 13) == [255, 255, 255, 255],
          "uv interpolates across the quad and samples the right texels")
    // Bilinear at pixel (8, 8): uv = 8.5/16, so 0.5625 of the way from
    // texel 0 to texel 1 each way. Red: 255·(0.4375² + 0.5625²) = 129.5;
    // green and blue: 255·0.5625 = 143.4.
    smp.MagFilter = .linear
    smp.WrapS = .clampToEdge
    smp.WrapT = .clampToEdge
    r.Draw(d, program: tp, state: st, target: target, textures: [raster.Binding(texture: checker, sampler: smp)])
    let mid = pixel(tex, 8, 8)
    check(abs(int(mid[0]) - 129) <= 1 && abs(int(mid[1]) - 143) <= 1 && abs(int(mid[2]) - 143) <= 1,
          "linear filtering weighs the four texels by distance (\(mid))")

    // Depth: a nearer triangle hides a farther one, whatever the order.
    let zvs = "attribute vec2 pos; uniform float z; void main() { gl_Position = vec4(pos, z, 1.0); }"
    guard let zp = program(zvs, flat) else { check(false, "the depth program compiles"); return }
    let depth = raster.Texture(width: 16, height: 16, format: .depthStencil)
    let dt = raster.Target(color: tex, depth: depth)
    st.DepthTest = .less
    r.Clear(dt, color: raster.Color(0, 0, 0, 1), depth: 1, stencil: 0, state: st)
    setUniform(zp, "z", [-0.5])
    setUniform(zp, "color", [0, 0, 1, 1])
    r.Draw(d, program: zp, state: st, target: dt, textures: [])
    setUniform(zp, "z", [0.5])
    setUniform(zp, "color", [1, 1, 0, 1])
    r.Draw(d, program: zp, state: st, target: dt, textures: [])
    // (The GPU's depth stays in its memory: only the CPU's is read.)
    check(pixel(tex, 5, 5) == [0, 0, 255, 255] && (gpu || abs(depth.At(level: 0)!.Floats[5 * 16 + 5] - 0.25) < 1e-6),
          "the depth test keeps the nearer surface, depth 0.25")

    // Stencil: write 1 where a half-quad draws, then draw only where it's 1.
    st.DepthTest = nil
    st.StencilTest = true
    st.StencilFront.Pass = .replace
    st.StencilFront.Ref = 1
    st.StencilBack = st.StencilFront
    st.WriteRed = false; st.WriteGreen = false; st.WriteBlue = false; st.WriteAlpha = false
    r.Clear(dt, color: raster.Color(0, 0, 0, 1), depth: 1, stencil: 0, state: raster.State())
    var half = d
    half.Primitive = .triangleStrip
    half.Attributes = [positions([-1, -1, 0, -1, -1, 1, 0, 1])]
    r.Draw(half, program: zp, state: st, target: dt, textures: [])
    st.WriteRed = true; st.WriteGreen = true; st.WriteBlue = true; st.WriteAlpha = true
    st.StencilFront.Func = .equal
    st.StencilFront.Pass = .keep
    st.StencilBack = st.StencilFront
    setUniform(zp, "color", [0, 1, 0, 1])
    r.Draw(d, program: zp, state: st, target: dt, textures: [])
    check(pixel(tex, 3, 3) == [0, 255, 0, 255] && pixel(tex, 12, 3) == [0, 0, 0, 255], "the stencil test limits drawing to where stencil is 1")

    // A window surface stores its top row first.
    let win = raster.Texture(width: 4, height: 4, format: .rgba8)
    let wt = raster.Target(color: win, depth: nil, flipY: true)
    var ws = raster.State()
    ws.Viewport = raster.Rect(x: 0, y: 0, width: 4, height: 4)
    r.Clear(wt, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: ws)
    setUniform(p, "color", [1, 1, 1, 1])
    var bottom = raster.Draw()
    bottom.Primitive = .triangleStrip
    bottom.Count = 4
    bottom.Attributes = [positions([-1, -1, 1, -1, -1, -0.5, 1, -0.5])]   // GL's bottom row (y = 0)
    r.Draw(bottom, program: p, state: ws, target: wt, textures: [])
    check(pixel(win, 0, 3) == [255, 255, 255, 255] && pixel(win, 0, 0) == [0, 0, 0, 0], "FlipY puts GL's bottom row last in memory")

    // Culling: a counter-clockwise triangle faces front, on either kind of target.
    for t in [target, wt] {
        let img = t.Color!
        var cs = raster.State()
        cs.Viewport = raster.Rect(x: 0, y: 0, width: img.Width, height: img.Height)
        cs.Cull = .back
        r.Clear(t, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: cs)
        var tri = raster.Draw()
        tri.Count = 3
        tri.Attributes = [positions([-1, -1, 3, -1, -1, 3])]   // counter-clockwise, covering everything
        r.Draw(tri, program: p, state: cs, target: t, textures: [])
        let front = pixel(img, 1, 1)
        r.Clear(t, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: cs)
        tri.Attributes = [positions([-1, -1, -1, 3, 3, -1])]   // clockwise
        r.Draw(tri, program: p, state: cs, target: t, textures: [])
        let back = pixel(img, 1, 1)
        check(front == [255, 255, 255, 255] && back == [0, 0, 0, 0], "culling back faces keeps counter-clockwise ones (flipY \(t.FlipY): \(front) \(back))")
    }

    // Scissored clears and draws touch only the scissor's pixels (y up, as GL's).
    for t in [target, wt] {
        let img = t.Color!
        var ss = raster.State()
        ss.Viewport = raster.Rect(x: 0, y: 0, width: img.Width, height: img.Height)
        r.Clear(t, color: raster.Color(0, 0, 0, 0), depth: nil, stencil: nil, state: ss)
        ss.Scissor = raster.Rect(x: 0, y: 0, width: 2, height: 1)   // GL's bottom-left pixels
        r.Clear(t, color: raster.Color(1, 0, 0, 1), depth: nil, stencil: nil, state: ss)
        let bottomRow = t.FlipY ? img.Height - 1 : 0
        let topRow = img.Height - 1 - bottomRow
        check(pixel(img, 1, bottomRow) == [255, 0, 0, 255] && pixel(img, 2, bottomRow) == [0, 0, 0, 0] && pixel(img, 0, topRow) == [0, 0, 0, 0],
              "a scissored clear touches only the scissor (flipY \(t.FlipY))")
        ss.Scissor = raster.Rect(x: img.Width - 1, y: img.Height - 1, width: 1, height: 1)   // GL's top-right pixel
        setUniform(p, "color", [0, 1, 0, 1])
        r.Draw(d, program: p, state: ss, target: t, textures: [])
        check(pixel(img, img.Width - 1, topRow) == [0, 255, 0, 255] && pixel(img, img.Width - 2, topRow) == [0, 0, 0, 0],
              "a scissored draw touches only the scissor (flipY \(t.FlipY))")
    }

    // gl_FragCoord counts from GL's bottom-left on either kind of target.
    let fcfs = "precision mediump float; void main() { gl_FragColor = vec4(gl_FragCoord.x < 1.0 ? 1.0 : 0.0, gl_FragCoord.y < 1.0 ? 1.0 : 0.0, 0.0, 1.0); }"
    guard let fc = program(vs, fcfs) else { check(false, "the gl_FragCoord program compiles"); return }
    for t in [target, wt] {
        let img = t.Color!
        var fs = raster.State()
        fs.Viewport = raster.Rect(x: 0, y: 0, width: img.Width, height: img.Height)
        r.Draw(d, program: fc, state: fs, target: t, textures: [])
        let bottomRow = t.FlipY ? img.Height - 1 : 0
        check(pixel(img, 0, bottomRow) == [255, 255, 0, 255] && pixel(img, 1, img.Height - 1 - bottomRow) == [0, 0, 0, 255],
              "gl_FragCoord is (0.5, 0.5) at GL's bottom-left pixel (flipY \(t.FlipY))")
    }
}

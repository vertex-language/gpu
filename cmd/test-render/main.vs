// The built-in gpu's rendering on Metal: Metal source compiled at run
// time, a pipeline, a pass that clears and draws a triangle, read back.
package main

import (
    "gpu"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

let source = """
#include <metal_stdlib>
using namespace metal;
struct In { float2 pos [[attribute(0)]]; };
struct Out { float4 pos [[position]]; float4 color; };
vertex Out vs(In in [[stage_in]], constant float4& tint [[buffer(30)]]) {
    Out o;
    o.pos = float4(in.pos, 0.0, 1.0);
    o.color = tint;
    return o;
}
fragment float4 fs(Out in [[stage_in]]) { return in.color; }
"""

func main() -> int32 {
    let d = gpu.Default()
    if d.IsCPU {
        print("no Metal device: nothing to test")
        return 0
    }
    do {
        let lib = try gpu.Library(device: d, source: source)
        var desc = gpu.PipelineDescriptor()
        desc.Attributes = [gpu.VertexAttribute(index: 0, format: 29, offset: 0, buffer: 0)]   // float2
        desc.Layouts = [gpu.VertexLayout(buffer: 0, stride: 8, step: 1, rate: 1)]
        let p = try gpu.RenderPipeline(device: d, vertex: try lib.Function("vs"), fragment: try lib.Function("fs"), desc)
        guard let tex = gpu.Texture.Create(device: d, width: 8, height: 8, format: .rgba8) else {
            check(false, "a texture")
            return 1
        }
        let blue: [float32] = [0, 0, 1, 1]   // vsc_TODO #49
        guard let pass = gpu.RenderPass.Begin(device: d, color: tex, clearColor: blue, depth: nil) else {
            check(false, "a pass")
            return 1
        }
        pass.SetPipeline(p)
        pass.SetViewport(x: 0, y: 0, width: 8, height: 8, near: 0, far: 1)
        // The left half of the screen, as two triangles.
        let verts: [float32] = [-1, -1, 0, -1, -1, 1, 0, -1, 0, 1, -1, 1]
        let tint: [float32] = [1, 0, 0, 1]
        verts.withUnsafeBytes { b in pass.SetBytes(fragment: false, b.baseAddress!, count: b.count, index: 0) }
        tint.withUnsafeBytes { b in pass.SetBytes(fragment: false, b.baseAddress!, count: b.count, index: 30) }
        pass.Draw(primitive: 3, start: 0, count: 6, instances: 1)
        pass.End()
        var px = [uint8](repeating: 0, count: 8 * 8 * 4)
        px.withUnsafeMutableBytes { b in tex.Read(x: 0, y: 0, width: 8, height: 8, into: b.baseAddress!, bytesPerRow: 32) }
        check(Array(px[0..<4]) == [255, 0, 0, 255] && Array(px[7 * 4..<7 * 4 + 4]) == [0, 0, 255, 255],
              "Metal draws red over the left half of a blue clear (\(Array(px[0..<4])), \(Array(px[28..<32])))")
        // Bad source reports the compiler's log.
        do {
            _ = try gpu.Library(device: d, source: "this is not metal")
            check(false, "bad source is refused")
        } catch let e as gpu.RenderError {
            if case .compile(let log) = e { check(!log.isEmpty, "bad source is refused, with a log") } else { check(false, "bad source: \(e)") }
        }
    } catch {
        check(false, "\(error)")
    }
    print(failures == 0 ? "all passed" : "\(failures) failed")
    return failures == 0 ? 0 : 1
}

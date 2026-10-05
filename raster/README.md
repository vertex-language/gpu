# gpu/raster

Triangles, lines and points into pixels, on the GPU (Metal) or the CPU behind one API. On the CPU it shades with [`shader/interp`](../../shader); on the GPU, shaders are translated by [`shader/msl`](../../shader) and drawn with the built-in `gpu`'s render passes.

```vertex
import "gpu/raster"

let r = raster.Renderer()
let target = raster.Target(color: raster.Texture(width: 720, height: 1280, format: .rgba8), depth: nil, flipY: true)
r.Clear(target, color: raster.Color(0, 0, 0, 1), depth: nil, stencil: nil, state: state)
r.Draw(draw, program: program, state: state, target: target, textures: bindings)
```

`raster.EnableGPU()` turns the GPU on for every renderer, when the machine has a Metal device; it returns false when it doesn't, and drawing stays on the CPU.

## Types

- **`Renderer`** (class): Draws and clears. One per thread; it keeps its scratch space.
- **`Program`** (class): A vertex and fragment `interp.Executable`, linked (varyings by name and type), with each stage's uniform values and each input's attribute location.
- **`Draw`** (struct), **`Attribute`** (struct), **`Buffer`** (class), **`Primitive`**, **`ComponentType`**: What to draw: vertices from typed buffers (or a constant), optional indices (1, 2, 4 bytes, primitive restart), instances.
- **`State`** (struct): Viewport and depth range, scissor, culling and winding, depth and stencil tests, blending, color masks, polygon offset, line width.
- **`Target`** (class): A color image and a depth-stencil image. `FlipY` stores GL's bottom row last, as a window shows it.
- **`Texture`** (class), **`Level`** (class), **`Format`**: 2D and cube images with mip levels: `rgba8`, `rgba32f`, `depthStencil`; storage each level owns, read and written in place. `GenerateMipmaps`, `Complete`, `MipmapComplete`. With the GPU on, a texture also has a device copy: read levels through `Read(face:level:)` and write them through `Write(face:level:)` (or `Define`), and the two copies stay in step.
- **`EnableGPU`**, **`GPUEnabled`**, **`GPUCounters`** (`GPUStats`), **`FlushGPU`**: The GPU path, and what it has done (draws, clears, uploads, readbacks, and draws that fell back to the CPU).
- **`Sampler`** (struct), **`Binding`** (struct), **`Filter`**, **`Wrap`**, **`Swizzle`**, **`CompareFunc`**: How a texture is read.

## How it draws

Vertices are shaded in batches of lanes, each distinct one once. Triangles are clipped against the near and far planes, culled, and covered in fixed point (8 bits of subpixel) with an antisymmetric fill rule, so triangles sharing an edge never both draw a pixel or both miss it. Lines and points become two triangles each. Fragments are shaded in 2×2 quads, helpers included, then stencil, depth, blending and masks. Lookups pick a level of detail from the quad's derivatives; RGBA8 2D textures without mipmaps have a fast path.

## On the GPU

A program's two stages are translated to Metal source once and compiled by the driver. Pipelines are cached per program, vertex layout, blending and target formats; depth-stencil and sampler states per state. Draws into one target share a render pass, and a pass ends when something else is drawn into, a texture is sent, or the host reads. Textures are sent when the host's copy is newer and read back only when the host asks (`Read`). Fans and line loops become index lists; vertex data Metal can't read as it is (GL_FIXED, unnormalized bytes, unaligned strides) is converted to four 32-bit components a vertex. Whole-target clears start a pass cleared; scissored or masked ones draw a rectangle.

GL's conventions are kept by flipping instead of copying. A target stored bottom row first is drawn upside down, so its winding is reversed, its viewport and scissor count from the top, and `gl_FragCoord` and `dFdy` are corrected in the shader.

What the GPU can't take goes to the CPU: a shader `shader/msl` can't translate (3D textures, arrays and shadow samplers so far), or a swizzled texture. The target and textures are read back first. Depth and stencil stay in GPU memory.

`cmd/test-raster` runs every check twice, on the CPU and then on the GPU, and adds winding, scissor and `gl_FragCoord` checks on both kinds of target.

`cmd/test-raster` checks coverage, interpolation, filtering, depth, stencil and the window flip against values worked out by hand.

Part of the [`gpu`](../README.md) repository.

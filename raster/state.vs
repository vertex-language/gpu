package raster

import (
    "shader"
    "shader/interp"
)

/// What a draw's vertices make.
public enum Primitive: Equatable {
    case points
    case lines
    case lineStrip
    case lineLoop
    case triangles
    case triangleStrip
    case triangleFan
}

/// Which faces a test or cull applies to.
public enum Face: Equatable {
    case front
    case back
    case frontAndBack
}

public enum BlendFactor: Equatable {
    case zero, one
    case srcColor, oneMinusSrcColor, dstColor, oneMinusDstColor
    case srcAlpha, oneMinusSrcAlpha, dstAlpha, oneMinusDstAlpha
    case constantColor, oneMinusConstantColor, constantAlpha, oneMinusConstantAlpha
    case srcAlphaSaturate
}

public enum BlendEquation: Equatable {
    case add, subtract, reverseSubtract, min, max
}

public enum StencilOp: Equatable {
    case keep, zero, replace, increment, incrementWrap, decrement, decrementWrap, invert
}

/// The stencil test and its updates for one facing.
public struct StencilFace {
    public var Func: CompareFunc = .always
    public var Ref: int = 0
    public var ReadMask: uint32 = 0xFFFF_FFFF
    public var WriteMask: uint32 = 0xFFFF_FFFF
    public var Fail: StencilOp = .keep
    public var DepthFail: StencilOp = .keep
    public var Pass: StencilOp = .keep

    public init() {}
}

/// A rectangle in window coordinates (y up, as GL's).
public struct Rect: Equatable {
    public var X: int
    public var Y: int
    public var Width: int
    public var Height: int

    public init(x: int, y: int, width: int, height: int) {
        X = x
        Y = y
        Width = width
        Height = height
    }
}

/// The fixed-function state of a draw.
public struct State {
    public var Viewport = Rect(x: 0, y: 0, width: 0, height: 0)
    public var DepthNear: float32 = 0
    public var DepthFar: float32 = 1
    /// Only pixels inside it are touched; nil: no scissor.
    public var Scissor: Rect? = nil
    /// The faces not drawn; nil: none culled.
    public var Cull: Face? = nil
    /// Counter-clockwise triangles (in window coordinates) face front.
    public var FrontCCW = true
    /// The depth test; nil: off (and depth isn't written).
    public var DepthTest: CompareFunc? = nil
    public var DepthWrite = true
    public var StencilTest = false
    public var StencilFront = StencilFace()
    public var StencilBack = StencilFace()
    public var Blend = false
    public var BlendRGB: BlendEquation = .add
    public var BlendAlpha: BlendEquation = .add
    public var SrcRGB: BlendFactor = .one
    public var DstRGB: BlendFactor = .zero
    public var SrcAlpha: BlendFactor = .one
    public var DstAlpha: BlendFactor = .zero
    public var BlendColor = Color(0, 0, 0, 0)
    public var WriteRed = true
    public var WriteGreen = true
    public var WriteBlue = true
    public var WriteAlpha = true
    public var PolygonOffset = false
    public var OffsetFactor: float32 = 0
    public var OffsetUnits: float32 = 0
    public var LineWidth: float32 = 1

    public init() {}
}

/// Where a draw writes: a color image and a depth-stencil image, either
/// optional. A window's surface stores its rows top first (FlipY): GL's
/// window coordinates have y up.
public final class Target {
    public var Color: Texture? = nil
    public var ColorLevel = 0
    public var ColorFace = 0
    public var Depth: Texture? = nil
    public var DepthLevel = 0
    public var FlipY = false

    public init() {}

    public init(color: Texture?, depth: Texture?, flipY: bool = false) {
        Color = color
        Depth = depth
        FlipY = flipY
    }

    var colorLevel: Level? { Color?.At(face: ColorFace, level: ColorLevel) }
    var depthLevel: Level? { Depth?.At(level: DepthLevel) }

    /// The size draws are bounded by: the smaller of the attachments.
    public var Width: int {
        let c = colorLevel?.Width ?? int.max
        let d = depthLevel?.Width ?? int.max
        let w = min(c, d)
        return w == int.max ? 0 : w
    }

    public var Height: int {
        let c = colorLevel?.Height ?? int.max
        let d = depthLevel?.Height ?? int.max
        let h = min(c, d)
        return h == int.max ? 0 : h
    }
}

/// Bytes a draw reads vertices or indices from.
public final class Buffer {
    public var Bytes: [uint8]

    public init(_ bytes: [uint8] = []) {
        Bytes = bytes
    }
}

/// How one attribute's components are stored.
public enum ComponentType: Equatable {
    case float32
    case float16
    case int8
    case uint8
    case int16
    case uint16
    case int32
    case uint32
    /// GL_FIXED: 16.16 fixed point.
    case fixed
}

/// Where a vertex attribute comes from.
public struct Attribute {
    public var Enabled = false
    public var Buffer: Buffer? = nil
    public var Offset = 0
    /// Bytes between vertices; 0: tightly packed.
    public var Stride = 0
    public var Type: ComponentType = .float32
    public var Count = 4
    public var Normalized = false
    /// For integer attributes (glVertexAttribIPointer): the bits, not a float.
    public var Integer = false
    /// Instances per step of this attribute (0: per vertex).
    public var Divisor = 0
    /// The value when not Enabled (glVertexAttrib4f): bits of four components.
    public var Constant: [uint32] = [0, 0, 0, float32(1).bitPattern]

    public init() {}
}

/// One draw.
public struct Draw {
    public var Primitive: Primitive = .triangles
    public var First = 0
    public var Count = 0
    public var Instances = 1
    /// Indices: 1, 2 or 4 bytes each, from IndexOffset.
    public var Indices: Buffer? = nil
    public var IndexOffset = 0
    public var IndexSize = 2
    /// The index that restarts a strip or fan (GL_PRIMITIVE_RESTART_FIXED_INDEX).
    public var RestartIndex: uint32? = nil
    /// By attribute location.
    public var Attributes: [Attribute] = []

    public init() {}
}

/// LinkError is a vertex and fragment shader that don't fit together.
public struct LinkError: Error {
    public let Message: string

    public init(_ message: string) {
        Message = message
    }
}

/// A varying: where the vertex stage leaves it and the fragment stage reads it.
struct Link {
    let From: int
    let To: int
    let Slots: int
    let Flat: bool
    /// Integer varyings aren't interpolated either.
    let Integer: bool
}

/// A linked pair of shaders, and the uniform values they read.
public final class Program {
    public let Vertex: interp.Executable
    public let Fragment: interp.Executable
    /// Uniform values, by slot of each stage.
    public var VertexUniforms: [uint32]
    public var FragmentUniforms: [uint32]
    /// Each vertex input's attribute location, by Vertex.Inputs index.
    public var InputLocations: [int]
    var links: [Link] = []
    var varyingSlots = 0
    var vertexMachine: interp.Machine? = nil
    var fragmentMachine: interp.Machine? = nil
    /// Its Metal side, once made; or that it can't be.
    var gpuState: GPUProgram? = nil
    var gpuFailed = false

    public init(vertex: interp.Executable, fragment: interp.Executable) throws {
        Vertex = vertex
        Fragment = fragment
        VertexUniforms = [uint32](repeating: 0, count: vertex.SlotCount)
        FragmentUniforms = [uint32](repeating: 0, count: fragment.SlotCount)
        var locations: [int] = []
        for input in vertex.Inputs { locations.append(max(0, input.Location)) }   // vsc_TODO #48: not map
        InputLocations = locations
        if vertex.Module.Stage != .vertex || fragment.Module.Stage != .fragment {
            throw LinkError("a program needs a vertex and a fragment shader")
        }
        if vertex.Module.Version != fragment.Module.Version {
            throw LinkError("the shaders' GLSL versions differ")
        }
        var at = 0
        for input in fragment.Inputs {
            guard let out = vertex.Outputs.first(where: { $0.Name == input.Name }) else {
                throw LinkError("varying \(input.Name) isn't written by the vertex shader")
            }
            if out.Type != input.Type { throw LinkError("varying \(input.Name) has different types") }
            links.append(Link(From: out.Offset, To: input.Offset, Slots: input.Slots, Flat: input.Flat || out.Flat,
                              Integer: input.Type.Kind == .int || input.Type.Kind == .uint))
            at += input.Slots
        }
        varyingSlots = at
        // Uniforms of the same name must have the same type in both stages.
        for u in vertex.Uniforms {
            if let f = fragment.FindUniform(u.Name), f.Type != u.Type {
                throw LinkError("uniform \(u.Name) has different types")
            }
        }
    }
}

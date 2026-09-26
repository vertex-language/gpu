// Normalizations, activations and signal operations of audio networks,
// over [channels, length] row-major buffers: instance norm, layer norm
// across channels, style modulation, LeakyReLU, Snake, a running sum
// along each row, an LSTM cell's step, and a short-time Fourier
// transform and its inverse.
import (
    "gpu"
    "gpu/parallel"
    "math"
)

// ---- normalization across a row, and across channels ----

func _instanceNorm(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                   _ cols: int, _ eps: float32, _ affine: bool) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var s: float32 = 0
    var j = me
    while j < cols {
        s = s + x[base + j]
        j += step
    }
    let mean = parallel.GroupSum(s) / float32(cols)
    var q: float32 = 0
    j = me
    while j < cols {
        let d = x[base + j] - mean
        q = q + d * d
        j += step
    }
    let inv = math.Rsqrt(parallel.GroupSum(q) / float32(cols) + eps)
    let a: float32 = affine ? w[row] : 1
    let c: float32 = affine ? b[row] : 0
    j = me
    while j < cols {
        y[base + j] = (x[base + j] - mean) * inv * a + c
        j += step
    }
}

/// InstanceNorm writes each row of x (rows x cols) normalized to zero
/// mean and unit variance (biased, as InstanceNorm1d's) into y, then
/// scaled and shifted by that row's weight and bias if given.
public func InstanceNorm(_ x: gpu.Buffer<float32>, weight: gpu.Buffer<float32>?, bias: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>,
                         rows: int, cols: int, eps: float32 = 1e-5) async throws {
    if rows * cols == 0 {
        return
    }
    let affine = weight != nil && bias != nil
    try await _instanceNorm.Launch(x, weight ?? x, bias ?? x, y, cols, eps, affine, over: rows * RowGroup, workgroup: RowGroup)
}

func _channelNorm(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                  _ rows: int, _ cols: int, _ eps: float32, _ affine: bool) kernel {
    let t = gpu.Index.x
    if t >= cols {
        return
    }
    var s: float32 = 0
    var r = 0
    while r < rows {
        s = s + x[r * cols + t]
        r += 1
    }
    let mean = s / float32(rows)
    var q: float32 = 0
    r = 0
    while r < rows {
        let d = x[r * cols + t] - mean
        q = q + d * d
        r += 1
    }
    let inv = math.Rsqrt(q / float32(rows) + eps)
    r = 0
    while r < rows {
        let n = (x[r * cols + t] - mean) * inv
        y[r * cols + t] = affine ? n * w[r] + b[r] : n
        r += 1
    }
}

/// ChannelNorm is layer norm across channels: each column of x (rows
/// channels x cols positions) normalized over its rows, then scaled and
/// shifted by each row's weight and bias if given. It is F.layer_norm
/// over the channels of an NCL tensor, taken without transposing it.
public func ChannelNorm(_ x: gpu.Buffer<float32>, weight: gpu.Buffer<float32>?, bias: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>,
                        rows: int, cols: int, eps: float32 = 1e-5) async throws {
    if rows * cols == 0 {
        return
    }
    let affine = weight != nil && bias != nil
    try await _channelNorm.Launch(x, weight ?? x, bias ?? x, y, rows, cols, eps, affine, over: cols)
}

func _modulate(_ x: gpu.Span<float32>, _ gamma: gpu.Span<float32>, _ beta: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let r = i / cols
        y[i] = (1 + gamma[r]) * x[i] + beta[r]
    }
}

/// Modulate writes (1 + gamma[row]) · x + beta[row] into y, for x of
/// rows x cols: how AdaIN and adaptive layer norm apply a style's scale
/// and shift after normalizing.
public func Modulate(_ x: gpu.Buffer<float32>, gamma: gpu.Buffer<float32>, beta: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, cols: int) async throws {
    if x.count == 0 {
        return
    }
    try await _modulate.Launch(x, gamma, beta, y, cols, over: x.count)
}

// ---- elementwise ----

func _leakyReLU(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ slope: float32) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let v = x[i]
        y[i] = v > 0 ? v : v * slope
    }
}

/// LeakyReLU writes x where it is positive and slope · x elsewhere.
public func LeakyReLU(_ x: gpu.Buffer<float32>, slope: float32, into y: gpu.Buffer<float32>) async throws {
    if x.count == 0 {
        return
    }
    try await _leakyReLU.Launch(x, y, slope, over: x.count)
}

func _snake(_ x: gpu.Span<float32>, _ alpha: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let a = alpha[i / cols]
        let v = x[i]
        let s = math.Sin(a * v)
        y[i] = v + (1 / a) * (s * s)
    }
}

/// Snake writes x + sin²(α·x)/α into y, α a row's (rows x cols): the
/// periodic activation of BigVGAN and iSTFTNet's resblocks.
public func Snake(_ x: gpu.Buffer<float32>, alpha: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, cols: int) async throws {
    if x.count == 0 {
        return
    }
    try await _snake.Launch(x, alpha, y, cols, over: x.count)
}

func _axpby(_ x: gpu.Span<float32>, _ y: gpu.Span<float32>, _ out: gpu.MutableSpan<float32>, _ a: float32, _ b: float32) kernel {
    let i = gpu.Index.x
    if i < out.count {
        out[i] = a * x[i] + b * y[i]
    }
}

/// Axpby writes a·x + b·y into out, element by element: a residual sum,
/// an average, a difference.
public func Axpby(_ a: float32, _ x: gpu.Buffer<float32>, _ b: float32, _ y: gpu.Buffer<float32>, into out: gpu.Buffer<float32>) async throws {
    if out.count == 0 {
        return
    }
    try await _axpby.Launch(x, y, out, a, b, over: out.count)
}

// ---- running sums ----

func _cumSum(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ cols: int) kernel {
    let r = gpu.Index.x
    if r >= rows {
        return
    }
    // Compensated, so a long row keeps what PyTorch's double accumulator
    // keeps.
    var s: float32 = 0
    var lost: float32 = 0
    var j = 0
    while j < cols {
        let v = x[r * cols + j] - lost
        let t = s + v
        lost = (t - s) - v
        s = t
        y[r * cols + j] = s
        j += 1
    }
}

/// CumSum writes each row's running sum into y: torch.cumsum along the
/// length, a row a work-item.
public func CumSum(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int) async throws {
    if rows * cols == 0 {
        return
    }
    try await _cumSum.Launch(x, y, rows, cols, over: rows)
}

// ---- LSTM ----

func _lstmCell(_ gates: gpu.Span<float32>, _ c: gpu.MutableSpan<float32>, _ h: gpu.MutableSpan<float32>, _ hidden: int, _ first: bool) kernel {
    let j = gpu.Index.x
    if j >= hidden {
        return
    }
    let i = math.Sigmoid(gates[j])
    let f = math.Sigmoid(gates[hidden + j])
    let g = math.Tanh(gates[2 * hidden + j])
    let o = math.Sigmoid(gates[3 * hidden + j])
    let cell = first ? i * g : f * c[j] + i * g
    c[j] = cell
    h[j] = o * math.Tanh(cell)
}

/// LSTMCell is one step of torch.nn.LSTM's recurrence, from gates already
/// holding W_ih·x + b_ih + W_hh·h + b_hh (4·hidden: input, forget, cell,
/// output, PyTorch's order): c = f·c + i·g, h = o·tanh(c). On the first
/// step c starts at zero.
public func LSTMCell(gates: gpu.Buffer<float32>, cell c: gpu.Buffer<float32>, into h: gpu.Buffer<float32>, hidden: int, first: bool) async throws {
    try await _lstmCell.Launch(gates, c, h, hidden, first, over: hidden)
}

// ---- short-time Fourier transform ----

/// STFTFrames is how many frames STFT makes of length samples: centered,
/// 1 + length / hop.
public func STFTFrames(_ length: int, hop: int) -> int {
    return 1 + length / hop
}

// twiddles are cos and sin of 2πm/n for m < n, exact at the multiples of
// a quarter turn, so that a bin the transform makes real comes out real.
func twiddles(_ n: int) -> [float32] {
    var out = [float32](repeating: 0, count: 2 * n)
    for m in 0..<n {
        var c = 0.0
        var s = 0.0
        if 4 * m % n == 0 {
            let q = 4 * m / n
            c = q == 0 ? 1 : (q == 2 ? -1 : 0)
            s = q == 1 ? 1 : (q == 3 ? -1 : 0)
        } else {
            let a = 2 * 3.14159265358979323846 * float64(m) / float64(n)
            c = cos64(a)
            s = cos64(a - 3.14159265358979323846 / 2)
        }
        out[m] = float32(c)
        out[n + m] = float32(s)
    }
    return out
}

// cos64 is the cosine in double, for the host's tables: the angle
// reduced to [-π, π], then the Taylor series, which there is good to
// 1e-15 -- far past the float32 the tables are kept in.
func cos64(_ x: float64) -> float64 {
    let twoPi = 6.28318530717958647692
    var a = x - twoPi * (x / twoPi).rounded(.toNearestOrEven)
    if a > 3.14159265358979323846 { a -= twoPi }
    if a < -3.14159265358979323846 { a += twoPi }
    let z = a * a
    var term = 1.0
    var sum = 1.0
    var k = 1.0
    while k < 40 {
        term = -term * z / (k * (k + 1))
        sum += term
        k += 2
    }
    return sum
}

/// HannWindow is torch.hann_window(n) (periodic): 0.5 - 0.5·cos(2πk/n).
public func HannWindow(_ n: int) -> [float32] {
    var w = [float32](repeating: 0, count: n)
    for k in 0..<n {
        w[k] = float32(0.5 - 0.5 * cos64(2 * 3.14159265358979323846 * float64(k) / float64(n)))
    }
    return w
}

func _stft(_ x: gpu.Span<float32>, _ win: gpu.Span<float32>, _ tw: gpu.Span<float32>, _ mag: gpu.MutableSpan<float32>, _ phase: gpu.MutableSpan<float32>,
           _ length: int, _ n: int, _ hop: int, _ frames: int) kernel {
    let i = gpu.Index.x
    let bins = n / 2 + 1
    if i >= bins * frames {
        return
    }
    let f = i / frames
    let m = i - f * frames
    var re: float32 = 0
    var im: float32 = 0
    var k = 0
    while k < n {
        // Centered: the signal padded by n/2 each side, reflected.
        var p = m * hop + k - n / 2
        if p < 0 {
            p = -p
        }
        if p >= length {
            p = 2 * (length - 1) - p
        }
        let v = win[k] * x[p]
        let at = (f * k) % n
        re = re + v * tw[at]
        im = im - v * tw[n + at]
        k += 1
    }
    // A real signal's DC and Nyquist bins are real; the FFT PyTorch uses
    // makes them exactly so, and their phase is then 0 or π.
    if f == 0 || 2 * f == n {
        im = 0
    }
    mag[i] = math.Sqrt(re * re + im * im)
    phase[i] = math.Atan2(im, re)
}

/// STFT writes torch.stft(x, n, hop, window: hann(n), center: true,
/// return_complex: true)'s magnitude and angle into mag and phase: each
/// (n/2 + 1) bins x STFTFrames(length, hop) frames, row-major by bin. The
/// DFT is taken directly: this is for the short windows (n of 16 to 64)
/// of vocoders like iSTFTNet.
public func STFT(_ x: gpu.Buffer<float32>, length: int, n: int, hop: int, magnitude mag: gpu.Buffer<float32>, phase: gpu.Buffer<float32>) async throws {
    let frames = STFTFrames(length, hop: hop)
    precondition(length > n / 2, "neural: a centered STFT reflects n/2 = \(n / 2) samples, and there are \(length)")
    let d = x.Device
    let win = try await d.Upload(HannWindow(n))
    let tw = try await d.Upload(twiddles(n))
    try await _stft.Launch(x, win, tw, mag, phase, length, n, hop, frames, over: (n / 2 + 1) * frames)
}

func _istft(_ mag: gpu.Span<float32>, _ phase: gpu.Span<float32>, _ win: gpu.Span<float32>, _ tw: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
            _ n: int, _ hop: int, _ frames: int, _ length: int) kernel {
    let p = gpu.Index.x
    if p >= length {
        return
    }
    let bins = n / 2 + 1
    let at = p + n / 2  // in the centered signal
    var sum: float32 = 0
    var env: float32 = 0
    // The frames that cover at: m·hop ≤ at < m·hop + n.
    var m = at / hop
    while m >= 0 && m * hop + n > at {
        if m < frames {
            let k = at - m * hop
            // The inverse real DFT at k: X₀ and the Nyquist bin once
            // (their imaginary parts ignored, as a C2R transform does),
            // the others twice.
            var v: float32 = 0
            var f = 0
            while f < bins {
                let a = mag[f * frames + m]
                let ph = phase[f * frames + m]
                let re = a * math.Cos(ph)
                let t = (f * k) % n
                if f == 0 || 2 * f == n {
                    v = v + re * tw[t]
                } else {
                    let im = a * math.Sin(ph)
                    v = v + 2 * (re * tw[t] - im * tw[n + t])
                }
                f += 1
            }
            let w = win[k]
            sum = sum + w * (v / float32(n))
            env = env + w * w
        }
        m -= 1
    }
    y[p] = sum / env
}

/// ISTFTLength is how many samples ISTFT makes of frames frames:
/// hop · (frames - 1), the centered padding taken off.
public func ISTFTLength(_ frames: int, hop: int) -> int {
    return hop * (frames - 1)
}

/// ISTFT writes torch.istft(mag · e^(i·phase), n, hop, window: hann(n),
/// center: true) into y: mag and phase are (n/2 + 1) bins x frames,
/// row-major by bin, and y is ISTFTLength(frames, hop) samples: each
/// frame inverted, windowed and overlap-added, divided by the window's
/// squared sum where they overlap.
public func ISTFT(magnitude mag: gpu.Buffer<float32>, phase: gpu.Buffer<float32>, frames: int, n: int, hop: int, into y: gpu.Buffer<float32>) async throws {
    let length = ISTFTLength(frames, hop: hop)
    if length <= 0 {
        return
    }
    precondition(hop <= n / 2, "neural: ISTFT with hop \(hop) leaves the window's ends uncovered for n = \(n)")
    let d = mag.Device
    let win = try await d.Upload(HannWindow(n))
    let tw = try await d.Upload(twiddles(n))
    try await _istft.Launch(mag, phase, win, tw, y, n, hop, frames, length, over: length)
}

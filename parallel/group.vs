// Group-scope device functions: every work-item of a workgroup calls them
// together, as every one must reach gpu.Barrier(), and they return to each
// the same answer (a reduction) or its own (a scan). They are @inlinable, so
// a kernel in any module compiles them into itself; nothing is launched.
//
// Each goes through shared storage in a fixed order -- a tree for
// reductions, Hillis-Steele for scans -- so a result is the same on every
// device for the same group size, floating-point sums included. That is
// what lets the CPU device be their oracle bit for bit. A group holds at
// most MaxGroup work-items.
import "gpu"
import "gpu/dtype"

/// MaxGroup is the most work-items a group-scope function supports.
public let MaxGroup = 1024

/// GroupRank is this work-item's index within its workgroup, counting x
/// fastest: the order the group-scope functions reduce and scan in.
@inlinable public func GroupRank() -> int {
    return gpu.LocalIndex.x + gpu.GroupSize.x * (gpu.LocalIndex.y + gpu.GroupSize.y * gpu.LocalIndex.z)
}

/// GroupCount is how many work-items the workgroup has.
@inlinable public func GroupCount() -> int {
    return gpu.GroupSize.x * gpu.GroupSize.y * gpu.GroupSize.z
}

/// _pow2 is the least power of two no smaller than n.
@inlinable public func _pow2(_ n: int) -> int {
    var w = 1
    while w < n {
        w *= 2
    }
    return w
}

/// _groupReduce combines x over the workgroup: op 0 sums (Plus), 1 takes
/// the least, 2 the greatest.
@inlinable public func _groupReduce<T: dtype.Number>(_ x: T, _ op: int32) -> T {
    let s = gpu.Shared<T>(count: 1024)
    let me = GroupRank()
    let n = GroupCount()
    s[me] = x
    gpu.Barrier()
    var stride = _pow2(n) / 2
    while stride > 0 {
        if me < stride && me + stride < n {
            let a = s[me]
            let b = s[me + stride]
            if op == 0 {
                s[me] = T.Plus(a, b)
            } else if op == 1 {
                s[me] = b < a ? b : a
            } else {
                s[me] = b > a ? b : a
            }
        }
        gpu.Barrier()
        stride /= 2
    }
    let r = s[0]
    gpu.Barrier()
    return r
}

/// GroupSum is the sum of x over the workgroup, added in a fixed tree
/// order; an integer sum wraps.
@inlinable public func GroupSum<T: dtype.Number>(_ x: T) -> T { return _groupReduce(x, 0) }

/// GroupMin is the least x in the workgroup.
@inlinable public func GroupMin<T: dtype.Number>(_ x: T) -> T { return _groupReduce(x, 1) }

/// GroupMax is the greatest x in the workgroup.
@inlinable public func GroupMax<T: dtype.Number>(_ x: T) -> T { return _groupReduce(x, 2) }

/// _groupScan is the inclusive and exclusive prefix sums of x at this
/// work-item, and the group's total, which is the last inclusive sum
/// exactly: the order a sum is taken in is the same for all three.
@inlinable public func _groupScan<T: dtype.Number>(_ x: T) -> (inclusive: T, exclusive: T, total: T) {
    let s = gpu.Shared<T>(count: 1024)
    let me = GroupRank()
    let n = GroupCount()
    s[me] = x
    gpu.Barrier()
    var offset = 1
    while offset < n {
        var v = s[me]
        if me >= offset {
            v = T.Plus(s[me - offset], v)
        }
        gpu.Barrier()
        s[me] = v
        gpu.Barrier()
        offset *= 2
    }
    let inclusive = s[me]
    var exclusive: T = 0
    if me > 0 {
        exclusive = s[me - 1]
    }
    let total = s[n - 1]
    gpu.Barrier()
    return (inclusive: inclusive, exclusive: exclusive, total: total)
}

/// GroupScan is the inclusive prefix sum of x up to and including this
/// work-item, in GroupRank order.
@inlinable public func GroupScan<T: dtype.Number>(_ x: T) -> T {
    return _groupScan(x).inclusive
}

/// GroupExclusiveScan is the prefix sum of x before this work-item, and
/// the whole group's total.
@inlinable public func GroupExclusiveScan<T: dtype.Number>(_ x: T) -> (prefix: T, total: T) {
    let r = _groupScan(x)
    return (prefix: r.exclusive, total: r.total)
}

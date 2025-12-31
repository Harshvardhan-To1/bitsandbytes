import ctypes as ct

from bitsandbytes.backends.cuda import ops as cuda_ops


def test_as_ctypes_int32_or_int64_threshold():
    n32, use64 = cuda_ops._as_ctypes_int32_or_int64(cuda_ops._INT32_MAX)
    assert use64 is False
    assert isinstance(n32, ct.c_int32)

    n64, use64 = cuda_ops._as_ctypes_int32_or_int64(cuda_ops._INT32_MAX + 1)
    assert use64 is True
    assert isinstance(n64, ct.c_int64)


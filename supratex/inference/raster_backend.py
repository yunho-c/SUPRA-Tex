"""Forward rasterization for inference on CUDA and ROCm.

The HIP port's CUDA-style triangle binning is unsuitable for MI300X. Use EGL
for clipping, coverage and perspective-correct barycentrics. Attribute lookup
and bilinear texture filtering continue to run on the accelerator.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def rasterizer():
    import torch
    import nvdiffrast.torch as dr

    if not torch.version.hip:
        return dr

    class Backend:
        RasterizeCudaContext = _EGLContext
        rasterize = staticmethod(_rasterize)
        interpolate = staticmethod(dr.interpolate)
        texture = staticmethod(dr.texture)
    return Backend


@lru_cache(maxsize=1)
def _software_device_index():
    # EGL enumerates compute-only DRM devices too. Select Mesa's explicit
    # software device before creating a graphics context on a headless VM.
    import ctypes as ct
    egl = ct.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ct.c_char_p]
    egl.eglGetProcAddress.restype = ct.c_void_p
    def extension(name, result, *args):
        addr = egl.eglGetProcAddress(name)
        if not addr:
            raise RuntimeError(f"EGL lacks {name.decode()}")
        return ct.CFUNCTYPE(result, *args)(addr)

    ptr = ct.c_void_p
    devices = extension(b"eglQueryDevicesEXT", ct.c_uint, ct.c_int, ct.POINTER(ptr), ct.POINTER(ct.c_int))
    string = extension(b"eglQueryDeviceStringEXT", ct.c_char_p, ptr, ct.c_int)
    count = ct.c_int()
    if not devices(0, None, ct.byref(count)) or not count.value:
        raise RuntimeError("No EGL devices available")
    handles = (ptr * count.value)()
    if not devices(count.value, handles, ct.byref(count)):
        raise RuntimeError("EGL could not enumerate graphics devices")
    for i, handle in enumerate(handles):
        if b"EGL_MESA_device_software" in (string(handle, 0x3055) or b"").split():
            return i
    raise RuntimeError("The ROCm texture exporter requires Mesa's software EGL device")


class _EGLContext:
    def __init__(self, device=None):
        import moderngl
        self.gl = moderngl.create_standalone_context(require=330, backend='egl',
                                               device_index=_software_device_index())
        self.program = self.gl.program(
            vertex_shader="""#version 330
                in vec4 clip;
                void main() { gl_Position = clip; }""",
            geometry_shader="""#version 330
                layout(triangles) in;
                layout(triangle_strip, max_vertices=3) out;
                out vec2 bary;
                flat out int face;
                void main() {
                    for (int i = 0; i < 3; ++i) {
                        gl_Position = gl_in[i].gl_Position;
                        face = gl_PrimitiveIDIn;
                        bary = vec2(i == 0, i == 1);
                        EmitVertex();
                    }
                    EndPrimitive();
                }""",
            fragment_shader="""#version 330
                in vec2 bary;
                flat in int face;
                out vec4 raster;
                void main() {
                    raster = vec4(bary, gl_FragCoord.z * 2.0 - 1.0, float(face + 1));
                }""",
        )


def _rasterize(context, positions, triangles, resolution):
    import moderngl
    import numpy as np
    import torch
    height, width = resolution
    gl = context.gl
    with gl:
        # OpenGL reads and writes row zero at the lower edge, matching
        # nvdiffrast's bottom-up clip-space convention.
        resources = []
        try:
            def keep(resource):
                resources.append(resource)
                return resource

            pixels = keep(gl.texture((width, height), 4, dtype='f4'))
            depth = keep(gl.depth_renderbuffer((width, height)))
            fbo = keep(gl.framebuffer([pixels], depth))
            fbo.use()
            gl.enable_only(moderngl.DEPTH_TEST)
            gl.depth_func = '<='
            indices = keep(gl.buffer(triangles.detach().int().cpu().contiguous().numpy()))
            images = []
            for verts in positions:
                vertices = keep(gl.buffer(verts.detach().float().cpu().contiguous().numpy()))
                vao = keep(gl.vertex_array(context.program, [(vertices, '4f', 'clip')], indices))
                fbo.clear(depth=1.0)
                vao.render(moderngl.TRIANGLES)
                data = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32)
                images.append(data.reshape(height, width, 4))
            result = torch.from_numpy(np.stack(images)).to(positions.device)
        finally:
            for resource in reversed(resources):
                resource.release()
    # SUPRA inference consumes only the raster, without coordinate derivatives.
    return result, None

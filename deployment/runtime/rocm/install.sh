#!/usr/bin/env bash
# Run inside the ROCm PyTorch container documented in docs/mi300x.md.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd)
build=${1:?Specify a persistent directory for native source and build products}
mkdir -p "$build"
build=$(cd "$build" && pwd)
export GPU_ARCHS=gfx942 PYTORCH_ROCM_ARCH=gfx942 BUILD_TARGET=rocm
export ROCM_PATH=/opt/rocm MAX_JOBS=${MAX_JOBS:-5}
python -c "import torch; assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx942', 'expected MI300X ROCm container'"
apt-get update -qq
apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1-mesa-dri
python -m pip install -r "$repo/deployment/runtime/requirements-python.txt" moderngl==5.12.0 glcontext==3.1.0 pybind11==3.1.0

checkout() {
    local directory="$build/$1" url="$2" revision="$3"
    if [[ ! -d "$directory/.git" ]]; then
        git init -q "$directory"
        git -C "$directory" remote add origin "$url"
        git -C "$directory" fetch --depth 1 origin "$revision"
        git -C "$directory" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || {
        echo "Expected $directory at $revision" >&2
        return 1
    }
    git -C "$directory" submodule update --init --recursive --depth 1
}

checkout dense https://github.com/martinus/unordered_dense.git 3234af2c03549bc85656bfd3a86993bf1cd8aef1
checkout flex https://github.com/Cardboard-box-a/FlexGEMM-rocm.git 2a92675f85027f66c7108381a5c0312b7b393310
checkout CuMesh https://github.com/Cardboard-box-a/CuMesh.git 2ef5f9d121e460d1aca9fd8cadb9264d6fa94ba4
checkout nvdiffrast-hip https://github.com/Cardboard-box-a/nvdiffrast-hip.git d29892337190f7e55f891b319d1daa6686bb681c
checkout trellis https://github.com/microsoft/TRELLIS.2.git 5565d240c4a494caaf9ece7a554542b76ffa36d3
checkout aiter https://github.com/ROCm/aiter.git 5c90446573c9544cd75b3008c700afe74a01431a
patch="$repo/deployment/runtime/rocm/o-voxel.patch"
if ! git -C "$build/trellis" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$build/trellis" apply "$patch"
fi
export CPLUS_INCLUDE_PATH="$build/dense/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
for project in flex CuMesh nvdiffrast-hip trellis/o-voxel; do
    (cd "$build/$project" && python setup.py build_ext --inplace)
done
(cd "$build/nvdiffrast-hip" && python setup.py egg_info)
PREBUILD_KERNELS=0 python -m pip install --no-deps --no-build-isolation -e "$build/aiter"
python - "$build" "$repo" <<'PY'
from pathlib import Path
import sys
import sysconfig

build, repo = map(Path, sys.argv[1:])
paths = [build / p for p in ("flex", "CuMesh", "nvdiffrast-hip", "trellis/o-voxel")]
paths.extend(repo / "deployment/runtime" / p for p in ("utils3d", "moge"))
(Path(sysconfig.get_path("purelib")) / "supra-mi300.pth").write_text(
    "".join(f"{path}\n" for path in paths)
)
PY
python -c 'import torch, aiter, flex_gemm, cumesh, o_voxel, nvdiffrast.torch, moge.model.v2, utils3d'

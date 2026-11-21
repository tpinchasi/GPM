"""The agent installs ONNX Runtime from a locked set (D179), on Windows too (D185).

docs/spec/host-agent.md §2.3. Nothing here reaches PyPI or needs a GPU: the sets are built here and
the machine's cards are what a test says they are. One test installs a set with real pip, from a
directory of small wheels made here, into a real environment on whatever system runs it — on the
Windows job, a Windows environment (`Scripts\\python.exe`). The release's tool that makes the sets
(`tools/onnx-lock`) runs against a stand-in for uv and for PyPI's JSON API.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from gpm_agent import onnx_install
from gpm_agent import platform as host_platform
from gpm_agent.installs import Installer, InstallRefused, Target, this_machine
from gpm_agent.onnx_install import OnnxRuntimeSource, OnnxSet
from gpm_agent.platform import LINUX, WINDOWS
from gpm_agent.vllm_install import this_python, venv_python
from test_agent_vllm_install import Started, digest, sha256, wheel

REPO = Path(__file__).resolve().parent.parent
PYTHON = this_python()
WINDOWS_PC = Target(WINDOWS, "amd64")

#: A Windows PC's card: Windows' driver numbers are not Linux's.
RTX = {"kind": "cuda", "name": "NVIDIA GeForce RTX 4070", "compute_capability": "8.9", "driver": "581.57"}
OLD_DRIVER = {**RTX, "driver": "572.16"}


def onnx_set(system="windows", accelerator="cpu", version="1.32.0", *, platform="x86_64", requirements="",
             size=10**8, floor=None, glibc=None) -> OnnxSet:
    return OnnxSet(onnxruntime=version, python=PYTHON, system=system, platform=platform, accelerator=accelerator,
                   requirements=requirements, size_bytes=size, glibc=glibc,
                   driver_floor=floor if accelerator == "cuda" else None)


SETS = (onnx_set(), onnx_set(accelerator="cuda", floor="580"),
        onnx_set("linux", glibc="2.28"), onnx_set("linux", "cuda", floor="580.65.06", glibc="2.28"))


def source(*sets, target=WINDOWS_PC, cards=(RTX,), glibc=None) -> OnnxRuntimeSource:
    return OnnxRuntimeSource(target, python=PYTHON, sets=list(sets or SETS),
                             cards=lambda: None if cards is None else [dict(c) for c in cards],
                             glibc=lambda: glibc, run=never_run, environ={})


async def never_run(argv, **_):
    raise AssertionError(f"nothing runs while a set is chosen: {argv}")


# --- which set, on a Windows PC ---


async def test_a_windows_pc_gets_the_windows_sets_and_cudas_where_its_card_and_driver_can_run_it():
    assert (await source().latest(None)).version == "1.32.0"
    chosen, not_cuda = await source().choose("1.32.0")
    assert (chosen.system, chosen.accelerator, not_cuda) == ("windows", "cuda", [])


@pytest.mark.parametrize(("cards", "says"), [
    (None, "nvidia-smi does not answer here"),
    ((OLD_DRIVER,), "driver 572.16 is older than 580, which ONNX Runtime's CUDA needs"),
    (({**RTX, "compute_capability": "5.0"},), "compute capability 5.0"),
])
async def test_a_pc_whose_card_cannot_run_cudas_set_gets_the_processors_and_says_why(cards, says):
    chosen, not_cuda = await source(cards=cards).choose("1.32.0")
    assert (chosen.system, chosen.accelerator) == ("windows", "cpu")
    assert says in "; ".join(not_cuda)


async def test_a_windows_pc_never_takes_a_linux_set_and_says_so_when_it_has_none():
    with pytest.raises(InstallRefused, match="no ONNX Runtime set for .* on windows x86_64"):
        await source(*SETS[2:]).latest(None)
    with pytest.raises(InstallRefused, match="not arm64|x86_64 and aarch64"):
        await source(target=Target(WINDOWS, "x86")).latest(None)


async def test_linux_still_gets_linux_sets():
    chosen, _ = await source(target=Target(LINUX, "amd64"), cards=({**RTX, "driver": "580.95.05"},),
                               glibc="2.35").choose("1.32.0")
    assert (chosen.system, chosen.accelerator) == ("linux", "cuda")


# --- a set installed with real pip, on the system running the test ---


def fake_onnxruntime(directory: Path) -> tuple[str, int]:
    """Three small wheels standing in for the set — the check imports numpy, PIL and onnxruntime."""
    directory.mkdir()
    made = [
        wheel(directory, "onnxruntime", "1.32.0", {"onnxruntime/__init__.py": "__version__ = '1.32.0'\n"}),
        wheel(directory, "numpy", "2.3.5", {"numpy/__init__.py": ""}),
        wheel(directory, "pillow", "12.0.0", {"PIL/__init__.py": ""}),
    ]
    requirements = "".join(f"{w.name.split('-')[0]}=={w.name.split('-')[1]} \\\n    --hash=sha256:{sha256(w)}\n"
                           for w in made)
    return requirements, sum(w.stat().st_size for w in made)


async def test_a_set_installs_with_real_pip_and_its_interpreter_runs_from_where_the_version_lives(tmp_path):
    machine = this_machine()
    here = OnnxRuntimeSource(machine).system
    if here is None or OnnxRuntimeSource(machine).processor is None:
        pytest.skip("no ONNX Runtime sets are made for this machine")
    requirements, size = fake_onnxruntime(tmp_path / "wheels")
    onnx = OnnxRuntimeSource(machine, sets=[onnx_set(here, platform=OnnxRuntimeSource(machine).processor,
                                                     requirements=requirements, size=size)],
                             cards=lambda: None, glibc=lambda: "2.99", macos=lambda: "99.0",
                             environ={"GPM_PYPI_INDEX": str(tmp_path / "wheels")})
    engine_dir, server = tmp_path / "engines" / "onnx", Started()
    installer = Installer(onnx, engine_dir, server, free_disk_bytes=lambda: 10**12)
    await installer.want("latest", False)
    await installer.finished()

    state = installer.state()
    assert state["error"] is None and state["running"] == "1.32.0", state
    python = venv_python(engine_dir / "1.32.0" / "venv")
    assert server.started == [(python, "1.32.0")]
    assert python.parent.name == ("Scripts" if host_platform.system() == WINDOWS else "bin")
    assert (engine_dir / "1.32.0" / onnx_install.SERVER).read_bytes() == onnx_install.server_source()
    ran = subprocess.run([str(python), "-I", "-c", "import sys, onnxruntime; print(sys.prefix, onnxruntime.__version__)"],
                         capture_output=True, text=True, timeout=60, check=False)
    assert ran.returncode == 0, ran.stderr
    prefix, version = ran.stdout.rsplit(" ", 1)
    assert Path(prefix).resolve() == (engine_dir / "1.32.0" / "venv").resolve() and version.strip() == "1.32.0"


# --- the release's tool, for Windows ---


def load_tool():
    path = REPO / "tools" / "onnx-lock"
    spec = importlib.util.spec_from_loader("onnx_lock", importlib.machinery.SourceFileLoader("onnx_lock", str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()

PYPI = {
    ("onnxruntime-gpu", "1.32.0"): ["onnxruntime_gpu-1.32.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                                    "onnxruntime_gpu-1.32.0-cp312-cp312-win_amd64.whl"],
    ("onnxruntime", "1.32.0"): ["onnxruntime-1.32.0-cp312-cp312-win_amd64.whl",
                                "onnxruntime-1.32.0-cp312-cp312-macosx_14_0_arm64.whl"],
    ("nvidia-cuda-runtime", "13.0.96"): ["nvidia_cuda_runtime-13.0.96-py3-none-manylinux_2_27_x86_64.whl",
                                         "nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl"],
    ("numpy", "2.3.5"): ["numpy-2.3.5-cp312-cp312-win_amd64.whl", "numpy-2.3.5-cp312-cp312-win32.whl"],
    ("pillow", "12.0.0"): ["pillow-12.0.0-cp312-cp312-win_amd64.whl"],
}


def pypi_files(name, version):
    return [{"filename": f, "size": 1000, "sha256": digest(f), "packagetype": "bdist_wheel"} for f in PYPI[(name, version)]]


def uv_for(runtime: str):
    asked = []

    def compile(requirements, python, system, processor, level, index, exclude_newer):
        asked.append((system, processor, level))
        pins = [k for k in PYPI if k[0] in (runtime, "numpy", "pillow") or (runtime.endswith("gpu") and k[0].startswith("nvidia"))]
        return "".join(f"{name}=={version}" + "".join(f" \\\n    --hash=sha256:{digest(f)}" for f in PYPI[(name, version)]) + "\n"
                       for name, version in pins)

    return compile, asked


def test_a_windows_cuda_set_takes_the_windows_wheels_and_windows_driver_floor():
    compile, asked = uv_for("onnxruntime-gpu")
    locked = TOOL.lock("1.32.0", "3.12", "windows-x86_64-cuda", compile=compile, files=pypi_files)
    assert asked == [("windows", "x86_64", "")]
    assert (locked["system"], locked["platform"], locked["accelerator"]) == ("windows", "x86_64", "cuda")
    assert locked["driver_floor"] == "580" and locked["cuda_runtime"] == "13.0.96"
    assert "glibc" not in locked and "macos" not in locked
    assert all(digest(f) in locked["requirements"] for f in (
        "onnxruntime_gpu-1.32.0-cp312-cp312-win_amd64.whl", "nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl",
        "numpy-2.3.5-cp312-cp312-win_amd64.whl"))
    assert digest("numpy-2.3.5-cp312-cp312-win32.whl") not in locked["requirements"]
    assert TOOL.file_name(locked) == "onnxruntime-1.32.0-py3.12-windows-x86_64-cuda.json"
    assert OnnxSet.from_json(json.loads(json.dumps(locked))).system == "windows"


def test_a_windows_processor_set_has_no_driver_floor():
    compile, _ = uv_for("onnxruntime")
    locked = TOOL.lock("1.32.0", "3.12", "windows-x86_64-cpu", compile=compile, files=pypi_files)
    assert locked["accelerator"] == "cpu" and "driver_floor" not in locked
    assert digest("onnxruntime-1.32.0-cp312-cp312-win_amd64.whl") in locked["requirements"]


def test_windows_tags_are_pips_order_for_64_bit_windows():
    tags = TOOL.windows_tags("3.12")
    assert tags[0] == ("cp312", "cp312", "win_amd64") and ("py3", "none", "any") in tags
    assert all(platform in ("win_amd64", "any") for _, _, platform in tags)


def test_every_set_the_agent_carries_is_whole_and_windows_has_both_for_every_python():
    carried = onnx_install.load_sets()
    for locked in carried:
        assert locked.system in ("linux", "windows", "macos") and locked.platform in ("x86_64", "aarch64")
        assert (locked.driver_floor is not None) == (locked.accelerator == "cuda")
        lines = [line for line in locked.requirements.splitlines() if "==" in line]
        assert lines and all("--hash=sha256:" in locked.requirements.split(line, 1)[1][:200] for line in lines)
        assert any(line.startswith(("onnxruntime==", "onnxruntime-gpu==")) for line in lines)
    windows = {(s.python, s.accelerator) for s in carried if s.system == "windows"}
    assert windows == {(p, a) for p in ("3.11", "3.12", "3.13", "3.14") for a in ("cpu", "cuda")}

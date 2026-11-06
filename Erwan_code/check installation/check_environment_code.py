"""Check the notebook's environment; optionally install missing UI packages.

The checker itself uses only Python's standard library. Scientific imports run
in separate processes, with temporary caches, never in the notebook kernel.
"""
from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = {
    "holopy": "3.5.0", "numpy": "1.20.3", "scipy": "1.10.1",
    "xarray": "0.20.1", "pandas": "1.5.3", "matplotlib": "3.7.1",
    "Pillow": "11.3.0", "ipywidgets": "8.1.7", "ipympl": "0.9.7",
    "ipykernel": "6.30.1", "IPython": "8.18.1",
    "h5py": "3.14.0", "h5netcdf": "1.8.1", "PyYAML": "6.0.2",
    "emcee": "2.2.1", "cma": "4.0.0",
    "traitlets": None, "jupyter_client": None, "jupyter_core": None,
}
FRONTENDS = ("notebook", "nbclassic", "jupyterlab", "jupyter_server",
             "widgetsnbextension", "jupyterlab_widgets")
INTERFACE_PACKAGES = ("ipywidgets", "ipympl", "ipykernel")
MARKER = "HOLOGRAM_ENV_REPORT="

METADATA_PROBE = r'''
import importlib.metadata as metadata
import json
import platform
import sys
versions = {}
for name in json.loads(sys.argv[1]):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print("HOLOGRAM_ENV_REPORT=" + json.dumps(dict(
    executable=sys.executable, python=platform.python_version(),
    python_info=list(sys.version_info[:3]), system=platform.system(),
    machine=platform.machine(), packages=versions)))
'''

PROBES = {
    "Scientific libraries and xarray interpolation": r'''
import numpy as np
import scipy.ndimage, scipy.signal, scipy.optimize
import pandas, xarray as xr
import h5py, h5netcdf, yaml, emcee, cma
from PIL import Image
# Exercise the pandas/xarray path used in HoloPy, not just their imports.
data = xr.DataArray([1., 2., 3.], coords={"x": [0., 1., 2.]}, dims="x")
assert np.isclose(data.interp(x=[.5]).values[0], 1.5)
''',
    "HoloPy, inference backend and small MieLens hologram": r'''
import numpy as np
import holopy as hp
from holopy.scattering import Sphere, calc_holo
from holopy.scattering.theory import MieLens
from holopy.inference import ExactModel, NmpfitStrategy, prior
from holopy.inference.third_party import nmpfit
assert callable(hp.fit) and callable(nmpfit.mpfit)
h = calc_holo(hp.detector_grid(shape=(8, 8), spacing=.1),
              Sphere(n=1.45, r=.5, center=(.35, .35, 10.)),
              medium_index=1.33, illum_wavelen=.66,
              illum_polarization=(1, 0), theory=MieLens(.8))
assert h.size == 64 and np.isfinite(h.values).all()
''',
    "Matplotlib PNG generation": r'''
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
figure = Figure(figsize=(1, 1))
canvas = FigureCanvasAgg(figure)
figure.subplots().plot([0, 1], [0, 1])
with BytesIO() as stream:
    canvas.print_png(stream)
    assert stream.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
''',
    "Jupyter kernel and widget creation": r'''
import ipykernel, IPython, jupyter_client, jupyter_core
import ipywidgets as W
widget = W.Dropdown(options=["test"])
assert widget.value == "test"
widget.close()
''',
    "Interactive Matplotlib backend": r'''
from ipympl.backend_nbagg import Canvas, FigureManager
from hologram_analysis.canvas import ManagedCanvas
''',
    "Local analysis code and interface creation": r'''
from pathlib import Path
import hologram_functions as hf
import hologram_analysis
project = Path(PROJECT_ROOT).resolve()
assert Path(hf.__file__).resolve().parent == project, "Another project copy was imported"
assert Path(hologram_analysis.__file__).resolve().parent == project / "hologram_analysis"
ui = hf.NotebookInterface(base_dir=PROJECT_ROOT)
try:
    assert ui.widget is not None
finally:
    ui.close()
''',
}


def command_text(arguments, system=None):
    """Commands are displayed for the target host; execution never uses a shell."""
    if (system or os.name) in ("Windows", "nt"):
        return subprocess.list2cmdline([str(arg) for arg in arguments])
    return " ".join(shlex.quote(str(arg)) for arg in arguments)


def run_probe(python, code, arguments=(), timeout=30):
    with TemporaryDirectory(prefix="hologram-check-") as cache:
        env = os.environ.copy()
        env.update(MPLBACKEND="Agg", MPLCONFIGDIR=cache,
                   PYTHONDONTWRITEBYTECODE="1", MPI4PY_RC_INITIALIZE="0")
        try:
            result = subprocess.run(
                [python, "-B", "-c", code] + list(arguments),
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                errors="replace", timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                report = json.loads(line[len(MARKER):])
            except ValueError:
                break
            if result.returncode == 0:
                return report
    detail = (result.stderr.strip() or result.stdout.strip()
              or "Process exited with code {}".format(result.returncode))
    return {"ok": False, "error": detail[-5000:]}


def inspect_metadata(python):
    return run_probe(python, METADATA_PROBE,
                     [json.dumps(list(PACKAGES) + list(FRONTENDS))], timeout=10)


def functional_check(python, source, timeout):
    wrapper = ("import sys, json, traceback\n"
               "sys.path.insert(0, {root!r})\n"
               "PROJECT_ROOT = {root!r}\n"
               "try:\n"
               "    exec({source!r})\n"
               "    result = dict(ok=True)\n"
               "except Exception:\n"
               "    result = dict(ok=False, error=traceback.format_exc())\n"
               "print({marker!r} + json.dumps(result))\n").format(
                   root=str(ROOT), source=source, marker=MARKER)
    return run_probe(python, wrapper, timeout=timeout)


def install_missing_interface(report):
    """Install only missing UI packages; constrain existing baseline packages."""
    missing = [name for name in INTERFACE_PACKAGES
               if report["packages"][name] is None]
    if not missing:
        print("Interface packages are already installed; nothing to install.")
        return True
    if report["python_info"][:2] < [3, 9]:
        print("Python 3.9 or newer is needed for the recorded UI versions.")
        return False
    with TemporaryDirectory(prefix="hologram-install-") as folder:
        constraints = Path(folder) / "constraints.txt"
        constraints.write_text("".join(
            "{}=={}\n".format(name, version)
            for name, version in report["packages"].items()
            if version and version.lower() != "unknown"), encoding="utf-8")
        command = [report["executable"], "-m", "pip", "install",
                   "--disable-pip-version-check", "--constraint", str(constraints)]
        command += ["{}=={}".format(name, PACKAGES[name]) for name in missing]
        print("Installing missing interface packages: " + ", ".join(missing), flush=True)
        try:
            result = subprocess.run(command, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("Installation failed:", exc)
            return False
    if result.returncode:
        print("Installation failed. See pip's message above; existing scientific "
              "package versions were constrained.")
        return False
    print("Installation finished. Restart the notebook kernel before using the interface.")
    return True


def print_help_commands(report):
    python = report["executable"]
    system = report["system"]
    print("\nNEXT STEPS")
    missing_ui = [name for name in INTERFACE_PACKAGES
                  if report["packages"][name] is None]
    if missing_ui:
        print("Install missing interface packages, then run the checks again:")
        print("  " + command_text(
            [python, str(Path(__file__).resolve()), "--install-interface"], system))
    print("For missing/broken scientific packages, follow your laboratory's "
          "HoloPy installation instructions.")
    print("For environment checks and optional interface installation, open {}.".format(
        ROOT / "check installation" / "check_environment.ipynb"))
    recipe = ROOT / "environment-tested.yml"
    if system == "Darwin" and recipe.is_file():
        print("The recorded macOS Intel recipe creates a separate Conda environment:")
        print("  " + command_text(
            ["conda", "env", "create", "--platform", "osx-64", "--file", str(recipe)], system))
        print("Apple Silicon needs Rosetta for this Intel recipe.")
    elif recipe.is_file() and system != "Darwin":
        print("The recorded Intel macOS recipe is not portable to this system.")
    print("HoloPy installation: https://holopy.readthedocs.io/en/master/tutorial/install.html")
    print("Use the reported Python as the notebook kernel. If it is not listed, register it:")
    print("  " + command_text([python, "-m", "ipykernel", "install", "--user",
                                "--name", "hologram-analysis", "--display-name",
                                "HoloPy analysis"], system))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter to inspect (default: current Python)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only list packages; skip imports and functional checks")
    parser.add_argument("--install-interface", action="store_true",
                        help="Install missing ipywidgets, ipympl and ipykernel, then check")
    parser.add_argument("--timeout", type=int, default=45,
                        help="Seconds allowed for each functional check (default: 45)")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    report = inspect_metadata(args.python)
    if "packages" not in report:
        print("[ERROR] Cannot inspect the selected Python:\n" + report["error"])
        print("Check the interpreter path and architecture. On Apple Silicon, "
              "an Intel Python requires Rosetta.")
        return 1
    print("HOLOGRAM ANALYSIS - ENVIRONMENT CHECK")
    print("project: {}".format(ROOT))
    for key in ("executable", "python", "system", "machine"):
        print("{}: {}".format(key, report[key]))
    if args.install_interface:
        if not install_missing_interface(report):
            return 1
        report = inspect_metadata(report["executable"])
        if "packages" not in report:
            print("[ERROR] Cannot recheck Python: " + report["error"])
            return 1
    failed = report["python_info"][:2] < [3, 9]
    if failed:
        print("[ERROR] Python 3.9 or newer is required for this environment recipe.")
    elif report["python_info"][:2] != [3, 9]:
        print("[INFO] Recorded environment: Python 3.9. Other versions need the checks below.")
    print("\nPROJECT FILES")
    for name in ("analyse_hologrammes.ipynb", "hologram_functions.py",
                 "hologram_analysis/__init__.py",
                 "hologram_analysis/optical_cache.py",
                 "hologram_analysis/preparation_cache.py"):
        present = (ROOT / name).is_file()
        failed = failed or not present
        print("[{}] {}".format("OK" if present else "MISSING", name))
    print("\nREQUIRED PACKAGES (different versions are informational, not automatic failures)")
    for name, baseline in PACKAGES.items():
        version = report["packages"][name]
        if version is None:
            failed = True
            print("[MISSING] " + name)
        else:
            note = " (recorded: {})".format(baseline) if baseline and version != baseline else ""
            print("[OK] {} {}{}".format(name, version, note))
    if not args.metadata_only:
        print("\nFUNCTIONAL CHECKS", flush=True)
        for label, source in PROBES.items():
            print("Checking {}...".format(label), flush=True)
            result = functional_check(report["executable"], source, args.timeout)
            if result["ok"]:
                print("[OK] " + label, flush=True)
            else:
                failed = True
                print("[ERROR] {}\n{}".format(label, result["error"]), flush=True)
    print("\nJUPYTER SERVER (informational: may be in a different environment)")
    for name in FRONTENDS:
        print("{}: {}".format(name, report["packages"][name] or "not installed here"))
    print("Notebook 7/Lab needs jupyterlab_widgets in the server environment; "
          "classic Notebook needs widgetsnbextension. VS Code uses its Jupyter extension.")
    print("A Python check cannot verify browser widget rendering or VS Code extensions.")
    if failed:
        print("\n[NOT READY] Missing requirements or failed checks; see details above.")
        print_help_commands(report)
    elif args.metadata_only:
        print("\n[METADATA ONLY] Packages are present. Run without --metadata-only to check them.")
    else:
        print("\n[READY FOR A TRIAL] All automated checks passed. Open the notebook with this Python.")
        print("Verify widget display and analyse one representative image before a full series.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

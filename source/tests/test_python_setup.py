"""CPU checks for folder containment, corrupt-cache preservation and setup reuse."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup_python.ps1"
MANIFEST = ROOT / "assets" / "python-runtime.json"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh.exe")


@unittest.skipUnless(POWERSHELL, "PowerShell is required")
class PythonSetupTests(unittest.TestCase):
    def invoke(self, command: str, directory: Path) -> subprocess.CompletedProcess:
        def literal(value: Path) -> str:
            return "'" + str(value).replace("'", "''") + "'"

        initialization = (
            f". {literal(SCRIPT)} -Root {literal(directory)} "
            f"-ManifestPath {literal(MANIFEST)} -FunctionsOnly; "
        )
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", initialization + command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", check=False,
        )

    def test_manifest_pins_official_signed_payloads(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "3.13.15")
        self.assertEqual(manifest["architecture"], "amd64")
        self.assertEqual({asset["name"] for asset in manifest["assets"]}, {"core.msi", "exe.msi", "lib.msi", "tcltk.msi", "dev.msi"})
        for asset in manifest["assets"]:
            self.assertEqual(asset["url"], manifest["source"] + asset["name"])
            self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(asset["size"], 0)
        self.assertIn("Python Software Foundation", manifest["signer_subject"])
        self.assertRegex(manifest["signer_thumbprint"], r"^[0-9A-F]{40}$")

    @unittest.skipUnless(os.name == "nt", "Windows ensurepip isolation integration")
    def test_ensurepip_ignores_user_pip_environment_and_configuration(self):
        # A fresh private venv avoids changing the Python used by the test runner.
        # Both target/prefix settings and a fake user's pip.ini point outside it.
        original_pip_environment = {key: value for key, value in os.environ.items() if key.upper().startswith("PIP_")}
        with tempfile.TemporaryDirectory(prefix="python-ensurepip-isolation-") as temporary:
            directory = Path(temporary)
            environment = directory / "private-runtime"
            venv.EnvBuilder(with_pip=False, clear=False).create(environment)
            python = environment / "Scripts" / "python.exe"
            outside = directory / "outside-install"
            profile = directory / "fake-user-profile"
            appdata = profile / "AppData" / "Roaming"
            configuration = appdata / "pip" / "pip.ini"
            configuration.parent.mkdir(parents=True)
            configuration.write_text(
                "[global]\n" + "target = " + str(outside) + "\n",
                encoding="utf-8",
            )
            child_environment = os.environ.copy()
            child_environment.update({
                "APPDATA": str(appdata),
                "USERPROFILE": str(profile),
                "PIP_CONFIG_FILE": str(configuration),
                "PIP_TARGET": str(outside),
                "PIP_PREFIX": str(outside),
                "PIP_USER": "1",
                "PIP_REQUIRE_VIRTUALENV": "0",
                "PIP_LOG": str(directory / "outside-pip.log"),
            })
            result = subprocess.run(
                [str(python), "-I", "-B", "-X", "utf8", "-m", "ensurepip", "--upgrade", "--default-pip"],
                env=child_environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(outside.exists(), "PIP_TARGET/PIP_PREFIX/user pip.ini redirected installation")
            self.assertFalse((directory / "outside-pip.log").exists(), "PIP_LOG escaped the private runtime")
            probe = subprocess.run(
                [str(python), "-I", "-B", "-X", "utf8", "-c", "import pathlib,pip,sys; print(pathlib.Path(pip.__file__).resolve()); print(sys.prefix)"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertTrue(Path(probe.stdout.splitlines()[0]).is_relative_to(environment), probe.stdout)
            self.assertEqual(configuration.read_text(encoding="utf-8"), "[global]\ntarget = " + str(outside) + "\n")
        self.assertEqual(
            {key: value for key, value in os.environ.items() if key.upper().startswith("PIP_")},
            original_pip_environment,
        )

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            result = self.invoke("Assert-LocalPath $Root (Join-Path $Root '..\\outside')", Path(temporary))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("escapes", result.stderr)

    def test_parent_folder_is_not_an_allowed_child(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            result = self.invoke("Assert-LocalPath $Root $Root", Path(temporary))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("escapes", result.stderr)

    def test_corrupt_finished_cache_is_retained(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            directory = Path(temporary)
            cache = directory / "downloads" / "python"
            cache.mkdir(parents=True)
            corrupt = cache / "core.msi"
            corrupt.write_bytes(b"corrupt-finished-cache")
            result = self.invoke(
                "$m=Get-Content -LiteralPath $ManifestPath -Raw|ConvertFrom-Json; "
                "Receive-PythonAsset (Join-Path $Root 'downloads\\python') $m.assets[0] $m $true",
                directory,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("size mismatch", result.stderr)
            self.assertEqual(corrupt.read_bytes(), b"corrupt-finished-cache")
            self.assertFalse((directory / "runtime").exists())

    def test_incomplete_offline_cache_fails_before_runtime_creation(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            directory = Path(temporary)
            result = self.invoke("Invoke-PythonSetup $Root $ManifestPath $true", directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Offline cache is incomplete", result.stderr)
            self.assertFalse((directory / "runtime").exists())

    def test_same_size_corrupt_asset_fails_hash_check(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            directory = Path(temporary)
            (directory / "corrupt.msi").write_bytes(b"1234")
            result = self.invoke(
                "Assert-Asset (Join-Path $Root 'corrupt.msi') "
                "@{name='corrupt.msi';size=4;sha256=('0'*64)} @{}",
                directory,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertEqual((directory / "corrupt.msi").read_bytes(), b"1234")

    def test_unknown_existing_runtime_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            directory = Path(temporary)
            runtime = directory / "runtime"
            runtime.mkdir()
            original = runtime / "keep.txt"
            original.write_text("existing runtime", encoding="utf-8")
            result = self.invoke("Invoke-PythonSetup $Root $ManifestPath $true", directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unregistered runtime", result.stderr)
            self.assertEqual(original.read_text(encoding="utf-8"), "existing runtime")

    def test_msi_directory_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            result = self.invoke(
                "Get-MsiTargetDirectory 'a' @{a=@('a','b','one'); b=@('b','a','two')} @{}",
                Path(temporary),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid MSI directory", result.stderr)

    def test_same_root_setup_is_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            result = self.invoke(
                "$data=Join-Path $Root 'data'; New-Item -ItemType Directory $data|Out-Null; "
                "$lock=[IO.File]::Open((Join-Path $data 'setup-python.lock'), 'OpenOrCreate','ReadWrite','None'); "
                "try { Invoke-PythonSetup $Root $ManifestPath $true } finally { $lock.Dispose() }",
                Path(temporary),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another Python setup", result.stderr)

    def test_operation_byte_lock_blocks_python_setup(self):
        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            result = self.invoke(
                "$data=Join-Path $Root 'data'; New-Item -ItemType Directory $data|Out-Null; "
                "$lock=[IO.File]::Open((Join-Path $data 'operation.lock'), 'OpenOrCreate','ReadWrite','ReadWrite'); "
                "$lock.Lock(0,1); "
                "try { Invoke-PythonSetup $Root $ManifestPath $true } finally { $lock.Unlock(0,1);$lock.Dispose() }",
                Path(temporary),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another operation", result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows byte-range lock integration")
    def test_msvcrt_lock_blocks_powershell_setup(self):
        import msvcrt

        with tempfile.TemporaryDirectory(prefix="python-setup-check-") as temporary:
            directory = Path(temporary)
            data = directory / "data"
            data.mkdir()
            with (data / "operation.lock").open("w+b") as handle:
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    result = self.invoke("Invoke-PythonSetup $Root $ManifestPath $true", directory)
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another operation", result.stderr)
            self.assertFalse((directory / "runtime").exists())


if __name__ == "__main__":
    unittest.main()

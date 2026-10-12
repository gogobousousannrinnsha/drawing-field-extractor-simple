"""CPU fixtures for optional backend setup, failed downloads and preserved selection."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("setup_saluki_test_module", ROOT / "scripts/setup_saluki.py")
setup_saluki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_saluki)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SetupSalukiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="saluki-setup-test-", dir=ROOT / "source/tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "assets").mkdir()
        (self.root / "source/simple_drawing").mkdir(parents=True)
        self.cache = self.root / "downloads/saluki"
        self.cache.mkdir(parents=True)
        self.config = self.root / "実行設定.json"
        self.config.write_text('{"input_folder":"入力 with spaces","llm_backend":"gemma","ollama_profile":"gemma3-4b","saluki_thinking":true}', encoding="utf-8")
        self.configuration_sha = setup_saluki.sha(self.config)
        self.models = []
        for name, content in (("model.gguf", b"fixed-model"), ("projector.gguf", b"fixed-projector")):
            self.models.append(self.asset(name, content, "models/saluki/" + name))
        self.executable = b"fake-pinned-server"
        self.dll = b"fake-cuda-backend"
        self.runtime_files = [{"path": "llama-server.exe", "bytes": len(self.executable), "sha256": digest(self.executable)},
                              {"path": "ggml-cuda.dll", "bytes": len(self.dll), "sha256": digest(self.dll)}]
        self.archives = [self.archive("runtime.zip", {"build/bin/llama-server.exe": self.executable, "build/bin/ggml-cuda.dll": self.dll}),
                         self.archive("cuda.zip", {"CUDA/LICENSE-CUDA.txt": b"additional vendor license"})]
        self.notices = [self.asset("model-license.txt", b"Apache model license", "models/saluki/LICENSE-APACHE-2.0.txt"),
                        self.asset("model-notice.txt", b"model notices", "models/saluki/NOTICE.txt"),
                        self.asset("runtime-license.txt", b"MIT runtime license", "runtime/llama-cpp/LICENSE-MIT.txt")]
        self.manifest = {"schema_version": 1, "backend": "saluki", "model": "fixture-model",
                         "source_revision": "f" * 40, "models": self.models, "notices": self.notices,
                         "minimum_initial_free_bytes": 1,
                         "runtime": {"version": "b11429", "source_commit": "d" * 40,
                                     "executable_sha256": digest(self.executable), "archives": self.archives,
                                     "files": self.runtime_files}}
        self.save_manifest()
        assets = {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in self.models}
        definition = "MODEL_PATH = 'models/saluki/model.gguf'\nPROJECTOR_PATH = 'models/saluki/projector.gguf'\nRUNTIME_VERSION = 'b11429'\nASSETS = {MODEL_PATH: " + repr(assets[self.models[0]["path"]]) + ", PROJECTOR_PATH: " + repr(assets[self.models[1]["path"]]) + "}\n"
        (self.root / "source/simple_drawing/saluki.py").write_text(definition, encoding="utf-8")
        self.addCleanup(patch.stopall)
        patch.object(setup_saluki, "EXPECTED_EXECUTABLE_SHA256", digest(self.executable)).start()

    def asset(self, filename, content, destination=None):
        (self.cache / filename).write_bytes(content)
        value = {"filename": filename, "url": "https://huggingface.co/fixture/resolve/" + "f" * 40 + "/" + filename,
                 "bytes": len(content), "sha256": digest(content)}
        if destination:
            value["path"] = destination
        return value

    def archive(self, filename, files):
        path = self.cache / filename
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return {"filename": filename, "url": "https://github.com/fixture/releases/download/b11429/" + filename,
                "bytes": path.stat().st_size, "sha256": setup_saluki.sha(path)}

    def save_manifest(self):
        setup_saluki.write(self.root / "assets/saluki-assets.json", self.manifest)

    def test_nested_archives_prepare_without_changing_model_selection(self):
        result = setup_saluki.setup(self.root, offline=True)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["inference_verified"])
        self.assertFalse(result["selection_changed"])
        self.assertEqual(setup_saluki.sha(self.config), self.configuration_sha)
        runtime = setup_saluki.read(self.root / "runtime/llama-cpp/asset-manifest.json")
        self.assertEqual(runtime["executable_sha256"], digest(self.executable))
        paths = {item["path"] for item in runtime["files"]}
        self.assertIn("legal/cuda.zip/CUDA/LICENSE-CUDA.txt", paths)
        self.assertIn("LICENSE-MIT.txt", paths)
        self.assertEqual((self.root / "runtime/llama-cpp/llama-server.exe").read_bytes(), self.executable)
        self.assertFalse(any((self.root / "temp").iterdir()))

    def test_completed_setup_rechecks_files_and_avoids_download(self):
        first = setup_saluki.setup(self.root, offline=True)
        with patch.object(setup_saluki, "download", side_effect=AssertionError("completed setup downloaded again")):
            again = setup_saluki.setup(self.root, offline=True)
        self.assertEqual(first, again)

    def test_corrupt_installed_model_is_repaired_from_checked_cache(self):
        setup_saluki.setup(self.root, offline=True)
        target = self.root / self.models[0]["path"]
        target.write_bytes(b"broken-model")
        result = setup_saluki.setup(self.root, offline=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(target.read_bytes(), b"fixed-model")
        self.assertEqual(setup_saluki.sha(self.config), self.configuration_sha)

    def test_missing_offline_asset_records_failure_without_complete_receipt(self):
        (self.cache / self.models[0]["filename"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "オフライン"):
            setup_saluki.setup(self.root, offline=True)
        result = setup_saluki.read(self.root / "data/setup-saluki.json")
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.root / "runtime/llama-cpp/asset-manifest.json").exists())
        self.assertEqual(setup_saluki.sha(self.config), self.configuration_sha)

    def test_mismatched_download_content_is_not_copied_or_successful(self):
        with patch.object(setup_saluki, "download", return_value=self.cache / self.models[1]["filename"]):
            with self.assertRaisesRegex(RuntimeError, "SHA・サイズ"):
                setup_saluki.setup(self.root)
        self.assertEqual(setup_saluki.read(self.root / "data/setup-saluki.json")["status"], "failed")
        self.assertFalse((self.root / self.models[0]["path"]).exists())

    def test_duplicate_verified_executable_in_archive_is_rejected(self):
        self.manifest["runtime"]["archives"][0] = self.archive("runtime.zip", {"a/llama-server.exe": self.executable,
                        "b/llama-server.exe": self.executable, "a/ggml-cuda.dll": self.dll})
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "一意"):
            setup_saluki.setup(self.root, offline=True)
        self.assertFalse((self.root / "runtime/llama-cpp/llama-server.exe").exists())

    def test_archive_escape_is_rejected_before_any_installation(self):
        self.manifest["runtime"]["archives"][0] = self.archive("runtime.zip", {"../outside.txt": b"escape"})
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "保存先"):
            setup_saluki.setup(self.root, offline=True)
        self.assertFalse((self.root / "temp/outside.txt").exists())
        self.assertFalse((self.root / self.models[0]["path"]).exists())

    def test_corrupt_completion_receipt_does_not_skip_model_validation(self):
        setup_saluki.setup(self.root, offline=True)
        marker = self.root / "data/setup-saluki.json"
        record = setup_saluki.read(marker)
        record["installed_files"] = []
        setup_saluki.write(marker, record)
        repaired = setup_saluki.setup(self.root, offline=True)
        self.assertGreater(len(repaired["installed_files"]), 0)

    def test_interruption_is_distinguished_and_can_resume(self):
        with patch.object(setup_saluki, "download", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                setup_saluki.setup(self.root, offline=True)
        self.assertEqual(setup_saluki.read(self.root / "data/setup-saluki.json")["status"], "interrupted")
        self.assertEqual(setup_saluki.setup(self.root, offline=True)["status"], "complete")

    def test_low_disk_capacity_fails_before_receipt_and_download(self):
        self.manifest["minimum_initial_free_bytes"] = 100
        self.save_manifest()
        with patch.object(setup_saluki.shutil, "disk_usage") as usage, patch.object(setup_saluki, "download") as download:
            usage.return_value.free = 99
            with self.assertRaisesRegex(RuntimeError, "空き容量"):
                setup_saluki.setup(self.root)
            download.assert_not_called()
        self.assertFalse((self.root / "data/setup-saluki.json").exists())

    def test_model_manifest_must_match_existing_adapter(self):
        self.manifest["models"][0]["sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "アダプター"):
            setup_saluki.setup(self.root, offline=True)

    @unittest.skipUnless(os.name == "nt", "Windows shared operation lock")
    def test_competing_operation_cannot_start_or_change_settings(self):
        (self.root / "data").mkdir()
        with setup_saluki.operation_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "実行中"):
                setup_saluki.setup(self.root, offline=True)
        self.assertEqual(setup_saluki.sha(self.config), self.configuration_sha)
        self.assertFalse((self.root / "data/setup-saluki.json").exists())


class PublishedSalukiManifestTests(unittest.TestCase):
    def test_published_assets_match_adapter_and_pinned_official_sources(self):
        manifest = setup_saluki.read(ROOT / "assets/saluki-assets.json")
        setup_saluki.validate_manifest(ROOT, manifest)
        revision = manifest["source_revision"]
        self.assertEqual(revision, "4f60eba09f4b6db63c72915d328d97e1d93edb8c")
        for asset in manifest["models"]:
            self.assertIn("/resolve/" + revision + "/", asset["url"])
            self.assertEqual(asset["license"], "Apache-2.0")
        for asset in manifest["runtime"]["archives"]:
            self.assertTrue(asset["url"].startswith("https://github.com/ggml-org/llama.cpp/releases/download/b11429/"))
        self.assertFalse(manifest["inference_verified_by_setup"])


if __name__ == "__main__":
    unittest.main()

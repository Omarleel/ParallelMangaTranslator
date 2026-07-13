from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import install_pmt
from parallel_manga_translator.infrastructure.execution_control import (
    ExecutionControl,
    JobCancelledError,
    JobPausedError,
    execution_checkpoint,
)
from parallel_manga_translator.ui.job_manager import JobManager, JobOptions
from parallel_manga_translator.ui.persistent_queue import PersistentJobQueue


class PersistentQueueTests(unittest.TestCase):
    def test_queue_survives_restart_and_recovers_processing_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "queue.sqlite3"
            first = PersistentJobQueue(database)
            first.enqueue("job-a")
            time.sleep(0.002)
            first.enqueue("job-b")
            self.assertEqual(first.pop_next(), "job-a")

            restarted = PersistentJobQueue(database)
            self.assertEqual(restarted.recover_processing(), 1)
            self.assertEqual(restarted.queued_ids(), ["job-a", "job-b"])
            self.assertEqual(restarted.position("job-b"), 2)


class ExecutionControlTests(unittest.TestCase):
    def test_pause_releases_worker_and_cancel_is_cooperative(self):
        paused = threading.Event()
        control = ExecutionControl(on_paused=paused.set)
        control.request_pause()

        with self.assertRaises(JobPausedError):
            control.checkpoint()
        self.assertTrue(paused.is_set())
        self.assertTrue(control.pause_triggered)

        control.resume()
        control.checkpoint()  # ya no está pausado
        control.request_cancel()
        with self.assertRaises(JobCancelledError):
            control.checkpoint()

    def test_external_call_checkpoint_respects_pause_and_cancel(self):
        control = ExecutionControl()
        reservation = control.reserve_external_call(
            kind="llm",
            provider="test",
            estimated_input_tokens=500,
        )
        self.assertEqual(reservation.kind, "llm")
        self.assertEqual(reservation.provider, "test")
        control.commit_external_call(reservation, actual_input_tokens=400)

        control.request_pause()
        with self.assertRaises(JobPausedError):
            control.reserve_external_call(kind="llm", provider="test")


class JobRecoveryAndRetryTests(unittest.TestCase):
    @staticmethod
    def _new_job(manager: JobManager, root: Path, *, retries: int = 1, job_id: str = "job-1"):
        job_root = root / job_id
        input_dir = job_root / "entrada"
        output_dir = job_root / "outputs"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        cv2.imwrite(str(input_dir / "pagina.png"), np.full((24, 24, 3), 255, np.uint8))
        return manager.create_job_from_paths(
            title="Prueba",
            input_dir=input_dir,
            output_dir=output_dir,
            root_dir=job_root,
            options=JobOptions(page_max_retries=retries, retry_backoff_seconds=0),
        )

    def test_job_inpaint_model_overrides_yaml_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            manager = JobManager(jobs_root=jobs_root, start_worker=False)
            job = self._new_job(manager, jobs_root, job_id="job-inpaint")
            job.options.inpaint_model = "aot"

            config = manager._build_config_for_job(job)

            self.assertEqual(config.translation.modelo_inpaint, "aot")

    def test_processing_job_is_requeued_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            first = JobManager(jobs_root=jobs_root, start_worker=False)
            job = self._new_job(first, jobs_root)
            job.status = "processing"
            job.pages[0].status = "processing"
            job.pages[0].attempt_count = 1
            first._save_manifest(job)

            restarted = JobManager(jobs_root=jobs_root, start_worker=False)
            recovered = restarted.get_job(job.job_id)
            self.assertEqual(recovered.status, "queued")
            self.assertEqual(recovered.pages[0].status, "pending")
            self.assertEqual(recovered.pages[0].attempt_count, 1)
            self.assertEqual(recovered.recovery_count, 1)
            self.assertEqual(restarted._queue.position(job.job_id), 1)

    def test_processing_page_with_complete_outputs_is_marked_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            first = JobManager(jobs_root=jobs_root, start_worker=False)
            job = self._new_job(first, jobs_root)
            page = job.pages[0]
            Path(page.clean_path).parent.mkdir(parents=True, exist_ok=True)
            Path(page.translated_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(page.clean_path, np.full((24, 24, 3), 255, np.uint8))
            cv2.imwrite(page.translated_path, np.full((24, 24, 3), 255, np.uint8))
            job.status = "processing"
            page.status = "processing"
            first._save_manifest(job)

            restarted = JobManager(jobs_root=jobs_root, start_worker=False)
            recovered = restarted.get_job(job.job_id)
            self.assertEqual(recovered.status, "ready")
            self.assertEqual(recovered.pages[0].status, "ready")
            self.assertIsNone(restarted._queue.position(job.job_id))

    def test_pausing_active_job_releases_worker_for_next_queued_job(self):
        class PausableProcessor:
            def __init__(self, started: threading.Event):
                self.started = started

            def procesar(self, *args, **kwargs):
                self.started.set()
                while True:
                    execution_checkpoint()
                    time.sleep(0.01)

        class SuccessProcessor:
            def __init__(self, output_name: str):
                self.output_name = output_name

            def procesar(self, input_dir, clean_dir, translation_dir, pages, trans_queue, trad_queue):
                image = np.full((24, 24, 3), 255, np.uint8)
                cv2.imwrite(str(Path(clean_dir) / self.output_name), image)
                cv2.imwrite(str(Path(translation_dir) / self.output_name), image)

        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            manager = JobManager(jobs_root=jobs_root, start_worker=False)
            first = self._new_job(manager, jobs_root, job_id="job-1")
            second = self._new_job(manager, jobs_root, job_id="job-2")
            started = threading.Event()
            processors = [PausableProcessor(started), SuccessProcessor(second.pages[0].output_filename)]
            module = "parallel_manga_translator.ui.job_manager"
            with (
                mock.patch(f"{module}.prepare_runtime"),
                mock.patch(f"{module}.prepare_assets"),
                mock.patch(f"{module}.build_image_processor", side_effect=processors),
            ):
                manager._worker_thread = threading.Thread(target=manager._worker_loop, daemon=True)
                manager._worker_thread.start()
                manager.start_job(first.job_id)
                manager.start_job(second.job_id)
                self.assertTrue(started.wait(timeout=2.0))
                manager.pause_job(first.job_id)

                deadline = time.time() + 4.0
                while time.time() < deadline and manager.get_job(second.job_id).status != "ready":
                    time.sleep(0.02)
                self.assertEqual(manager.get_job(first.job_id).status, "paused")
                self.assertEqual(manager.get_job(second.job_id).status, "ready")
                manager.shutdown()

    def test_page_is_retried_and_succeeds_after_transient_failure(self):
        class FlakyProcessor:
            def __init__(self, output_name: str):
                self.output_name = output_name
                self.calls = 0

            def procesar(self, input_dir, clean_dir, translation_dir, pages, trans_queue, trad_queue):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("fallo transitorio")
                image = np.full((24, 24, 3), 255, np.uint8)
                cv2.imwrite(str(Path(clean_dir) / self.output_name), image)
                cv2.imwrite(str(Path(translation_dir) / self.output_name), image)

        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            manager = JobManager(jobs_root=jobs_root, start_worker=False)
            job = self._new_job(manager, jobs_root, retries=1)
            processor = FlakyProcessor(job.pages[0].output_filename)
            module = "parallel_manga_translator.ui.job_manager"
            with (
                mock.patch(f"{module}.prepare_runtime"),
                mock.patch(f"{module}.prepare_assets"),
                mock.patch(f"{module}.build_image_processor", return_value=processor),
            ):
                manager._run_job(job.job_id)

            completed = manager.get_job(job.job_id)
            self.assertEqual(processor.calls, 2)
            self.assertEqual(completed.pages[0].attempt_count, 2)
            self.assertEqual(completed.pages[0].status, "ready")
            self.assertEqual(completed.status, "ready")


class InstallerSelectionTests(unittest.TestCase):
    @staticmethod
    def info(name: str | None, cuda: float | None):
        gpus = () if name is None else (install_pmt.NvidiaGpu(name, "test", 8192),)
        return install_pmt.HardwareInfo("Windows", "AMD64", gpus, cuda)

    def test_rtx_5070_selects_cu129(self):
        profile, _ = install_pmt.select_profile(self.info("NVIDIA GeForce RTX 5070", 12.9))
        self.assertEqual(profile, "cu129")

    def test_gtx_1080_ti_selects_cu124(self):
        profile, _ = install_pmt.select_profile(self.info("NVIDIA GeForce GTX 1080 Ti", 12.4))
        self.assertEqual(profile, "cu124")

    def test_blackwell_with_older_driver_downgrades_safely(self):
        profile, notes = install_pmt.select_profile(self.info("NVIDIA GeForce RTX 5070", 12.4))
        self.assertEqual(profile, "cu124")
        self.assertTrue(any("baja a cu124" in note for note in notes))

    def test_older_nvidia_driver_uses_cu118_fallback(self):
        profile, _ = install_pmt.select_profile(self.info("NVIDIA GeForce GTX 1080 Ti", 11.8))
        self.assertEqual(profile, "cu118")

    def test_apple_silicon_selects_mps(self):
        info = install_pmt.HardwareInfo("Darwin", "arm64", (), None, apple_silicon=True)
        profile, _ = install_pmt.select_profile(info)
        self.assertEqual(profile, "mps")

    def test_amd_with_rocm_selects_rocm64(self):
        info = install_pmt.HardwareInfo(
            "Linux", "x86_64", (), None,
            amd_gpus=("AMD Radeon RX",), rocm_available=True, rocm_version=6.4,
        )
        profile, _ = install_pmt.select_profile(info)
        self.assertEqual(profile, "rocm64")

    def test_no_nvidia_uses_cpu(self):
        profile, _ = install_pmt.select_profile(self.info(None, None))
        self.assertEqual(profile, "cpu")


if __name__ == "__main__":
    unittest.main()

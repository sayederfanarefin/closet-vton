import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class ApiConfig:
    base_url: str = "http://localhost:8000"
    username: str = "admin"
    password: str = "changeme"
    poll_interval_sec: float = 1.0
    timeout_sec: float = 15 * 60
    delete_on_finish: bool = True


class ApiRunner:
    def __init__(self, cfg: ApiConfig):
        self.cfg = cfg

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.cfg.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def generate_async(
        self,
        *,
        cloth_path: Path,
        person_path: Path,
        overrides_json: Optional[str] = None,
        attempt_number: int = 1,
    ) -> Dict[str, Any]:
        # Submit job
        with cloth_path.open("rb") as f_cloth, person_path.open("rb") as f_person:
            files = {
                "file50": (cloth_path.name, f_cloth),
                "file59": (person_path.name, f_person),
            }
            data = {"attempt_number": str(attempt_number)}
            if overrides_json:
                data["overrides"] = overrides_json

            r = requests.post(
                f"{self.cfg.base_url}/generate_async",
                auth=(self.cfg.username, self.cfg.password),
                files=files,
                data=data,
                timeout=60,
            )
            r.raise_for_status()
            job = r.json()

        job_id = job.get("job_id")
        if not job_id:
            raise RuntimeError(f"Unexpected /generate_async response: {job}")

        # Poll job
        t0 = time.time()
        while True:
            if time.time() - t0 > self.cfg.timeout_sec:
                raise TimeoutError(f"Job {job_id} timed out after {self.cfg.timeout_sec}s")
            jr = requests.get(
                f"{self.cfg.base_url}/jobs/{job_id}",
                auth=(self.cfg.username, self.cfg.password),
                timeout=30,
            )
            jr.raise_for_status()
            j = jr.json()
            status = j.get("status")
            if status in {"succeeded", "failed"}:
                if self.cfg.delete_on_finish:
                    try:
                        self.delete_job(job_id)
                    except Exception:
                        # Best-effort cleanup; ignore failures so evaluation can proceed.
                        pass
                return j
            time.sleep(self.cfg.poll_interval_sec)

    def delete_job(self, job_id: str) -> None:
        r = requests.delete(
            f"{self.cfg.base_url}/jobs/{job_id}",
            auth=(self.cfg.username, self.cfg.password),
            timeout=30,
        )
        r.raise_for_status()

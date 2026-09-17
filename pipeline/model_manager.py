import logging
import os
import subprocess
import time
import atexit
import httpx

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages LLM and Embedding server lifecycles to avoid VRAM OOM."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.llm_process = None
        self.emb_process = None
        self.active_model = None
        atexit.register(self.stop_all)

    def _stop_process(self, proc, name):
        import signal
        if proc:
            logger.info(f"Stopping {name} server...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            logger.info(f"{name} server stopped.")

    def stop_all(self):
        self._stop_process(self.llm_process, "LLM")
        self._stop_process(self.emb_process, "Embedding")
        self.llm_process = None
        self.emb_process = None
        self.active_model = None

    def _wait_for_port(self, port: int, timeout: int = 30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"http://localhost:{port}/v1/models", timeout=1.0)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def require_llm(self):
        if self.active_model == "llm" and self.llm_process and self.llm_process.poll() is None:
            return
        
        # Stop embedding server to free VRAM
        if self.emb_process:
            self._stop_process(self.emb_process, "Embedding")
            self.emb_process = None

        logger.info("Starting LLM server...")
        self.llm_process = subprocess.Popen(
            [os.path.abspath("llm_server/start_server.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.abspath("llm_server"),
            preexec_fn=os.setsid
        )
        self.active_model = "llm"
        
        if self._wait_for_port(8080, timeout=90):
            logger.info("LLM server is ready.")
        else:
            logger.warning("LLM server did not report ready in time.")

    def require_embedding(self):
        if self.active_model == "embedding" and self.emb_process and self.emb_process.poll() is None:
            return

        # Stop LLM server to free VRAM
        if self.llm_process:
            self._stop_process(self.llm_process, "LLM")
            self.llm_process = None

        logger.info("Starting Embedding server...")
        self.emb_process = subprocess.Popen(
            [os.path.abspath("llm_server/start_embedding_server.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.abspath("llm_server"),
            preexec_fn=os.setsid
        )
        self.active_model = "embedding"
        
        if self._wait_for_port(8081, timeout=60):
            logger.info("Embedding server is ready.")
        else:
            logger.warning("Embedding server did not report ready in time.")

model_manager = ModelManager()

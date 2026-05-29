import threading

# ──── Configuration ────

ENGINE_NNUNET_URL = "http://localhost:8104/nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"

# Global inference lock — only 1 engine can infer at a time
_inference_lock = threading.Lock()

# Timeout for HTTP requests (seconds) — inference can be slow
HTTP_TIMEOUT = 600.0

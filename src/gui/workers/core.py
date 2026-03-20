import threading

# ──── Configuration ────

ENGINE_NNUNET_URL_PRETRAINED = "http://localhost:8101"
ENGINE_NNUNET_URL = "http://localhost:8104/nnUNetTrainerDicewBCELoss_1vs50_150ep__nnUNetPlans__3d_fullres"
ENGINE_AUTOPET_URL = "http://localhost:8102"
ENGINE_TOTALSEG_URL = "http://localhost:8103"

# Global inference lock — only 1 engine can infer at a time
_inference_lock = threading.Lock()

# Timeout for HTTP requests (seconds) — inference can be slow
HTTP_TIMEOUT = 600.0

"""TotalSegmentator engine — settings shared by main.py and engine.py."""

import os

# Port (8104 is the nnU-Net engine)
PORT = int(os.getenv("ENGINE_TOTALSEG_PORT", "8105"))

# TotalSegmentator task served by this engine. Default (non-fast) "total" = 117 classes
# predicted by 5 part models at 1.5 mm: organs, vertebrae, cardiac, muscles, ribs.
TASK = "total"

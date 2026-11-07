"""Build-time step (run by the Dockerfile): bake the TotalSegmentator "total" weights in.

Downloads only the part models the default (non-fast) "total" task runs, turns off
TotalSegmentator's anonymous usage statistics (it would otherwise POST to
stats.totalsegmentator.com after every prediction), and fails the build if a weight
folder is missing or the server's imports break (e.g. a missing system library).
"""

import sys

from totalsegmentator.config import (get_config_key, get_weights_dir, set_config_key,
                                     setup_nnunet, setup_totalseg)
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.map_tasks_config import TASK_CONFIGS

TASK = "total"


def main() -> int:
    setup_nnunet()
    setup_totalseg()
    set_config_key("send_usage_stats", False)
    set_config_key("statistics_disclaimer_shown", True)

    task_ids = TASK_CONFIGS[TASK]["sub_modes"]["default"]["task_id"]
    for tid in task_ids:
        download_pretrained_weights(tid)

    wd = get_weights_dir()
    missing = [t for t in task_ids if not any(wd.glob(f"Dataset{t}_*"))]
    if missing:
        print(f"ERROR: no weights for datasets {missing} in {wd}", file=sys.stderr)
        return 1
    if get_config_key("send_usage_stats") is not False:
        print("ERROR: could not disable usage statistics", file=sys.stderr)
        return 1

    # Same imports as the server, so they fail here rather than on the first request.
    import totalsegmentator.nnunet  # noqa: F401
    import totalsegmentator.python_api  # noqa: F401

    print(f"OK: task '{TASK}' weights {task_ids} in {wd}; usage statistics disabled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

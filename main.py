import sys
import traceback
from pathlib import Path

from ui.main_window import main

if __name__ == "__main__":
    log_path = Path(__file__).parent / "startup.log"
    try:
        sys.exit(main())
    except Exception as exc:
        msg = f"FATAL: {exc}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        try:
            log_path.write_text(msg, encoding="utf-8")
        except OSError:
            pass
        sys.exit(1)

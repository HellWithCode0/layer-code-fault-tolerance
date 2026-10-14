"""Wait for the IBM job, then download, save and analyze the results.
Network errors are retried, so a dropped connection does not stop the watch."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cut_obstruction_hw as hw

job_id = sys.argv[1]
last, st = None, ""
while True:
    try:
        st = str(hw.service().job(job_id).status())
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "network error, retrying:", type(e).__name__, flush=True)
        time.sleep(60)
        continue
    if st != last:
        print(time.strftime("%H:%M:%S"), "IBM", st, flush=True)
        last = st
    if st.endswith(("DONE", "ERROR", "CANCELLED")):
        break
    time.sleep(120)
if st.endswith("DONE"):
    for attempt in range(10):
        try:
            hw.mode_analyze(job_id)      # writes hw_counts.json and hw_results.json
            print("saved hw_counts.json and hw_results.json", flush=True)
            break
        except Exception as e:
            print("download failed, retrying:", type(e).__name__, flush=True)
            time.sleep(60)

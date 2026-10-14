"""Poll an IBM job until it finishes, printing each status change."""
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import cut_obstruction_hw as hw

job = hw.service().job(sys.argv[1])
last = None
while True:
    st = str(job.status())
    if st != last:
        print(time.strftime("%H:%M:%S"), st, flush=True)
        last = st
    if st in ("DONE", "ERROR", "CANCELLED", "JobStatus.DONE", "JobStatus.ERROR", "JobStatus.CANCELLED"):
        break
    time.sleep(30)

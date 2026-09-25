"""Offline synthetic Linux CPU sizing; no quality or service-SLA claim."""
import json
from pathlib import Path
import sys
import time
import resource
import psutil
import torch
from rateloop_evaluator.backends import GLiNERBackend
from rateloop_evaluator.templates import overall_approval

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
p=psutil.Process()
def report(stage, **extra):
    data={"stage":stage,"rssMiB":round(p.memory_info().rss/1048576,1),"peakRssMiB":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),**extra}
    for name in ("memory.current","memory.peak","memory.events"):
        path=Path("/sys/fs/cgroup")/name
        if path.exists(): data[name]=path.read_text().strip()
    print(json.dumps(data),flush=True)

start=time.monotonic();b=GLiNERBackend(sys.argv[1]);b.load();report("loaded",seconds=time.monotonic()-start)
for language in ("en","de"):
    q=[v.model_dump() for v in overall_approval(language).questions]
    low,high=1,512
    while low<high:
        middle=(low+high+1)//2
        if b.count_tokens("customer "*middle,q)<=512: low=middle
        else: high=middle-1
    for text in ("The approved refund arrives within five working days.","customer "*low):
        tokens=b.count_tokens(text,q)
        start=time.monotonic();scores=b.predict(text,q)
        report("inference",language=language,tokens=tokens,seconds=time.monotonic()-start,scores=scores)

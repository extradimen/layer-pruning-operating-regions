"""Phase-local CUDA and Linux process counters; never substitutes for lease costs."""
import os,resource,sys
from pathlib import Path
from cost_recorder import measured

def process_io():
    p=Path('/proc/self/io')
    if not p.exists():return None
    return {k:int(v) for k,v in (line.split(':',1) for line in p.read_text().splitlines())}

class ResourceProbe:
    def begin(self):
        import torch
        self.cuda=torch.cuda.is_available();self.io=process_io()
        if self.cuda:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            self.start=torch.cuda.Event(enable_timing=True);self.end=torch.cuda.Event(enable_timing=True)
            self.start.record()
    def finish(self):
        import torch
        values={};io=process_io()
        if io is not None and self.io is not None:
            for key in ('read_bytes','write_bytes'):
                delta=io[key]-self.io[key]
                if delta<0:raise ValueError('process IO counter reset')
                values[key]=measured(delta,'Linux /proc/self/io process-only delta','bytes')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        values['peak_cpu_rss_bytes']=measured(rss if sys.platform=='darwin' else rss*1024,'getrusage process-lifetime high-water RSS; not phase-local','bytes')
        if self.cuda:
            self.end.record();torch.cuda.synchronize()
            values['gpu_cuda_event_s']=measured(self.start.elapsed_time(self.end)/1000,'synchronized current-stream CUDA event interval; includes stream idle gaps, not kernel-active time','s')
            values['peak_gpu_allocated_bytes']=measured(torch.cuda.max_memory_allocated(),'torch phase-reset max_memory_allocated','bytes')
            values['peak_gpu_reserved_bytes']=measured(torch.cuda.max_memory_reserved(),'torch phase-reset max_memory_reserved','bytes')
        return values

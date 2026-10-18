"""Complete cost-schema events; unavailable values are never inferred as zero.

This primitive measures parent-process CPU/wall. GPU, cgroup, I/O and resource
lease samplers must supply their measurements explicitly before deployment.
"""
from pathlib import Path
from contextlib import contextmanager
import datetime,json,resource,time,uuid
from journal import Journal

SCHEMA=json.loads(Path(__file__).with_name('cost_event.schema.json').read_text())
METRICS=SCHEMA['properties']['metrics']['required']

def unavailable(reason):return {'value':None,'status':'unavailable','instrument':None,'null_reason':reason}
def measured(value,instrument,unit):return {'value':value,'status':'measured','instrument':instrument,'null_reason':None,'unit':unit}

class Costs:
    def __init__(self,journal,run_id,gpu_uuids=(),instrument_factory=None):
        self.journal=journal;self.run_id=run_id;self.gpu_uuids=list(gpu_uuids)
        self.instrument_factory=instrument_factory
    @contextmanager
    def phase(self,work_id,phase,scope,*,point_id=None,shared_consumers=(),dependencies=(),lease=False):
        if phase not in SCHEMA['properties']['phase']['enum'] or scope not in SCHEMA['properties']['scope']['enum']:raise ValueError('cost phase')
        ready_file=self.journal.root/'model_ready.json'
        ready_id=None
        if ready_file.exists():
            import hashlib
            ready_id=hashlib.sha256(ready_file.read_bytes()).hexdigest()
        ledger='lease' if lease else 'postready' if ready_id else 'setup'
        event={'work_id':work_id,'event_id':uuid.uuid4().hex,'run_id':self.run_id,'point_id':point_id,'model_ready_id':ready_id,
            'ledger':ledger,'phase':phase,'scope':scope,'gpu_uuids':self.gpu_uuids,'dependencies':list(dependencies),'shared_consumers':list(shared_consumers),
            'start_monotonic_ns':time.monotonic_ns(),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'completed','metrics':{k:unavailable('NOT_INSTRUMENTED_BY_THIS_PRIMITIVE') for k in METRICS},
            'measurement_boundary_notes':'Parent-process CPU only; wall includes this recorder. CPU children, GPU occupancy and counters require separate verified instruments.',
            'currency':None,'pricing_source':None,'identity':self.journal.identity}
        cpu=resource.getrusage(resource.RUSAGE_SELF)
        instrument=self.instrument_factory() if self.instrument_factory else None
        instrument_started=False
        try:
            if instrument is not None:
                instrument.begin();instrument_started=True
            yield event['metrics']
        except BaseException:
            event['status']='failed';raise
        finally:
            if instrument_started:
                try:event['metrics'].update(instrument.finish())
                except Exception as exc:
                    event['status']='failed'
                    event['measurement_boundary_notes']+=' Instrument finalization failed: '+type(exc).__name__
                    self.journal._commit('instrument-failure-'+event['event_id'],{'work_id':work_id,'error_type':type(exc).__name__})
            end=resource.getrusage(resource.RUSAGE_SELF);event['end_monotonic_ns']=time.monotonic_ns();event['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            event['metrics']['wall_s']=measured((event['end_monotonic_ns']-event['start_monotonic_ns'])/1e9,'time.monotonic_ns','s')
            event['metrics']['cpu_user_s']=measured(end.ru_utime-cpu.ru_utime,'getrusage:RUSAGE_SELF','s')
            event['metrics']['cpu_system_s']=measured(end.ru_stime-cpu.ru_stime,'getrusage:RUSAGE_SELF','s')
            event['metrics']['cpu_core_s']=measured(end.ru_utime-cpu.ru_utime+end.ru_stime-cpu.ru_stime,'getrusage:RUSAGE_SELF','core*s')
            self.journal._commit('cost-'+event['event_id'],event)

def shared_total(events,metric):
    seen={};subtotal=0.;unknown=[]
    for event in events:
        wid=event['work_id']
        if wid in seen:
            if event!=seen[wid]:raise ValueError('conflicting actual work_id events')
            continue
        seen[wid]=event;value=event['metrics'][metric]
        if value['value'] is None:unknown.append(wid)
        else:subtotal+=value['value']
    return {'measured_or_estimated_subtotal':subtotal,'complete_total':None if unknown else subtotal,'unknown_work_ids':unknown,'deduplicated_work_ids':list(seen)}

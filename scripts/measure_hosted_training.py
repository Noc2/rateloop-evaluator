"""Offline synthetic sizing of isolated CPU training and exact model switching.

Run in the reviewed CPU image with --network none, one CPU, an explicit memory
cap, the pinned PUBLIC checkpoint read-only and an ephemeral writable /tmp.
This uses explicit fixed-budget synthetic training to measure resource use and
mechanics. It does not bypass the hosted validation gate or establish quality.
"""
from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
import gc
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import psutil


def report(stage, **extra):
    process=psutil.Process()
    value={"stage":stage,"rssMiB":round(process.memory_info().rss/1048576,1),**extra}
    for name in ("memory.current","memory.peak","memory.events","memory.swap.current","memory.swap.peak","cpu.stat"):
        path=Path('/sys/fs/cgroup')/name
        if path.exists(): value[name]=path.read_text().strip()
    stats=Path('/sys/fs/cgroup/memory.stat')
    if stats.exists():
        counters={key:int(value) for key,value in (line.split() for line in stats.read_text().splitlines())}
        value['memoryBreakdown']={key:counters[key] for key in ('anon','file','inactive_file','active_file','kernel')}
        value['workingSetMiB']=round((int(value['memory.current'])-counters['inactive_file'])/1048576,1)
    print(json.dumps(value),flush=True)


def train_child(model_dir, root, steps, stress, channel):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from cryptography.fernet import Fernet
    from rateloop_evaluator.comparison import compare_snapshot
    from rateloop_evaluator.backends import GLiNERBackend, render_input
    from rateloop_evaluator.datasets import import_dataset
    from rateloop_evaluator.learning import LearningStore
    from rateloop_evaluator.templates import custom_text_evaluation
    from rateloop_evaluator.training import TrainOptions, train_snapshot
    root=Path(root); key=root/'encryption.key'
    key.write_bytes(Fernet.generate_key()); key.chmod(0o600)
    store=LearningStore(root/'learning',key)
    template=custom_text_evaluation('en','Does the document specify a numeric project budget?',
        'States a numeric project budget','Does not state a numeric project budget')
    grant=store.add_grant(workspace_id='synthetic-sizing',rights=['private_training'],
        expires_at=time.time()+7200,authorization_until=time.time()+899,
        evidence='Authored synthetic CPU sizing fixture; no customer examples or quality claim')
    rows=[]
    sizing_backend=GLiNERBackend(model_dir) if stress else None
    for index in range(160):
        positive=index%2==0
        text=(f'Project {index}: The approved project budget is EUR {1000+index*17}.' if positive else
            f'Project {index}: The project budget will be decided at the upcoming meeting. No amount has been set.')
        text+=' '+('The proposal describes milestones, staffing, deadlines, delivery and scope. '*12)
        if stress:
            questions=[q.model_dump() for q in template.questions]
            while sizing_backend.count_tokens(render_input({'text':text+' detail'}),questions)<=512:
                text+=' detail'
        rows.append({'case_id':f'case-{index}','group_id':f'group-{index}','text':text,
            'label':'approved' if positive else 'rejected'})
    if sizing_backend is not None:
        counts=[sizing_backend.count_tokens(render_input({'text':row['text']}),questions) for row in rows]
        report('stress_token_check',minimumTokens=min(counts),maximumTokens=max(counts))
        sizing_backend.unload()
    dataset=import_dataset(store,workspace_id='synthetic-sizing',dataset_id='sizing',template=template,
        content='\n'.join(json.dumps(row) for row in rows).encode(),format='jsonl',provenance='synthetic',
        evidence='Authored synthetic capacity test',model_bundle_id='base',authorized_grant_id=grant['id'])
    snapshot=store.create_snapshot('synthetic-sizing',template.id,template.version,
        dataset_version_ids=[dataset['id']],include_feedback=False)
    load=store.load_snapshot
    def authorized(*args,**kwargs):
        store.renew_authorization(grant['id'],'synthetic-sizing',time.time()+899)
        return load(*args,**kwargs)
    store.load_snapshot=authorized
    started=time.monotonic()
    with open(os.devnull,'w') as sink,redirect_stdout(sink),redirect_stderr(sink):
        trained=train_snapshot(store,snapshot['id'],'synthetic-sizing',model_dir,root/'candidate',bundle_id='candidate',
            options=TrainOptions(device='cpu',epochs=5,max_steps=steps,learning_rate=.0001,
                validation_fraction=0,validation_interval=25,early_stopping_patience=3,min_validation_per_label=5))
        comparison=compare_snapshot(store,snapshot['id'],'synthetic-sizing',{
            'baseline':GLiNERBackend(model_dir),'candidate':GLiNERBackend(trained['modelDir'])})
    channel.send({'candidate':trained['modelDir'],'seconds':round(time.monotonic()-started,2),
        'optimizerSteps':trained['training']['optimizerSteps'],'selection':trained['training']['selection'],
        'trainExamples':trained['training']['trainExamples'],'testGroups':comparison['test_group_count'],
        'syntheticAgreement':{name:value['agreement']['estimate'] for name,value in comparison['models'].items()},
        'qualityClaim':False,'stress512Tokens':stress})
    channel.close()


def main():
    parser=ArgumentParser();parser.add_argument('model_dir');parser.add_argument('--max-steps',type=int,default=200);parser.add_argument('--stress-tokens',action='store_true');parser.add_argument('--idle-seconds',type=int,default=180)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from rateloop_evaluator.worker_runtime import _CheckpointCache
    from rateloop_evaluator.templates import custom_text_seed
    questions=[q.model_dump() for q in custom_text_seed('en').questions]
    cache=_CheckpointCache('cpu')
    cache.invoke('base',args.model_dir,'predict','The project budget is EUR 1500.',questions)
    report('base_ready')
    cache.close();gc.collect()
    report('inference_unloaded')
    context=multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='rateloop-capacity-') as root:
        receive,send=context.Pipe(duplex=False)
        process=context.Process(target=train_child,args=(args.model_dir,root,args.max_steps,args.stress_tokens,send))
        process.start();send.close();result=None;last=time.monotonic()
        while process.is_alive() or receive.poll():
            if receive.poll(1):
                try:result=receive.recv()
                except EOFError:break
            if time.monotonic()-last>=30:
                report('training');last=time.monotonic()
        process.join()
        if process.exitcode!=0 or result is None:raise RuntimeError('Isolated sizing training failed')
        # IDs and per-step private selections stay local; sizing emits counts only.
        selection=result.pop('selection')
        report('training_finished',**{k:v for k,v in result.items() if k!='candidate'},
            selectedStep=selection['selectedStep'],validationExamples=selection['validationExamples'])
        for key,path in (('candidate',result['candidate']),('base',args.model_dir),('candidate',result['candidate'])):
            started=time.monotonic();cache.invoke(key,path,'predict','The project budget is EUR 1500.',questions)
            report('switch_'+key,seconds=round(time.monotonic()-started,2))
        deadline=time.monotonic()+args.idle_seconds
        while time.monotonic()<deadline:
            time.sleep(min(30,max(0,deadline-time.monotonic())))
            report('private_idle')
        cache.close();receive.close();process.close()


if __name__=='__main__':main()

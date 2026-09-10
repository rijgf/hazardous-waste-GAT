"""Independent numeric consistency checks for the final Markdown deliverable."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/ppo-objects-v8'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=read(OUT/'summary.json');audit=read(OUT/'audit_final.json')
    assert audit['status']=='PASS' and summary['audit_complete'] and audit['action_provenance']
    assert audit['audit_source_sha256']==sha(ROOT/'audit_ppo_objects_v8_final.py')
    text=(OUT/'results.md').read_text(encoding='utf8')
    records={x['trial']:x for x in audit['trials']}
    assert set(records)=={r['id'] for r in summary['trials']}
    for r in summary['trials']:
        path=ROOT/r['result_path'];raw=read(path)
        assert sha(path)==r['result_sha256']==records[r['id']]['result_sha256']
        assert records[r['id']]['all_candidate_statuses_and_selection_replayed']
        assert r['attempts']==raw['attempts']==12000
        for key,metric in [('J','weighted_objective'),('C','cost'),('R','risk')]:
            assert r[key]==raw['metrics'][metric]
    for group in summary['groups']:
        runs=[r for r in summary['trials'] if r['group']==group['group']]
        assert len(runs)==group['n']==3
        assert {r['seed'] for r in runs}=={3381415411,3078273277,958147205}
        matches=[line for line in text.splitlines() if line.startswith('| '+group['label']+' |')]
        assert len(matches)==1
        fields=[x.strip() for x in matches[0].strip('|').split('|')]
        assert len(fields)==10
        for key,index,tolerance in [('J',5,5.01e-10),('C',7,.000501),('R',8,.000501),('seconds_contended',9,.00501)]:
            actual=statistics.mean(r[key] for r in runs)
            assert abs(group['mean_'+key]-actual)<1e-12
            assert abs(float(fields[index])-actual)<tolerance
    for model in summary['models']:
        assert sha(ROOT/model['checkpoint'])==model['sha256'] and model['sha256'] in text
        assert model['layers']==1 and model['actions']==36864 and model['optimizer_steps']==1152
    registry_path=ROOT/'configs/ppo_objects_v8_exploratory.json';registry=read(registry_path)
    assert sha(ROOT/registry['checkpoint'])==registry['sha256']
    assert sha(ROOT/registry['test_instance'])==registry['test_instance_sha256']
    assert sha(ROOT/registry['dataset_manifest'])==registry['dataset_manifest_sha256']
    chosen=[r for r in summary['trials'] if r['id'] in registry['evaluation_runs']]
    assert len(chosen)==3 and all(r['all_objects_from_ppo'] for r in chosen)
    assert statistics.mean(r['J'] for r in chosen)==registry['mean_J']
    assert {r['seed'] for r in chosen}==set(registry['seeds'])
    instance=read(ROOT/registry['test_instance'])
    assert [instance['b_C'],instance['b_R']]==[registry['b_C'],registry['b_R']]
    for trial in chosen:
        config=read(OUT/'evaluate'/trial['id']/'config.json');args=config['args']
        assert config['checkpoint']['sha256']==registry['sha256']
        for key,registered in [('budget','candidate_budget'),('width','candidate_pool_width'),
                               ('temperature','temperature'),('horizon','horizon'),('memoize','memoize'),
                               ('masks','masks'),('object_ablation','object_ablation')]:
            assert args[key]==registry[registered]
    training=read(ROOT/registry['training_record']);completion=read(ROOT/registry['training_completed'])
    assert training['args']['seed']==registry['training_seed']
    assert len(training['training_inputs'])==registry['training_instances']==24
    assert completion['optimizer_steps']==registry['optimizer_steps']==1152
    assert completion['checkpoint_sha256']==registry['sha256']
    baseline=ROOT/'output/ppo-objects-v7/train/v7b_capacity_context/large_model.pt'
    assert sha(baseline)=='7d0cf70b185ec544b5bf52677566ff4ce35f208261ab39ed6f2f9187f44acc40'
    previous=read(ROOT/'output/ppo-objects-v7/audit.json')
    for record in previous['trials']:
        folder=ROOT/'output/ppo-objects-v7/evaluate'/record['trial']
        assert sha(folder/'result.json')==record['result_sha256']
        assert sha(folder/'actions.jsonl')==record['actions_sha256']
    result={'status':'PASS','trials':len(records),'models':len(summary['models']),
            'historical_v7_trials_preserved':len(previous['trials']),
            'report_sha256':sha(OUT/'results.md'),'summary_sha256':sha(OUT/'summary.json'),
            'full_audit_sha256':sha(OUT/'audit_final.json'),'audit_source_sha256':sha(Path(__file__)),
            'registry_sha256':sha(registry_path),'registry_matches_training_and_three_runs':True}
    (OUT/'delivery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()

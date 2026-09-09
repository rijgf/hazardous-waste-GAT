"""Independent v9 delivery checks and original full-MILP replay of every saved plan.

Action logs are checked for exact counts and same-state uniqueness. This does
not claim an independent re-execution of every candidate's neural probability.
"""
import argparse
import csv
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
import run_ppo_frontier_v9 as run
from src.pareto_experiment import ParetoArchive,coverage_pair
from src.reproducibility import deserialize_plan,file_sha256,plan_sha256
from src.solution_utils import evaluate_solution,route_plan_to_solution

ROOT,OUT=run.ROOT,run.OUT
read,write=run.read,run.write


def check_log(record):
    path=OUT/record['action_log_path'];assert file_sha256(path)==record['action_log_sha256']
    events=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    candidates=[x for x in events if x['event']=='candidate'];batches=[x for x in events if x['event']=='batch_end']
    actual=record['actual_candidate_attempts'];cap=record['candidate_budget']
    assert actual==len(candidates)<=cap and actual+record['unused_candidate_budget']==cap
    assert [x['attempt'] for x in candidates]==list(range(1,actual+1))
    assert sum(x['batch_attempts'] for x in batches)==actual
    assert all(0<=x['batch_attempts']<=8 for x in batches)
    seen=set();state_key=None
    for item in candidates:
        assert math.isfinite(item['log_probability']) and item['log_probability']<=1e-7
        key=(item['state_id'],min(item['state_age'],1024))
        if key!=state_key:seen=set()
        state_key=key;parts=tuple(item['action'].values());assert parts not in seen;seen.add(parts)
    if actual<cap:
        assert record['stop_reason']=='positive_probability_action_space_exhausted'
        assert batches[-1]['exhausted'] and not batches[-1]['accepted']
    else:assert record['stop_reason'] in ('candidate_budget_reached','positive_probability_action_space_exhausted')
    assert record['ledger_counts']['candidate_attempts']==actual
    assert sum(record['search_counts'].get(op+'_attempts',0) for op in ('drop','relocate','swap','two_opt','facility','add'))==actual
    return {'path':record['action_log_path'],'sha256':file_sha256(path),'attempts':actual,'cap':cap,
            'stop_reason':record['stop_reason'],'same_state_duplicate_actions':0}


def audit(require_report=False):
    executed_source_sha256=file_sha256(Path(__file__))
    protocol=run.load_protocol();protocol_hash=file_sha256(OUT/'protocol.json')
    old=read(run.OLD/'protocol.json');samples=defaultdict(dict);checks=[];logs=[]
    for name,h in protocol['instances'].items():
        assert h==old['instances'][name] and file_sha256(run.OLD/f'instances/{name}.json')==h
    instances={name:read(OUT/f'instances/{name}.json') for name in protocol['instances']}
    params={name:run.checked_instance(data)[0] for name,data in instances.items()}
    model_checks={}
    for name,item in protocol['registry']['models'].items():
        entry=run.model_entry(name);assert item==entry
        cfg=read(ROOT/item['training_record']);completed=read(ROOT/item['training_completed'])
        folder=(ROOT/item['training_record']).parent
        history=read(folder/'history.json');coverage=read(folder/'preference_coverage.json')
        assert len(history)==48 and history[-1]['attempts']==36864
        assert completed['history_sha256']==file_sha256(folder/'history.json')
        assert cfg['training']['optimizer_steps']==completed['optimizer_steps']==1152
        train_hashes={read(ROOT/p)['instance_sha256'] for p in cfg['training_inputs']}
        assert not train_hashes & {x['instance_sha256'] for x in instances.values()}
        model_checks[name]={'checkpoint_sha256':item['sha256'],'training_actions':36864,'optimizer_steps':1152,
                           'training_instances':24,'test_parameter_hash_overlap':0,
                           'coverage_sha256':file_sha256(folder/'preference_coverage.json')}
    def check_plan(name,plan_data,point,identity,preference=(.5,.5)):
        data=instances[name];refs=(data['b_C'],data['b_R']);plan=deserialize_plan(plan_data)
        assert plan_sha256(plan)==identity
        metric=evaluate_solution(params[name],route_plan_to_solution(params[name],plan),preference,objective_refs=refs)
        assert metric['feasible'] and not metric['violations']
        for key in ('cost','risk'):
            assert math.isclose(metric[key],point[key],rel_tol=1e-11,abs_tol=1e-7),(name,identity,key)
        if 'weighted_objective' in point:
            assert math.isclose(metric['weighted_objective'],point['weighted_objective'],rel_tol=1e-10,abs_tol=1e-8)
        samples[name][identity]=(plan,metric)
    expected=[];fronts={}
    for task in protocol['tasks']:
        done=OUT/f'completed/{task["id"]}.json';assert done.exists(),f'Missing {done}'
        assert read(done)['protocol_sha256']==protocol_hash
        for budget in task.get('budgets',[40320]):
            archive_id=f"{task['id']}-B{budget}";path=OUT/f'fronts/{archive_id}.json';expected.append(path)
            front=read(path);fronts[archive_id]=front
            assert front['task']==task and front['budget']==budget
            if task['kind']=='NSGA' and protocol.get('nsga_reuse'):
                run.verify_reused_nsga(protocol,front,path)
            else:
                assert front['protocol_sha256']==protocol_hash
            name=task['instance'];data=instances[name];refs=(data['b_C'],data['b_R'])
            assert front['instance_sha256']==data['instance_sha256'] and (front['b_C'],front['b_R'])==refs
            assert front['reference_plan_sha256']==front['search_initial_plan_sha256']==data['reference_plan_sha256']
            archive=ParetoArchive(refs)
            for point in front['points']:
                stored=read(OUT/point['solution_path'])
                assert stored['instance_sha256']==data['instance_sha256'] and stored['solution_id']==point['solution_id']
                assert (stored['b_C'],stored['b_R'])==refs
                check_plan(name,stored['plan'],point,point['solution_id']);archive.add(point)
            assert len(archive.sorted_points())==len(front['points'])==front['strict_feasible_points']>0
            if task['kind']=='PPO':
                assert front['checkpoint']==protocol['registry']['models'][task['model']]
                scalar=front['final_solutions'];assert len(scalar)==21
                assert [x['preference'] for x in scalar]==protocol['preferences']
                for point in scalar:
                    logs.append(check_log(point))
                    check_plan(name,point['plan'],point['metrics'],point['solution_id'],tuple(point['preference']))
                assert sum(x['actual_candidate_attempts'] for x in scalar)==front['counts']['candidate_attempts']<=40320
            else:assert front['counts']['candidate_attempts']==budget
            checks.append({'archive_id':archive_id,'sha256':file_sha256(path),'points':len(front['points'])})
    assert set(expected)==set((OUT/'fronts').glob('*.json')) and len(expected)==63
    assert len(protocol['tasks'])==60
    coverage_checks=[]
    if (OUT/'tables/coverage_pairs.csv').exists():
        rows=list(csv.DictReader((OUT/'tables/coverage_pairs.csv').open(encoding='utf-8-sig',newline='')))
        assert len(rows)==27
        seen=set()
        for row in rows:
            key=(row['instance'],row['model'],int(row['repeat']),int(row['nsga_budget']))
            assert key not in seen;seen.add(key)
            p=fronts[row['P_archive']];n=fronts[row['N_archive']]
            assert p['task']['instance']==n['task']['instance']==row['instance']
            assert p['task']['model']==row['model'] and p['task']['repeat']==n['task']['repeat']==key[2]
            assert n['budget']==key[3]
            assert int(row['n_P'])==len(p['points']) and int(row['n_N'])==len(n['points'])
            assert (float(row['b_C']),float(row['b_R']))==(p['b_C'],p['b_R']) and float(row['tolerance'])==1e-8
            pp=np.array([[v['cost'],v['risk']] for v in p['points']])
            nn=np.array([[v['cost'],v['risk']] for v in n['points']])
            scale=np.array([p['b_C'],p['b_R']])
            # Independent broadcast calculation, not the report's coverage helper.
            p_count=int(np.any(np.all((pp[:,None,:]-nn[None,:,:])/scale<=1e-8,axis=2),axis=0).sum())
            n_count=int(np.any(np.all((nn[:,None,:]-pp[None,:,:])/scale<=1e-8,axis=2),axis=0).sum())
            assert row['status']=='valid' and int(row['P_covers_N_count'])==p_count and int(row['N_covers_P_count'])==n_count
            assert math.isclose(float(row['P_covers_N']),p_count/len(nn),abs_tol=1e-12)
            assert math.isclose(float(row['N_covers_P']),n_count/len(pp),abs_tol=1e-12)
            coverage_checks.append({'P':row['P_archive'],'N':row['N_archive'],
                                    'P_covers_N':p_count/len(nn),'N_covers_P':n_count/len(pp)})
        low,high=protocol['nsga']['budgets']
        expected_pairs={(scale,model,repeat,budget) for scale in ('Test-1','Test-2','Test-3','Test-4')
                        for model in ('small','large') for repeat in range(3)
                        for budget in ([low,high] if scale=='Test-4' and model=='large' else [low])}
        assert seen==expected_pairs
    for index,preference in enumerate(protocol['small_preferences']):
        for repeat,seed in enumerate(protocol['small_seeds'][index]):
            result=read(OUT/f'small/ppo-p{index}-r{repeat}.json')
            assert result['protocol_sha256']==protocol_hash and result['seed']==seed
            assert result['preference']==preference and result['checkpoint']==protocol['registry']['models']['small']
            logs.append(check_log(result));check_plan('small-fixed',result['plan'],result['metrics'],result['solution_id'],tuple(preference))
    full={}
    from audit_scalar_advantage_v5 import full_model_replay
    for name,plans in sorted(samples.items()):
        full[name]=full_model_replay(params[name],list(plans.values()))
        print(f"FULL MILP {name}: {len(plans)} saved solutions PASS",flush=True)
    import normalize_pareto_milp as normalize
    milp_checks=[]
    for index in range(5):
        raw_path=OUT/f'milp/p{index}.json';clean_path=OUT/f'milp/validated/p{index}.json'
        raw=read(raw_path);clean=read(clean_path)
        assert clean['original_result_sha256']==file_sha256(raw_path)
        assert read(OUT/'milp/protocol.json')['solver_options']['time_limit']==3600.
        assert raw['full_solver_vector'] is not None
        fresh=normalize.normalize_record(raw,params['small-fixed'])
        assert fresh['full_cleaned_solver_vector']==clean['full_cleaned_solver_vector']
        assert fresh['metrics']==clean['metrics'] and fresh['proven_optimal']==clean['proven_optimal']
        milp_checks.append({'preference_index':index,'raw_sha256':file_sha256(raw_path),
                           'validated_sha256':file_sha256(clean_path),'proven_optimal':clean['proven_optimal']})
    report_check=None
    if require_report:
        report=read(OUT/'report_verification.json');paper=ROOT/report['manuscript_path']
        assert report['passed'] and report['protocol_sha256']==protocol_hash
        assert report['manuscript_sha256']==file_sha256(paper)
        assert report['formal_front_records']==63 and report['small_ppo_runs']==15 and report['milp_runs']==5
        assert len(coverage_checks)==27
        assert report['manuscript_path']=='供应链管理写作/数值实验与结果分析_论文稿.md'
        handoff=read(OUT/'milp/report_optimality_verdicts.json')
        assert handoff['source_flags_preserved'] and handoff['additional_solver_calls']==0
        assert handoff['report_source_sha256']==file_sha256(ROOT/'report_ppo_frontier_v9.py')
        assert len(handoff['verdicts'])==5 and handoff['verdicts']==report['milp_optimality_verdicts']
        selected_rows=list(csv.DictReader((OUT/'tables/table4_selected_solutions.csv').open(encoding='utf-8-sig',newline='')))
        assert len(selected_rows)==5
        for index,verdict in enumerate(handoff['verdicts']):
            raw=read(OUT/f'milp/p{index}.json');clean=read(OUT/f'milp/validated/p{index}.json')
            solver=raw['solver']
            assert verdict['preference_index']==index and verdict['preference']==raw['preference']
            assert verdict['original_result_sha256']==file_sha256(OUT/f'milp/p{index}.json')
            assert verdict['normalized_result_sha256']==file_sha256(OUT/f'milp/validated/p{index}.json')
            assert clean['solver']==solver and clean['proven_optimal']==raw['proven_optimal']
            assert clean['original_objective_consistent']==raw['objective_consistent']
            proof=solver['status']==0 and solver['success'] and solver['mip_gap']==0
            proof=proof and math.isclose(solver['fun'],solver['mip_dual_bound'],rel_tol=1e-9,abs_tol=1e-9)
            # Full normalized vectors have already been independently rebuilt above.
            proof=proof and clean['strict_feasible'] and clean['objective_consistent']
            proof=proof and all(0<=clean[key]<=1e-5 for key in ('max_matrix_residual','max_bound_residual','max_integer_residual'))
            proof=proof and all(math.isclose(clean[key],solver['fun'],rel_tol=1e-7,abs_tol=1e-7)
                                 for key in ('original_linear_objective','cleaned_linear_objective'))
            proof=proof and math.isclose(clean['metrics']['weighted_objective'],solver['fun'],rel_tol=1e-7,abs_tol=1e-7)
            assert bool(proof)==verdict['report_proven_optimal']
            assert verdict['reconciled_source_flag']==bool(proof and not raw['proven_optimal'])
            assert verdict['source_flags']=={'raw_proven_optimal':raw['proven_optimal'],
                'raw_objective_consistent':raw['objective_consistent'],'normalized_proven_optimal':clean['proven_optimal'],
                'normalized_objective_consistent':clean['objective_consistent']}
            runs=[read(OUT/f'small/ppo-p{index}-r{repeat}.json') for repeat in range(3)]
            best=min(runs,key=lambda item:(item['metrics']['weighted_objective'],item['repeat']))
            row=selected_rows[index]
            assert int(row['selected_repeat'])==best['repeat'] and row['solution_id']==best['solution_id']
            assert row['preference']==str(tuple(raw['preference']))
            for column,key in [('ppo_cost','cost'),('ppo_risk','risk'),('ppo_J','weighted_objective')]:
                assert math.isclose(float(row[column]),best['metrics'][key],rel_tol=1e-12,abs_tol=1e-9)
            mj=clean['metrics']['weighted_objective']
            assert math.isclose(float(row['milp_J']),mj,abs_tol=1e-12)
            assert math.isclose(float(row['relative_difference_percent']),100*(best['metrics']['weighted_objective']-mj)/mj,abs_tol=1e-10)
            assert math.isclose(float(row['ppo_total_seconds']),sum(r['seconds'] for r in runs),abs_tol=1e-9)
            assert float(row['milp_seconds'])==raw['elapsed_seconds']
            assert row['milp_raw_proven_optimal']==str(raw['proven_optimal'])
            assert row['milp_report_proven_optimal']==str(verdict['report_proven_optimal'])
        report_check={'manuscript_sha256':file_sha256(paper),'report_verification_sha256':file_sha256(OUT/'report_verification.json')}
    assert file_sha256(Path(__file__))==executed_source_sha256,'Audit source changed during execution'
    result={'status':'PASS','protocol_sha256':protocol_hash,'models':model_checks,'fronts':checks,
            'independent_coverage_pairs':coverage_checks,
            'small_runs':15,'milp':milp_checks,'action_logs':logs,'full_milp':full,'report':report_check,
            'independent_neural_probability_replay':False,
            'action_audit_scope':'Recorded counts, finite probabilities and same-input nonrepetition; not independent probability reexecution',
            'audit_dependency_sha256':{name:file_sha256(ROOT/name) for name in
                ('audit_scalar_advantage_v5.py','normalize_pareto_milp.py','run_frontier_v9_milp.py',
                 'report_ppo_frontier_v9.py','src/solution_utils.py','hazardous_waste_model.py')},
            'source_sha256':executed_source_sha256}
    write(OUT/'audit'/('numerics-with-report.json' if require_report else 'numerics.json'),result)
    print(json.dumps({'status':'PASS','fronts':len(checks),'logged_preferences':len(logs),'report_checked':bool(report_check)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-report',action='store_true')
    audit(parser.parse_args().require_report)

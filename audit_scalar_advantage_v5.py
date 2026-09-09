"""Independent checks for the one-change large-case experiment (no searches)."""
from __future__ import annotations
import ast
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import run_scalar_advantage_v5 as run
from hazardous_waste_model import HazardousWasteMILP, pickup_type
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, file_sha256, plan_sha256
from src.solution_utils import route_plan_to_solution, evaluate_solution


def compare_source_semantics():
    before = ast.parse(run.SOURCE_BEFORE.read_text(encoding='utf8'))
    after = ast.parse((run.ROOT/'src/ppo_improver.py').read_text(encoding='utf8'))
    def classes(tree):
        return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    old_classes, new_classes = classes(before), classes(after)
    assert set(old_classes) == set(new_classes)
    for name, old in old_classes.items():
        new = new_classes[name]
        if name != 'PPOImprover':
            assert ast.dump(old) == ast.dump(new), name
            continue
        old_methods = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        new_methods = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert set(new_methods)-set(old_methods) == {'_scalarize_advantages'}
        for method in old_methods:
            if method != 'train':
                assert ast.dump(old_methods[method]) == ast.dump(new_methods[method]), method
        old_train, new_train = old_methods['train'], new_methods['train']
        old_loop = next(n for n in ast.walk(old_train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'iteration')
        new_loop = next(n for n in ast.walk(new_train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'iteration')
        begin = next(i for i,n in enumerate(old_loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='advantages_vector' and isinstance(n.value,ast.BinOp))
        replacement = next(n for n in new_loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='advantages' and isinstance(n.value,ast.Call))
        old_loop.body[begin:begin+3] = [replacement]
        assert ast.dump(old_train) == ast.dump(new_train), 'Other training logic changed'


def full_model_replay(params, samples):
    """Fill auxiliaries and stream every original MILP row without solving.

    Sparse arrays avoid allocating a multi-million-variable model per solution.
    Only row storage and variable indices change; original algebra is executed.
    The arc membership container becomes a set for speed, not a subset of arcs.
    """
    values = {}
    expected_cost, expected_risk = [], []
    for index, (plan, point) in enumerate(samples):
        raw = route_plan_to_solution(params, plan)['raw']
        for (vehicle, period), route in plan.items():
            raw['e', vehicle, period] = 1.
            loads = {waste: 0. for waste in params.waste_types}
            for a,b in zip(route,route[1:]):
                if a in params.pickup_nodes:
                    loads[pickup_type(a)] += raw.get(('q',a,vehicle,period),0.)
                    for waste in params.waste_types:
                        raw['l',a,waste,vehicle,period] = loads[waste]
                for waste in params.waste_types:
                    raw['H',a,b,waste,vehicle,period] = float(raw.get(('F',a,b,waste,vehicle,period),0.)>0.)
                for s1,s2 in combinations(params.waste_types,2):
                    w = raw['H',a,b,s1,vehicle,period]*raw['H',a,b,s2,vehicle,period]
                    raw['W',a,b,s1,s2,vehicle,period] = w
                    raw['G',a,b,s1,s2,vehicle,period] = w*(raw.get(('F',a,b,s1,vehicle,period),0.)+raw.get(('F',a,b,s2,vehicle,period),0.))
        for facility in params.facilities:
            for waste in params.waste_types:
                for period in params.periods:
                    raw['omega',facility,waste,period] = float(raw['BD',facility,waste,period] <= params.processing_capacity[facility,waste,period]*params.technology[facility,waste])
        for key, value in raw.items():
            if value:
                values.setdefault(key,np.zeros(len(samples)))[index] = value
        expected_cost.append(point['cost']); expected_risk.append(point['risk'])

    class Stream(HazardousWasteMILP):
        def _v(self,*key):
            return key
        def _add_var(self,key,lb=0.,ub=np.inf,integer=False,cost=0.):
            self.variable_count += 1
            vector = values.get(key)
            if vector is not None:
                self.bound_residual = max(self.bound_residual,float(max(0.,np.max(lb-vector),np.max(vector-ub))))
                if integer:
                    self.integrality_residual = max(self.integrality_residual,float(np.max(np.abs(vector-np.round(vector)))))
                if cost:
                    if key[0] in ('e','x','p'):
                        self.cost += cost/params.cost_weight*vector
                    else:
                        self.risk += cost/params.risk_weight*vector
            else:
                assert lb <= 0 <= ub
            return key
        def _row(self,coeffs,lb=-np.inf,ub=np.inf):
            self.row_count += 1
            terms = [coefficient*values[key] for key,coefficient in coeffs.items() if key in values]
            if terms:
                activity = sum(terms)
                residual = float(max(0.,np.max(lb-activity),np.max(activity-ub)))
            else:
                residual = max(0.,lb,-ub)
            self.row_residual = max(self.row_residual,residual)
            if residual > 1e-5:
                self.violated_rows += 1
        def _constraints(self,arcs,*args):
            return super()._constraints(set(arcs),*args)

    assert params.cost_weight > 0 and params.risk_weight > 0
    model = Stream(params)
    model.variable_count = model.row_count = model.violated_rows = 0
    model.row_residual = model.bound_residual = model.integrality_residual = 0.
    model.cost = np.zeros(len(samples)); model.risk = np.zeros(len(samples))
    started = time.perf_counter(); model._build()
    assert model.violated_rows == 0
    assert max(model.bound_residual,model.integrality_residual) <= 1e-5
    assert np.allclose(model.cost,expected_cost,rtol=1e-10,atol=1e-7)
    assert np.allclose(model.risk,expected_risk,rtol=1e-10,atol=1e-7)
    return {'solutions': len(samples), 'variables': model.variable_count,'rows':model.row_count,
            'violated_rows':model.violated_rows,'max_row_residual':model.row_residual,
            'max_bound_residual':model.bound_residual,'max_integrality_residual':model.integrality_residual,
            'max_cost_difference':float(np.max(np.abs(model.cost-expected_cost))),
            'max_risk_difference':float(np.max(np.abs(model.risk-expected_risk))),
            'seconds':time.perf_counter()-started}


def audit():
    protocol = run.protocol()
    compare_source_semantics()
    record = run.read(run.OUT/'training_record.json')
    checkpoint = torch.load(run.MODEL,map_location='cpu',weights_only=False)
    assert file_sha256(run.MODEL) == record['checkpoint_sha256']
    assert checkpoint['training_protocol_sha256'] == file_sha256(run.OUT/'protocol.json')
    assert checkpoint['network_config'] == protocol['network'] == record['network']
    assert checkpoint['algorithm_config'] == {'ppo':protocol['algorithm']}
    assert checkpoint['seed'] == protocol['training_seed']
    assert all(float(state['step'])==480 for state in checkpoint['optimizer_state_dict']['state'].values())
    assert all(torch.isfinite(t).all().item() for t in checkpoint['state_dict'].values())
    assert record['instance_episode_counts'] == {item['instance_id']:20 for item in protocol['training_instances']}
    history = run.read(run.OUT/'training_history.json')
    assert all(len(series)==20 and all(math.isfinite(v) for v in series) for series in history.values())
    data = run.read(run.ROOT/protocol['instance_path']); params=params_from_json_data(data['params'])
    refs=(data['b_C'],data['b_R']); samples={}; methods=Counter(); counts=[]
    for repeat in range(3):
        new = run.read(run.OUT/f'fronts/PPO-Test-4-large-r{repeat}-B40320.json')
        assert new['checkpoint']['sha256']==record['checkpoint_sha256']
        assert new['protocol_sha256']==file_sha256(run.OUT/'protocol.json')
        assert new['counts']['candidate_attempts']==40320 and len(new['final_solutions'])==21
        for index,final in enumerate(new['final_solutions']):
            assert final['preference']==protocol['preferences'][index]
            plan=deserialize_plan(final['plan']); assert plan_sha256(plan)==final['solution_id']
            metrics=evaluate_solution(params,route_plan_to_solution(params,plan),tuple(final['preference']),objective_refs=refs)
            assert metrics['feasible']
            for key in ('cost','risk','weighted_objective'):
                assert math.isclose(metrics[key],final['metrics'][key],rel_tol=1e-10,abs_tol=1e-8)
        old = run.read(run.BASE/f'fronts/PPO-Test-4-large-r{repeat}-B40320.json')
        collections=[('new_PPO',run.OUT,new),('old_PPO',run.BASE,old)]
        for budget in protocol['nsga_budgets']:
            nsga=run.read(run.BASE/f'fronts/NSGA-Test-4-r{repeat}-B{budget}.json')
            collections.append(('NSGA',run.BASE,nsga))
            a=np.array([[p['cost']/refs[0],p['risk']/refs[1]] for p in nsga['points']])
            b=np.array([[p['cost']/refs[0],p['risk']/refs[1]] for p in new['points']])
            counts.append({'repeat':repeat,'nsga_budget':budget,
                'N_covers_P':float((a[:,None,:]<=b[None,:,:]+protocol['tolerance']).all(2).any(0).mean()),
                'P_covers_N':float((b[:,None,:]<=a[None,:,:]+protocol['tolerance']).all(2).any(0).mean())})
        for method,directory,front in collections:
            assert front['instance_sha256']==data['instance_sha256']
            assert (front['b_C'],front['b_R'])==refs
            assert front['reference_plan_sha256']==data['reference_plan_sha256']
            coordinates=np.array([[p['cost']/refs[0],p['risk']/refs[1]] for p in front['points']])
            weak=(coordinates[:,None,:]<=coordinates[None,:,:]+protocol['tolerance']).all(2)
            np.fill_diagonal(weak,False)
            assert not weak.any(), 'Duplicate or dominated point inside archive'
            for point in front['points']:
                key=(method,point['solution_id'])
                if key in samples: continue
                saved=run.read(directory/point['solution_path']); plan=deserialize_plan(saved['plan'])
                assert plan_sha256(plan)==point['solution_id']
                samples[key]=(plan,point); methods[method]+=1
    results=run.read(run.OUT/'results.json')
    for actual in counts:
        recorded=next(r for r in results['paired_results'] if r['repeat']==actual['repeat'] and r['nsga_budget']==actual['nsga_budget'])
        assert all(recorded[k]==v for k,v in actual.items())
    matrix=full_model_replay(params,list(samples.values()))
    run.protocol()
    evidence={'status':'PASS','only_actor_advantage_scalarization_changed':True,
        'training_instances':24,'training_interaction_steps':11520,'optimizer_updates':480,
        'new_fronts':3,'new_final_solutions':63,'nsga_searches_started':0,
        'paired_coverage_recomputed':counts,'full_model_checks':matrix,'unique_solutions_by_method':dict(methods),
        'preserved_historical_inputs':len(run.read(run.OUT/'preserved_inputs.json')),
        'protocol_sha256':file_sha256(run.OUT/'protocol.json'),'checkpoint_sha256':file_sha256(run.MODEL),
        'auditor_sha256':file_sha256(Path(__file__)),
        'results_sha256':file_sha256(run.OUT/'results.json'),'report_sha256':file_sha256(run.OUT/'results.md')}
    run.write(run.OUT/'audit.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()

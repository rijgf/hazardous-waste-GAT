"""Independent raw-plan/MILP and manuscript consistency checks for NSGA rerun."""
import argparse
from collections import defaultdict
from datetime import datetime
import math
import re

from run_nsga_red_budget import ROOT, OUT, BASE, PAPER, read, write, verify
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, file_sha256, plan_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution
from audit_scalar_advantage_v5 import full_model_replay


def check_front(points,refs):
    assert len({p['solution_id'] for p in points})==len(points)
    for i,a in enumerate(points):
        assert a['feasible'] and all(math.isfinite(a[k]) for k in ('cost','risk'))
        for b in points[i+1:]:
            ab=all((a[k]-b[k])/ref<=1e-8 for k,ref in zip(('cost','risk'),refs))
            ba=all((b[k]-a[k])/ref<=1e-8 for k,ref in zip(('cost','risk'),refs))
            assert not ab and not ba,'Dominated or duplicate objective in saved front'


def main(require_report=False):
    protocol=verify();samples=defaultdict(dict);hashes={};events=[];fronts={}
    for task in protocol['tasks']:
        meta=read(OUT/f"execution/{task['id']}.json")
        done=OUT/f"completed/{task['id']}.json"
        assert meta['status']=='completed' and meta['task']==task and meta['threads']==1
        assert meta['completed_sha256']==file_sha256(done)
        for path,h in read(done)['files'].items():assert file_sha256(OUT/path)==h
        start,end=map(datetime.fromisoformat,(meta['start_utc'],meta['end_utc']))
        assert start<=end and start.tzinfo and end.tzinfo
        events.extend([(start,1),(end,-1)])
        prior_seconds=-1
        for budget in task['budgets']:
            path=OUT/f"fronts/{task['id']}-B{budget}.json";d=read(path)
            assert d['task']==task and d['status']=='success'
            assert d['protocol_sha256']==file_sha256(OUT/'protocol.json')
            assert d['parameters']=={k:protocol[k] for k in ('population','crossover','mutation','initial_service_probability')}
            assert d['budget']==d['counts']['candidate_attempts']==budget
            assert d['seconds']>prior_seconds;prior_seconds=d['seconds']
            assert len(d['points'])==d['strict_feasible_points']>0
            check_front(d['points'],(d['b_C'],d['b_R']))
            counts=d['counts'];assert counts['candidate_objective_evaluations']+counts['cache_hits']==budget
            hashes[path.relative_to(ROOT).as_posix()]=file_sha256(path)
            fronts[task['instance'],task['repeat'],budget]=d
            instance=read(BASE/f"instances/{task['instance']}.json")
            assert all(d[k]==instance[k] for k in ('instance_sha256','b_C','b_R','reference_plan_sha256','search_initial_plan_sha256'))
            for point in d['points']:
                solution=OUT/point['solution_path'];saved=read(solution)
                assert saved['instance_sha256']==d['instance_sha256']
                if point['solution_id'] in samples[task['instance']]:
                    prior=samples[task['instance']][point['solution_id']][1]
                    assert all(math.isclose(prior[k],point[k],rel_tol=1e-12,abs_tol=1e-10) for k in ('cost','risk'))
                samples[task['instance']][point['solution_id']]=(deserialize_plan(saved['plan']),point)
                hashes[solution.relative_to(ROOT).as_posix()]=file_sha256(solution)
    assert len(fronts)==15
    concurrent=peak=0
    for _,delta in sorted(events): concurrent+=delta;peak=max(peak,concurrent)
    assert peak<=4 and concurrent==0
    ppo={}
    for scale in range(1,5):
        name=f'Test-{scale}'
        for model in ('small','large'):
            for repeat in range(3):
                path=BASE/f'fronts/PPO-{name}-{model}-r{repeat}-B252000.json';d=read(path)
                ppo[name,model,repeat]=d['points']
                check_front(d['points'],(d['b_C'],d['b_R']))
                hashes[path.relative_to(ROOT).as_posix()]=file_sha256(path)
                for point in d['points']:
                    solution=BASE/point['solution_path'];saved=read(solution)
                    if point['solution_id'] in samples[name]:
                        prior=samples[name][point['solution_id']][1]
                        assert all(math.isclose(prior[k],point[k],rel_tol=1e-12,abs_tol=1e-10) for k in ('cost','risk'))
                    samples[name][point['solution_id']]=(deserialize_plan(saved['plan']),point)
                    hashes[solution.relative_to(ROOT).as_posix()]=file_sha256(solution)
    replay={}
    for name,group in samples.items():
        instance=read(BASE/f'instances/{name}.json');params=params_from_json_data(instance['params'])
        refs=(instance['b_C'],instance['b_R'])
        for sid,(plan,point) in group.items():
            assert plan_sha256(plan)==sid
            value=evaluate_solution(params,route_plan_to_solution(params,plan),(.5,.5),refs)
            assert value['feasible'],value['violations']
            assert all(math.isclose(value[k],point[k],rel_tol=1e-11,abs_tol=1e-8) for k in ('cost','risk'))
        print('FULL MILP',name,len(group),flush=True)
        replay[name]=full_model_replay(params,list(group.values()))
    result=dict(status='PASS',new_archives=15,reused_ppo_archives=24,unique_plans=sum(map(len,samples.values())),
                milp=replay,observed_task_concurrency_peak=peak,preserved_files=len(protocol['preserved']),hashes=hashes)
    if require_report:
        from report_nsga_red_budget import ms
        rv=read(OUT/'report_verification.json')
        assert file_sha256(PAPER)==rv['paper_sha256']==file_sha256(OUT/PAPER.name)
        for file,h in rv['figures'].items():assert file_sha256(OUT/'figures'/file)==h
        for file,h in rv['tables'].items():assert file_sha256(OUT/'tables'/file)==h
        assert rv['source_sha256']==file_sha256(ROOT/'report_nsga_red_budget.py')
        table=read(OUT/'results.json');text=PAPER.read_text(encoding='utf8')
        assert len(table['pairs'])==27
        for pair in table['pairs']:
            n=fronts[pair['instance'],pair['repeat'],pair['budget']]
            p=ppo[pair['instance'],pair['model'],pair['repeat']]
            def count(a,b):
                return sum(any((x['cost']-y['cost'])/n['b_C']<=1e-8 and
                               (x['risk']-y['risk'])/n['b_R']<=1e-8 for x in a) for y in b)
            assert pair['covered_P']==count(n['points'],p)
            assert pair['covered_N']==count(p,n['points'])
            assert pair['n_N']==len(n['points']) and pair['n_P']==len(p)
            assert pair['N_to_P']==pair['covered_P']/len(p)
            assert pair['P_to_N']==pair['covered_N']/len(n['points'])
        for model,label in [('small','Train-S'),('large','Train-L')]:
            cells=[]
            for scale in range(1,5):
                rows=[p for p in table['pairs'] if (p['instance'],p['model'],p['budget'])==(f'Test-{scale}',model,40320)]
                cells.extend(ms([p[k] for p in rows]) for k in ('N_to_P','P_to_N'))
            assert '| '+label+' | '+' | '.join(cells)+' |' in text
        cells=[]
        for budget in (40320,120960):
            rows=[p for p in table['pairs'] if (p['instance'],p['model'],p['budget'])==('Test-4','large',budget)]
            cells.extend(ms([p[k] for p in rows]) for k in ('N_to_P','P_to_N'))
        assert '| PPO-L vs NSGA-II | '+' | '.join(cells)+' |' in text
        for row in table['timing']:
            assert f"| {row['method']} | {row['instance']} | {row['seconds']} | {row['points']} | 3/3 | 100% | 100% |" in text
        before=(OUT/'before'/PAPER.name).read_text(encoding='utf8')
        assert re.findall(r'\$\$(.*?)\$\$',before,re.S)==re.findall(r'\$\$(.*?)\$\$',text,re.S)
        for a,b in [('## （二）','## （三）'),('### A.1 ','### A.2 '),('### A.3 ','### A.4 '),('### A.5 ',None)]:
            def segment(s):return s[s.index(a):s.index(b)] if b else s[s.index(a):]
            assert segment(before)==segment(text),a
        width=None
        for line in text.splitlines():
            if line.startswith('|'):
                n=line.count('|');width=n if width is None else width;assert n==width,line
            else:width=None
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):assert (PAPER.parent/link).exists(),link
        selection=read(OUT/'figure_selection.json')
        assert selection['hv_reference']==[1.1,1.1]
        for record in selection['all_hv']:
            r=record['repeat'];method=record['method']
            raw=ppo['Test-4','large',r] if method=='PPO-V9' else fronts['Test-4',r,40320 if method.endswith('B1') else 120960]['points']
            inst=read(BASE/'instances/Test-4.json')
            xy=[(p['cost']/inst['b_C'],p['risk']/inst['b_R']) for p in raw]
            xs=sorted({x for x,y in xy if x<1.1 and y<1.1}|{1.1})
            area=sum((right-left)*(1.1-min(y for x,y in xy if x<=left)) for left,right in zip(xs,xs[1:]))
            assert math.isclose(area,record['hv'],rel_tol=1e-12,abs_tol=1e-12)
        for method,selected in selection['selected'].items():
            records=[r for r in selection['all_hv'] if r['method']==method]
            assert selected['repeat']==max(records,key=lambda r:(r['hv'],-r['repeat']))['repeat']
        for point in selection['coordinates']:
            r=point['repeat']
            source=ppo['Test-4','large',r] if point['method']=='PPO-V9' else fronts['Test-4',r,40320 if point['method'].endswith('B1') else 120960]['points']
            raw=next(p for p in source if p['solution_id']==point['solution_id'])
            assert (point['C'],point['R'])==(raw['cost'],raw['risk'])
        result.update(paper_sha256=file_sha256(PAPER),report_verification_sha256=file_sha256(OUT/'report_verification.json'),
                      coverage_pairs=27,formulas_unchanged=True,unchanged_PPO_MILP_sections=True)
    write(OUT/('audit-with-report.json' if require_report else 'audit.json'),result)
    print('PASS',result['unique_plans'],'plans; report',require_report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-report',action='store_true');a=p.parse_args();main(a.require_report)

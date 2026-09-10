"""Generate the transparent fixed-budget experiment ledger, including failures."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/ppo-objects-v8'
BASELINE=.19692419108728887
NSGA=.2286833693057271
LABELS={
 'mask':'旧 PT，仅补处理能力掩码',
 'mask_progress':'掩码 + 进度 horizon=12000',
 'mask_width4':'掩码 + 每状态 4 候选',
 'mask_width1':'掩码 + 每状态 1 候选',
 'anneal':'旧 PT + 掩码 + 非单调接受',
 'base_T1':'原掩码，全部头温度 1',
 'object_T1':'原掩码，算子温度 2 / 对象温度 1',
 'operator_T3':'原掩码，算子温度 3 / 对象温度 2',
 'independent96':'独立头宽度 96，严格掩码重训',
 'conditional96':'条件化头宽度 96，严格掩码重训',
 'conditional96_T1':'条件化 96，温度 1',
 'conditional64':'条件化头宽度 64，严格掩码重训',
 'independent_unique':'重训独立头 96，完整动作不放回',
 'conditional96_unique':'条件化 96，完整动作不放回',
 'deep64':'条件化 64，连续 1536 步训练',
 'deep64_unique':'连续训练模型，完整动作不放回',
 'deep64_objects_uniform':'连续训练模型，对象头均匀消融',
 'width1_objects_uniform':'每状态 1 候选，对象头均匀消融',
 'prefix64':'条件化 64，补充路段载量及下一访问边',
 'prefix64_unique':'补充路段输入，完整动作不放回',
 'base_memory':'原 v7b，状态不变时不重试动作',
 'conditional_memory':'条件化 96，状态不变时不重试动作',
 'memory_objects_uniform':'状态记忆，对象头均匀消融',
 'memory_all_uniform':'状态记忆，完全无 PPO 消融',
 'critic_mlp':'价值头增加隐藏层 + GELU',
 'critic_mlp_memory':'价值头增加隐藏层 + GELU，配合状态记忆',
}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    groups=defaultdict(list);rows=[]
    incomplete=[p.parent.name for p in sorted((OUT/'evaluate').glob('*/config.json')) if not (p.parent/'result.json').exists()]
    for path in sorted((OUT/'evaluate').glob('*/result.json')):
        r=read(path);config=read(path.parent/'config.json');args=config['args']
        actions=[json.loads(x) for x in (path.parent/'actions.jsonl').read_text(encoding='utf8').splitlines()]
        width=args['width'];duplicates=sum(len(block)-len({tuple(x['action'].values()) for x in block})
            for i in range(0,len(actions),width) for block in [actions[i:i+width]])
        assert r['attempts']==len(actions)==12000
        key=re.sub(r'_r\d+(?:_retry)?$','',path.parent.name)
        row={'id':path.parent.name,'group':key,'seed':args['seed'],'J':r['metrics']['weighted_objective'],
             'C':r['metrics']['cost'],'R':r['metrics']['risk'],'seconds_contended':r['seconds'],
             'duplicates':duplicates,'infeasible':r['counts'].get('infeasible',0),'attempts':r['attempts'],
             'all_objects_from_ppo':r['object_heads_used'],
             'result_path':str(path.relative_to(ROOT)).replace('\\','/'),'result_sha256':sha(path)}
        rows.append(row);groups[key].append(row)
    summary=[]
    for key,items in groups.items():
        items.sort(key=lambda x:[3381415411,3078273277,958147205].index(x['seed']))
        summary.append({'group':key,'label':LABELS.get(key,key),'n':len(items),
            'all_objects_from_ppo':all(x['all_objects_from_ppo'] for x in items),
            **{f'mean_{field}':statistics.mean(x[field] for x in items) for field in ['J','C','R','seconds_contended','duplicates','infeasible']},
            'J_values':[x['J'] for x in items]})
    summary.sort(key=lambda x:x['mean_J'])
    models=[]
    for path in sorted((OUT/'train').glob('*/completed.json')):
        config=read(path.parent/'config.json');completion=read(path);history=read(path.parent/'history.json')
        assert completion['checkpoint_sha256']==sha(path.parent/'large_model.pt')
        models.append({'id':path.parent.name,'sha256':completion['checkpoint_sha256'],
            'dimension':config['args']['dimension'],'layers':config['args']['layers'],
            'actions':history[-1]['attempts'],'optimizer_steps':completion['optimizer_steps'],
            'seconds_contended':completion['seconds'],'checkpoint':str((path.parent/'large_model.pt').relative_to(ROOT)).replace('\\','/')})
    audit=read(OUT/'audit_final.json') if (OUT/'audit_final.json').exists() else None
    audited={x['trial']:x['result_sha256'] for x in audit['trials']} if audit else {}
    complete_audit=bool(audit and audit['status']=='PASS' and set(audited)=={r['id'] for r in rows}
                        and all(audited[r['id']]==r['result_sha256'] for r in rows)
                        and audit['audit_source_sha256']==sha(ROOT/'audit_ppo_objects_v8_final.py'))
    data={'exploratory':True,'baseline_J':BASELINE,'NSGA_J':NSGA,'trials':rows,'groups':summary,
          'models':models,'total_search_candidates':len(rows)*12000,'audit_complete':complete_audit,'incomplete_runs':incomplete}
    (OUT/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    lookup={g['group']:g for g in summary}
    chosen=lookup['conditional_memory']
    lines=['# PPO 固定预算深化探索结果','',
        '范围：固定原 Test-4、风险偏好 0.5/0.5，每次从公共参考解开始。J 越小越好；b_C=51700.426665000006，b_R=15393.350068342135。',
        '', '## 本轮结论与保留版本','',
        f"本轮最好组合为：**96 维条件化对象头 + 处理能力可行性掩码 + 相同状态不重复尝试动作**。三次平均 J={chosen['mean_J']:.9f}，较原 v7b 改善 {(BASELINE-chosen['mean_J'])/BASELINE*100:.2f}%，比最强已有 NSGA 标量点低 {(NSGA-chosen['mean_J'])/NSGA*100:.2f}%。三个 PPO 结果都低于该 NSGA 标量点，但相对旧 PPO 的逐种子比较是两胜一负。",
        '', '保留 `output/ppo-objects-v8/train/conditional96/large_model.pt`，求解入口为 `run_ppo_objects_v8c.py`，启用 `--memoize`。该 PT 的价值头仍是单层线性输出；新增价值隐藏层对照没有替代它。登记见 `configs/ppo_objects_v8_exploratory.json`。这只是当前案例上的探索版本，不自动替代正式 v4 冻结模型。',
        '', '算子与三个对象仍由 PPO 输出概率选择；三个对象头各有独立的 512 类参数，但后续评分可读取前序已选对象。没有增加算子、启发式插入、NSGA 热启动或候选预算，也没有叠加 Transformer。状态记忆是在输入完全不变时，直接从 PPO 剩余联合概率中抽取未试动作，不是按目标值生成动作。',
        '', f"同样启用状态记忆，对象头改为均匀抽样时平均 J={lookup['memory_objects_uniform']['mean_J']:.9f}，完全不用 PPO 时为 {lookup['memory_all_uniform']['mean_J']:.9f}；完整网络分别低 {(1-chosen['mean_J']/lookup['memory_objects_uniform']['mean_J'])*100:.2f}% 和 {(1-chosen['mean_J']/lookup['memory_all_uniform']['mean_J'])*100:.2f}%。这支持学习决策在此配置中有贡献，但不是统计显著性结论。",
        '', f"平均成本 C={chosen['mean_C']:.3f}、风险 R={chosen['mean_R']:.3f}；用于比较的最强 NSGA 标量点 C=21166.016788、R=738.409854。新 PPO 成本更低但风险更高，因此胜出仅指固定 0.5/0.5 下的加权 J，不是同时支配成本、风险或完整前沿。",
        '',f'每次预算严格为 **12,000 个候选**；本轮调试共完成 {len(rows)} 次、累计 {len(rows)*12000:,} 个候选。每组按预定三个种子求平均，没有把跨次最优解合并成一次求解。',
        '',f'原成功 v7b：平均 J={BASELINE:.10f}；最强已有 NSGA 标量点 J={NSGA:.10f}。NSGA 来自原有更高预算档案，不是等预算或等时间比较。',
        '', '对象头均匀消融仅用于判断学习贡献，不满足“PPO 选择全部对象”的要求，不能当作最终方法。',
        '', '## 所有组（包含失败，按平均 J 排序）','',
        '| 方法/设置 | 次数 | 第 1 次 J | 第 2 次 J | 第 3 次 J | 平均 J | 相对原 PPO 改善 | 平均 C | 平均 R | 平均时间/s* |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary:
        vals=[f'{v:.9f}' for v in g['J_values']]+['']*(3-g['n'])
        lines.append('| '+ ' | '.join([g['label'],str(g['n']),*vals,f"{g['mean_J']:.9f}",
            f"{(BASELINE-g['mean_J'])/BASELINE*100:+.3f}%",f"{g['mean_C']:.3f}",f"{g['mean_R']:.3f}",f"{g['mean_seconds_contended']:.2f}"])+ ' |')
    if incomplete:
        lines += ['', '另保留未完成运行目录：'+', '.join(f'`{name}`' for name in incomplete)+'。这些目录不计入均值；完整试验计数不代表含失败启动在内的全部计算开销。`critic_mlp_r2` 在首批动作日志写出前退出且没有错误文本，原目录保留，使用同种子、同参考解在 `critic_mlp_r2_retry` 重新完整运行，未续接或挑选残留方案；详情见 `runtime_events.json`。']
    lines += ['', '*上述时间为实际记录的墙钟时间；部分任务并发运行，存在资源争用，不能作为独占速度比较。所有搜索配置都不增加候选预算，宽度 1 及条件化不放回会增加神经网络调用次数。',
              '', '## 新训练模型（不覆盖或自动替代旧冻结模型）','',
              '| 模型 | 宽度 | Transformer 层数 | 训练动作 | Adam 更新 | PT SHA256 |',
              '|---|---:|---:|---:|---:|---|']
    for m in models:lines.append(f"| {m['id']} | {m['dimension']} | {m['layers']} | {m['actions']} | {m['optimizer_steps']} | {m['sha256']} |")
    lines += ['', '## 价值头核验与单因素对照（用户追加）','',
              '当前 v7/v8 的共享聚合上下文已经是 `Linear(101,96) -> GELU`，之后才接 `Linear(96,1)` 价值输出。旧 v4 的价值头则为 `Linear(fused_dim,2)`。Linear 就是全连接层；最终输出线性，不意味着整个价值函数只是一条线性映射。',
              '本次仅给价值头增加隐藏层：`Linear(96,96) -> GELU -> Linear(96,1)`，增加 9,312 个参数。相同种子下所有非价值参数的初始值逐元素一致，最后输出仍保留正负及不受限值域；没有给输出加 ReLU/Sigmoid，没有叠加 Transformer。',
              '价值头只在训练中影响价值估计、优势及共享编码器梯度；求解择优仍使用真实 C/R/J，不用预测价值替代目标。新旧模型都从头训练，训练动作和更新次数不变。',
              '对照结果不支持“最后一层没有激活导致当前表现不佳”这一解释：普通搜索只小幅改善，而已优化的状态记忆配置反而退步。更大的价值头改变训练期优势和共享梯度，并不保证最终搜索更好；当前证据也不足以进一步把退步归因为过拟合或梯度冲突。未中断其他已启动试验，新增模型及全部有利/不利结果均保留。']
    for new,old,setting in [('critic_mlp','conditional96','普通批内有放回搜索'),('critic_mlp_memory','conditional_memory','相同状态不重试搜索')]:
        if new in lookup and lookup[new]['n']==3:
            before,after=lookup[old]['mean_J'],lookup[new]['mean_J']
            lines += ['',f'{setting}：旧线性价值头平均 J={before:.9f}，新增隐藏层平均 J={after:.9f}，相对改善 {(before-after)/before*100:+.3f}%。这是同一训练种子下的探索性对照，不能单凭一轮结果证明价值头容量是普遍根因。']
        else:lines += ['',f'{setting}：新增价值头三次对照尚未全部完成。']
    lines += ['', '训练均从头开始，仅使用原 v4 的 24 个 TRAIN 大实例及各自固定 b。每个模型预定最终检查点，未用测试或 NSGA 方案训练。模型/超参数版本选择仍受固定测试反馈影响，因此是探索性案例结果。',
              '', '## 已证实的问题与反例','',
              '- 末期处理能力漏掩码：原最终解的 76 个允许删除中，66 个必然导致设施末期库存不清空。新纯可行性掩码已通过逐项独立评估，原输入编码保持不变。但仅修此点并未改善均值，不能把修复正确性等同于优化效果。',
              '- 后续对象头原始评分不感知前序对象。条件化版本加入前序对象及相应原始边输入，三个头各自仍有 512 个输出和独立参数；采样与 PPO 更新的联合概率一致。表达能力修复不保证在同训练预算内变强。',
              '- 重训后的动作分布过于集中，且有放回抽样会重复。第 0 次原 v7b 只有 50/12000 重复元组，独立头重训为 3008/12000，条件化 96 为 1834/12000。不放回版本直接从 PPO 联合概率中扣除已抽完整元组质量，无额外重抽、评价或启发式选对象。',
              '- 上述重复数只统计每个 8 候选批内。进一步按相同计划及相同截断进度跨批计数，条件化 96 第 0 次重复达到 4709/12000，旧 v7b 为 516/12000；说明仅批内去重还遗漏了后期大量重复。条件化 96 的均值由普通搜索 0.205932859，经批内去重 0.201073814，再到同状态记忆 0.187767994。原 v7b 单独加状态记忆反而为 0.197922905，不能把组合收益全部归给一个改动。',
              '- horizon 延长、降低全部头温度、缩窄网络、允许非单调接受等对照也全部保留，不能只展示最优一次或一组有利设置。',
              '', '## 核验与局限','',
              f"完整终验状态：{'PASS，覆盖全部当前结果' if complete_audit else '尚未覆盖全部当前结果，不能据此宣称终验完成'}。",
              '入口：`py -B audit_ppo_objects_v8_final.py --workers 4`。核验每次输入/模型/源代码哈希、全部动作概率、每个候选的合法性与接受状态、批内择优、完整原 MILP 行/变量界/整数性、C/R 重算，以及 3325 个历史输入文件不变。只允许复用实现、输入、动作及结果哈希完全一致的已完成完整核验；全部最终解仍重新通过原 MILP。',
              '复现某次试验：查看对应 `evaluate/<id>/config.json` 的完整参数及 `source/` 快照，用新 id 运行所记录入口，不能覆盖旧目录。',
              '本轮是固定单实例、固定偏好的调试，不是无偏泛化验证。三次均值优势不能自动称为统计显著，也不能证明已达到全局最优或算法极限。正式文稿、Word、NSGA 前沿、旧 PT 未替换，未自动提交或推送 GitHub。',
              '诊断采用 diagnosing-bugs 的可复现反例、单因素对照和回归核验流程；详细约束见 `docs/ppo_objects_v8_protocol.md`。','']
    (OUT/'results.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'trials':len(rows),'models':len(models),'best':summary[0] if summary else None,'audit_complete':complete_audit},ensure_ascii=False))


if __name__=='__main__':main()

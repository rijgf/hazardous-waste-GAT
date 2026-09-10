"""Only vary the Bernoulli probability in the original random initialization."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import run_nsga_parameter_probe as previous
import run_nsga_spread_probe as spread
from src import pareto_experiment as core
from src.reproducibility import file_sha256

CONFIGS = {
    'init80': (100,.9,.25,.8),
    'init95': (100,.9,.25,.95),
    'init80_m50': (100,.9,.5,.8),
    'init80_n200': (200,.9,.25,.8),
    'init65': (100,.9,.25,.65),
    'init95_m05': (100,.9,.05,.95),
}


@contextmanager
def probability(value):
    assert 0 <= value <= 1
    original=core.MultiPeriodEncoding
    class Encoding(original):
        def random_genome(self,rng):
            return [(int(period == self.params.periods[-1] or rng.random() < value),
                     rng.randrange(len(self.params.vehicles)), rng.randrange(len(self.params.facilities)), rng.random())
                    for _, period in self.slots]
    core.MultiPeriodEncoding=Encoding
    try:
        yield
    finally:
        core.MultiPeriodEncoding=original


def run(name,seed,budget):
    spread.CONFIGS[name]=CONFIGS[name][:3]
    spread.configure()
    manifest=spread.OUT/f'{name}-s{seed}-B{budget}.initialization.json'
    assert not manifest.exists(),'No overwrite'
    # Store outside result glob: probability metadata must never be mistaken for a front.
    manifest=spread.OUT/'initialization'/manifest.name
    assert not manifest.exists(),'No overwrite'
    previous.write(manifest,dict(probability=CONFIGS[name][3],params=CONFIGS[name][:3],seed=seed,budget=budget,
                                 source_sha256=file_sha256(Path(__file__))))
    with probability(CONFIGS[name][3]):
        spread.run(name,seed,budget)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('names',nargs='+',choices=list(CONFIGS))
    p.add_argument('--seed',type=int,default=20260910)
    p.add_argument('--budget',type=int,default=3000)
    a=p.parse_args()
    for name in a.names: run(name,a.seed,a.budget)

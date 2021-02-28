from hpbandster.optimizers import BOHB, randomsearch
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from IOHexperimenter import IOH_function, IOH_logger
import ConfigSpace as CS
import numpy as np
import numpy.random as npr
import pickle
import os

class IOHworker(Worker):
    def __init__(self, fid, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.problem = IOH_function(fid=fid, dim=dim, iid=1, suite='PBO')
        self.problem.add_logger(IOH_logger(
            location='experiments',
            foldername='fid{:02d}-dim{:03d}'.format(fid, dim),
            name='BOHB',
        ))
    
    def compute(self, config, budget, **kwargs):
        generation = int(budget)
        config = [value for key, value in sorted(config.items())]
		## Evaluating the solution using fitness function
        result = np.array([self.problem(config) for _ in range(10)])
        real = np.mean(result)
        result += npr.normal(0,100, 10)

        return ({'loss': real*-1, 
                 'info': {
					  'fitness': real,
                      'std': np.std(result),
                      'generation': generation,
                  }
                })
        
    def config_space(self):
        cs = CS.ConfigurationSpace()
        for i in range(self.dim):
            cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
                name='{:03d}'.format(i),
                lower=self.problem.lowerbound[i],
                upper=self.problem.upperbound[i],
            ))
        return cs



NWorker = 10
dir = 'EA_'
NS = hpns.NameServer(run_id='EA', host='127.0.0.2', port=None)
NS.start()
min_budget=1
max_budget=50000
functionID = 23
dims = [16, 25, 36, 49, 64]


for dimensions in dims:
    sdir = str(dir+str(dimensions))
    os.mkdir(sdir)
    workers=[]
    for i in range(NWorker):
        w = IOHworker(functionID, dimensions, nameserver='127.0.0.2', run_id='EA', id=i)
        w.run(background=True)
        workers.append(w)
    result_logger = hpres.json_result_logger(directory=sdir, overwrite=False)

    bohb = BOHB(configspace=w.config_space(), run_id='EA', nameserver='127.0.0.2', result_logger=result_logger, min_budget=min_budget, max_budget= max_budget)
    res = bohb.run(n_iterations=1, min_n_workers=NWorker)
    # store results
    with open(os.path.join(sdir, str('results_EA_dim_'+str(dimensions)+'.pkl')), 'wb') as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    w.problem.reset()
    w.problem.clear_logger()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    all_runs = res.get_all_runs()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
    print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))
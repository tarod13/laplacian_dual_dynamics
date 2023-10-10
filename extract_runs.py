import pandas as pd
import wandb
from tqdm import tqdm

d = 11
n_samples = 50
cosine_keys = [f'cosine_similarity_{i}' for i in range(0,d)]
inner_keys = [f'inner({i},{i})' for i in range(0,d)]
beta_keys = [f'beta({i},{i})' for i in range(0,d)]
barrier_keys = ['barrier_coeff']
keys = ['grad_step', 'cosine_similarity'] + cosine_keys + inner_keys + beta_keys + barrier_keys

api = wandb.Api(timeout=600)

project_name = "tarod13/laplacian-encoder"
exp_name = 'EXP1-M'
runs = api.runs(project_name, filters={"config.exp_label": exp_name}, )

history_list, config_list, name_list = [], [], []
for run in tqdm(runs):
    history_list.append(run.history(samples=n_samples, keys=keys))

    config_list.append(
        {k: v for k,v in run.config.items()
        if not k.startswith('_')})

    name_list.append(run.name)
    
runs_df = pd.DataFrame({
    "history": history_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_pickle(f'./results/curves/{exp_name}.pkl')
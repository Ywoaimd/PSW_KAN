"""Explicit-seed ablation runner. Dry-run by default; --execute starts training."""
import argparse, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent

def prepare(opt):
    repo=Path(opt.repo).resolve()
    rows=json.loads((HERE/'configurations.json').read_text(encoding='utf-8'))
    row=next(r for r in rows if r['dataset']==opt.dataset and r['horizon']==opt.horizon)
    args=dict(row['args'])
    args['root_path']=str((repo/args['root_path']).resolve())
    args.update(use_multi_gpu=False,use_gpu=True,gpu=opt.gpu,num_workers=0,output_attention=False)
    args['ablation_patch']=opt.variant!='no_patch'
    args['ablation_reswave']=opt.variant!='no_reswave'
    if opt.variant=='no_se':args.update(use_se=False,se_alpha=0.0)
    args['model']='TimeKAN'
    run=Path(opt.output).resolve()/f'{opt.dataset}_{opt.horizon}_{opt.variant}_seed{opt.seed}'
    args['checkpoints']=str(run/'checkpoints')
    return repo,SimpleNamespace(**args),run

def load_model(variant):
    path=HERE/'TimeKAN_baseline.py' if variant=='baseline' else HERE/'PSWKAN_variants.py'
    spec=importlib.util.spec_from_file_location('selected_model',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True)
    ap.add_argument('--dataset',choices=['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity'],required=True)
    ap.add_argument('--horizon',type=int,choices=[96,192,336,720],required=True)
    ap.add_argument('--variant',choices=['baseline','full','no_patch','no_reswave','no_se'],required=True)
    ap.add_argument('--seed',type=int,choices=[42,123,456],required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--output',default='ablation_runs')
    ap.add_argument('--execute',action='store_true')
    opt=ap.parse_args();repo,args,run=prepare(opt)
    record={'variant':opt.variant,'seed':opt.seed,'args':vars(args),'output':str(run)}
    print(json.dumps(record,indent=2))
    if not opt.execute:return
    if run.exists():raise FileExistsError(f'Refusing to overwrite {run}')
    sys.path.insert(0,str(repo))
    import random,numpy as np,torch
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    module=load_model(opt.variant)
    random.seed(opt.seed);np.random.seed(opt.seed);torch.manual_seed(opt.seed);torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    class Experiment(Exp_Long_Term_Forecast):
        def _build_model(self):
            return module.Model(self.args).float()
    run.mkdir(parents=True);(run/'configuration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    os.chdir(run)
    # Existing experiment writes relative result/test folders; cwd isolates every run.
    experiment=Experiment(args);setting=run.name
    experiment.train(setting);experiment.test(setting)
if __name__=='__main__':main()

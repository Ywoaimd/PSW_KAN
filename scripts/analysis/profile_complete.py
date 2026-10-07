"""Forward arithmetic estimate, MAC=2; FFT and transcendental conventions explicit."""
import argparse,json,sys,math
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from run_ablation import prepare,load_model
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class Count(TorchDispatchMode):
 def __init__(self):super().__init__();self.count=Counter();self.unknown=Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  y=func(*args,**(kwargs or {}));name=func._schema.name.split('::')[-1];out=y[0] if isinstance(y,tuple) else y;n=out.numel() if isinstance(out,torch.Tensor) else 0;c=0
  if name in ['mm','bmm']:c=2*n*args[0].shape[-1]
  elif name=='addmm':c=2*n*args[1].shape[-1]+n
  elif name=='convolution':c=2*n*args[1].numel()/args[1].shape[0]+(n if args[2] is not None else 0)
  elif name in ['add','add_','sub','sub_','mul','mul_','div','div_','pow','sqrt','rsqrt','cos','acos','tanh','exp','neg','abs','relu','relu_','clamp']:c=n
  elif name=='sigmoid':c=4*n
  elif name=='gelu':c=8*n
  elif name in ['mean','sum']:c=args[0].numel()
  elif name=='var':c=3*args[0].numel()+n
  elif name=='native_layer_norm':c=5*args[0].numel()+2*n
  elif name=='_softmax':c=5*n
  elif name in ['avg_pool2d','avg_pool1d']:c=n*math.prod(args[1])
  elif name in ['_fft_r2c','_fft_c2r']:
   dims=args[1];shape=args[0].shape if name=='_fft_r2c' else out.shape;length=math.prod(shape[d] for d in dims);c=2.5*math.prod(shape)*math.log2(length)
  elif name not in 'view _unsafe_view reshape _reshape_alias transpose t permute squeeze unsqueeze expand as_strided slice select detach clone copy_ _to_copy empty empty_like empty_strided zeros zeros_like new_empty new_zeros zero_ fill_ cat stack split split_with_sizes unbind unfold repeat alias lift_fresh resolve_conj replication_pad1d constant_pad_nd resize_'.split():self.unknown[name]+=1
  if isinstance(out,torch.Tensor) and out.is_complex() and name in ['add','add_','sub','sub_','mul','mul_','div','div_']:
   c *= 6 if name in ['mul','mul_'] and all(isinstance(a,torch.Tensor) and a.is_complex() for a in args[:2]) else 2
  self.count[name]+=c
  return y
ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--output',required=True);o=ap.parse_args();sys.path.insert(0,o.repo)
torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
rows=[]
for ds in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity']:
 for h in [96,192,336,720]:
  for v in ['baseline','full']:
   _,cfg,_=prepare(SimpleNamespace(repo=o.repo,dataset=ds,horizon=h,variant=v,seed=42,gpu=0,output='.'))
   torch.manual_seed(42);m=load_model(v).Model(cfg).eval();x=torch.randn(1,96,cfg.enc_in)
   with torch.no_grad():reference=m(x,None,None,None)
   counter=Count()
   with torch.no_grad(),counter:y=m(x,None,None,None)
   assert torch.allclose(reference,y) and tuple(y.shape)==(1,h,cfg.enc_in)
   r=dict(dataset=ds,horizon=h,variant=v,parameters=sum(p.numel() for p in m.parameters()),estimated_flops=sum(counter.count.values()),operators=dict(counter.count),unsupported=dict(counter.unknown),torch=torch.__version__)
   rows.append(r);Path(o.output).write_text(json.dumps(rows,indent=2));print(ds,h,v,r['estimated_flops'],r['unsupported'],flush=True)

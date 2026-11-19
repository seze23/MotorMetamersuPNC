"""Evaluate published Mathis CNNs on manifest spindle inputs, without training."""
import argparse
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
import torch
from scipy.signal import savgol_filter
from inference.test_model_utils_new import load_model,parse_config_value
from utils.spindle_FR_helper import normalize,load_coefficients,get_sampled_coefficients
from extract_data.generate_train_test_data import process_chunk
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();batch=out/'batch';batch.mkdir(exist_ok=True)
    manifest=yaml.safe_load((out/'manifest.yaml').read_text())
    stimuli=pd.read_csv(out/'stimuli.csv').set_index('id')
    cfg=yaml.safe_load((ROOT/'extract_data/configs/train_test_data_spindles_extended.yaml').read_text())
    muscle_data={}; normalized={}
    for row in manifest['trajectories']:
        with np.load(out/row['muscle_data']) as data:
            lengths=data['fiber_lengths'].T[None].astype(np.float32)
            times=data['times'];dt=float(np.median(np.diff(times)))
            velocity=savgol_filter(np.gradient(lengths,dt,axis=2),31,1,axis=2).astype(np.float32)
            acceleration=np.gradient(velocity,dt,axis=2).astype(np.float32)
            normalized[row['id']]=normalize(lengths,velocity,acceleration,cfg['optimal_lengths'])
            muscle_data[row['id']]=times.copy()
    torch.set_num_threads(2)
    inventory=[]
    base=ROOT/'trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints'
    for coef in range(5):
        cfg['seed']=coef
        for kind in ['i_a','ii']:
            cfg[kind+'_coeff_path']=str(ROOT/'data/extended_spindle_coefficients'/kind/'linear/coefficients.csv')
            cfg[kind+'_sampled_coeff_path']=str(ROOT/'data/extended_spindle_coefficients'/kind/'linear'/f'sampled_coefficients_{kind}_5_{coef}.csv')
        coefficients={k:load_coefficients(cfg[k+'_coeff_path']) for k in ['i_a','ii']}
        sampled=get_sampled_coefficients(cfg,[5,5],cfg['muscles'],coefficients)
        rates={name:process_chunk(data,coefficients,[5,5],cfg['muscles'],chunk_size=1,sampled_coefficients=sampled).astype(np.float32) for name,data in normalized.items()}
        for train in [0,1,2,9]:
            model_path=base/f'spatiotemporal_4_8-8-32-64_7171_{coef}_{train}'
            config={k:parse_config_value(v) for k,v in yaml.safe_load((model_path/'config.yaml').read_text()).items()}
            config['padding_mode']='zeros'
            tester=load_model(config,str(model_path),'letter_reconstruction_joints',torch.device('cpu'),None,save_dir=str(ROOT))
            dest=batch/f'model_{coef}_{train}';dest.mkdir(exist_ok=True)
            with torch.inference_mode():
                for name,x in rates.items():
                    tensor=torch.from_numpy(x)
                    xn,_=tester._normalize_batch(tensor,torch.zeros((1,len(muscle_data[name]),7)))
                    scores,_=tester._compute_scores(xn)
                    pred=scores[0].cpu().numpy()
                    if pred.shape!=(len(muscle_data[name]),7) or not np.isfinite(pred).all():
                        raise ValueError(f'Invalid predictions for {name}, model {coef}_{train}')
                    np.savez_compressed(dest/f'{name}.npz',predictions=pred,times=muscle_data[name],coefficient_seed=coef,training_seed=train)
                    inventory.append(dict(id=name,coefficient_seed=coef,training_seed=train,
                                          predictions=str((dest/f'{name}.npz').relative_to(out)),
                                          condition=stimuli.loc[name,'condition'],e=float(stimuli.loc[name,'e'])))
            print(f'Completed model {coef}_{train}: {len(rates)} stimuli',flush=True)
    pd.DataFrame(inventory).to_csv(batch/'inventory.csv',index=False)
    provenance=dict(source='https://zenodo.org/records/15111808',models=20,predictions=len(inventory),
                    torch=torch.__version__,input_sampling_hz=240,resampling=False,
                    source_signature=json.loads((out/'generation.json').read_text())['signature'],
                    weights={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob('*/model.ckpt')})
    (batch/'provenance.json').write_text(json.dumps(provenance,indent=2))

if __name__=='__main__':main()

"""Run isolated row batches, then the one configured CNN on all circles.

Separate experiment manifests prevent parallel IK/muscle workers from writing
one shared manifest. Completed child outputs live beneath the parent run.
"""
from pathlib import Path
import argparse,copy,os,subprocess,sys,shutil,json,hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
import yaml
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--opensim-python',type=Path,default=ROOT/'.venv-opensim/bin/python');parser.add_argument('--inference-python',type=Path,default=ROOT/'.venv-mujoco/bin/python');args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text());out=ROOT/'outputs'/cfg['experiment'];out.mkdir(exist_ok=True)
    parent_config=out/'experiment.yaml';parent_config.write_text(yaml.safe_dump(cfg,sort_keys=False));(out/'chunks').mkdir(exist_ok=True);(out/'.matplotlib').mkdir(exist_ok=True)
    environment=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'.matplotlib'),CENTEROUT_WORKERS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def command(python,config,stages):
        return [str(python),'-c',f"from run_pipeline import Pipeline; p=Pipeline({str(config)!r}); "+'; '.join(f'p.run_stage({s!r})' for s in stages)]
    with (out/'path.log').open('w') as log:subprocess.run(command(args.opensim_python,parent_config,['path']),cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    positions=pd.read_csv(out/'positions.csv')
    def run_row(row,ids):
        dest=out/'chunks'/f'row_{row:02d}'
        child=copy.deepcopy(cfg);child['experiment']=f"{cfg['experiment']}_row_{row:02d}";child['path']['circle_positions']['selected_ids']=list(ids)
        if (dest/'complete.json').exists():
            old_cfg=yaml.safe_load((dest/'experiment.yaml').read_text())
            old_positions=pd.read_csv(dest/'positions.csv').sort_values('id')
            requested=positions[positions.id.isin(ids)].sort_values('id')
            same_grid=len(old_positions)==len(requested) and np.allclose(old_positions[['right_cm','forward_cm']],requested[['right_cm','forward_cm']])
            if old_cfg==child and same_grid:return row,dest
            suffix=hashlib.sha256(json.dumps(old_cfg,sort_keys=True).encode()).hexdigest()[:8]
            archive=dest.with_name(dest.name+'_prior_'+suffix);index=1
            while archive.exists():
                archive=dest.with_name(dest.name+'_prior_'+suffix+'_'+str(index));index+=1
            dest.rename(archive)
        child_out=ROOT/'outputs'/child['experiment'];child_out.mkdir(exist_ok=True);child_config=child_out/'experiment.yaml';child_config.write_text(yaml.safe_dump(child,sort_keys=False))
        print(f'Started row {row}: {len(ids)} positions',flush=True)
        with (child_out/'pipeline.log').open('w') as log:
            result=subprocess.run(command(args.opensim_python,child_config,['path','inverse-kinematics','motion','muscle-lengths','muscle-signals']),cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Row {row} failed: {child_out}/pipeline.log')
        (child_out/'complete.json').write_text(json.dumps(dict(positions=list(ids))))
        if dest.exists():raise FileExistsError(dest)
        shutil.move(str(child_out),str(dest));print(f'Completed row {row}',flush=True)
        return row,dest
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_row,int(row),g.id.tolist()) for row,g in positions.groupby('row')]
        completed=[f.result() for f in as_completed(futures)]
    manifest=yaml.safe_load((out/'manifest.yaml').read_text());entries={v['id']:v for v in manifest['trajectories']}
    for row,dest in completed:
        child=yaml.safe_load((dest/'manifest.yaml').read_text())
        for item in child['trajectories']:
            entry=entries[item['id']]
            for key,value in item.items():
                if key in ['id','desired_path','samples','sample_rate_hz']:continue
                entry[key]=(dest.relative_to(out)/value).as_posix() if isinstance(value,str) and (dest/value).is_file() else value
    (out/'manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    with (out/'validation.log').open('w') as log:
        subprocess.run([str(args.opensim_python),'process/circle_position_analysis.py','--output',str(out),'--quality-only'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All input validation passed. Running the single configured CNN.',flush=True)
    with (out/'inference.log').open('w') as log:subprocess.run(command(args.inference_python,parent_config,['inference']),cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'readout.log').open('w') as log:subprocess.run([str(args.opensim_python),'process/circle_position_analysis.py','--output',str(out)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {len(positions)} circle positions with one checkpoint: {out}',flush=True)

if __name__=='__main__':main()

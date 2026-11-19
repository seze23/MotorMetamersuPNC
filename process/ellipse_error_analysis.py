"""Describe single-CNN reconstruction error and human PSE uncertainty.

No model ensemble, trial-noise injection, or bootstrap over autocorrelated
frames is used. Model PSE bars are grid brackets, not statistical intervals.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1]

def mean_ci(values):
    values=np.asarray(values,dtype=float)
    mean=values.mean();sd=values.std(ddof=1)
    half=student_t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,n=len(values))

def crossing_bracket(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if not np.isfinite(y).all():return (np.nan,)*3
    order=np.argsort(x);x,y=x[order],y[order];roots=[]
    for i in range(len(x)):
        if abs(y[i])<1e-10:roots.append((x[i],x[i],x[i]))
        elif i<len(x)-1 and y[i]*y[i+1]<0:
            root=x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])
            roots.append((root,x[i],x[i+1]))
    return roots[0] if len(roots)==1 else (np.nan,)*3

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pse-predictions',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    predictions=pd.read_csv(out/'batch/shape_predictions.csv')
    pse_predictions=pd.read_csv(args.pse_predictions) if args.pse_predictions else predictions
    if len(pse_predictions[['coefficient_seed','training_seed']].drop_duplicates())!=1:
        raise ValueError('PSE refinement must use the same single checkpoint.')
    if len(predictions[['coefficient_seed','training_seed']].drop_duplicates())!=1:
        raise ValueError('This analysis requires one configured CNN.')
    errors=[]
    for row in predictions.itertuples():
        actual=np.load(out/'muscles'/f'{row.id}.npz')
        decoded=np.load(out/row.physical_prediction)
        indices=np.searchsorted(actual['times'],decoded['times'])
        np.testing.assert_allclose(actual['times'][indices],decoded['times'],atol=1e-10)
        delta=decoded['xyz']-actual['wrist_xyz_reference_world'][indices]
        point_error=np.linalg.norm(delta,axis=1)
        centered_delta=(decoded['xyz']-decoded['xyz'].mean(axis=0))-(actual['wrist_xyz_reference_world'][indices]-actual['wrist_xyz_reference_world'][indices].mean(axis=0))
        errors.append(dict(id=row.id,condition=row.condition,e=row.e,
                           mean_point_error_cm=point_error.mean(),sd_point_error_cm=point_error.std(ddof=1),
                           rmse_3d_cm=np.sqrt(np.mean(np.sum(delta**2,axis=1))),
                           centered_rmse_3d_cm=np.sqrt(np.mean(np.sum(centered_delta**2,axis=1))),
                           peak_point_error_cm=point_error.max(),n_phase_samples=len(point_error)))
    errors=pd.DataFrame(errors);errors.to_csv(out/'trajectory_errors.csv',index=False)
    errors.groupby('condition').agg(mean_rmse_cm=('rmse_3d_cm','mean'),sd_rmse_across_shapes_cm=('rmse_3d_cm','std'),n_shapes=('rmse_3d_cm','size')).to_csv(out/'condition_errors.csv')
    people=pd.read_csv(ROOT/'notebooks/data/viviani_table1_participant_pses.csv')
    comparison=[]
    for condition in 'ABC':
        h=mean_ci(people[condition]);g=pse_predictions[pse_predictions.condition==condition]
        model,lo,hi=crossing_bracket(g.e,g.predicted_e)
        comparison.append(dict(condition=condition,human_pse=h['mean'],human_sd=h['sd'],human_ci95_low=h['ci95_low'],human_ci95_high=h['ci95_high'],human_n=h['n'],cnn_pse=model,cnn_bracket_low=lo,cnn_bracket_high=hi,cnn_minus_human=model-h['mean']))
    comparison=pd.DataFrame(comparison);comparison.to_csv(out/'human_model_comparison.csv',index=False)
    comp=comparison.set_index('condition');shifts=[]
    for condition in 'BC':
        h=mean_ci(people[condition]-people.A)
        shifts.append(dict(contrast=condition+'-A',human_mean_shift=h['mean'],human_ci95_low=h['ci95_low'],human_ci95_high=h['ci95_high'],human_n=h['n'],
                           cnn_shift=comp.loc[condition,'cnn_pse']-comp.loc['A','cnn_pse'],
                           cnn_bracket_low=comp.loc[condition,'cnn_bracket_low']-comp.loc['A','cnn_bracket_high'],
                           cnn_bracket_high=comp.loc[condition,'cnn_bracket_high']-comp.loc['A','cnn_bracket_low']))
    pd.DataFrame(shifts).to_csv(out/'pse_shift_comparison.csv',index=False)

if __name__=='__main__':main()

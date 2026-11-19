"""Geometric ellipse diagnostics for decoded paths in a physical plane."""
import numpy as np
from scipy.optimize import least_squares

def extent_shape(xy,tolerance=1.05):
    span=np.ptp(xy,axis=0)
    if not np.isfinite(span).all() or span.min()<1e-5:return 'invalid',np.nan
    ratio=span[0]/span[1]
    label='circle' if 1/tolerance<=ratio<=tolerance else 'horizontal ellipse' if ratio>1 else 'vertical ellipse'
    return label,float(ratio)

def fit_ellipse(xy):
    xy=np.asarray(xy,dtype=float)
    if len(xy)<20 or not np.isfinite(xy).all() or np.ptp(xy,axis=0).min()<1e-5:
        return dict(axis_ratio=np.nan,angle_deg=np.nan,major_radius_cm=np.nan,minor_radius_cm=np.nan,fit_rmse_cm=np.nan)
    center=xy.mean(axis=0);span=np.ptp(xy,axis=0);scale=float(span.max())
    _,vectors=np.linalg.eigh(np.cov((xy-center).T));v=vectors[:,-1];theta=float(np.arctan2(v[1],v[0]))
    theta=(theta+np.pi/2)%np.pi-np.pi/2
    rotated=(xy-center)@np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    radii=np.maximum(np.ptp(rotated,axis=0)/2,scale*.02)
    def residual(p):
        c,s=np.cos(p[4]),np.sin(p[4]);delta=xy-p[:2]
        u=delta[:,0]*c+delta[:,1]*s;v=-delta[:,0]*s+delta[:,1]*c
        a,b=np.exp(p[2:4]);return (np.sqrt((u/a)**2+(v/b)**2)-1)*np.sqrt(a*b)
    lo=np.r_[xy.min(axis=0)-scale,np.log([scale*.01]*2),-2*np.pi]
    hi=np.r_[xy.max(axis=0)+scale,np.log([scale*2]*2),2*np.pi]
    fit=least_squares(residual,np.r_[center,np.log(radii),theta],bounds=(lo,hi),max_nfev=300)
    a,b=np.exp(fit.x[2:4]);theta=fit.x[4]
    if b>a:a,b=b,a;theta+=np.pi/2
    angle=(np.rad2deg(theta)+90)%180-90
    return dict(axis_ratio=float(a/b),angle_deg=float(angle),major_radius_cm=float(a),minor_radius_cm=float(b),fit_rmse_cm=float(np.sqrt(np.mean(residual(fit.x)**2))))

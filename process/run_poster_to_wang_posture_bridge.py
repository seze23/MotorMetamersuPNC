"""Corrected-TMIN common-history bridge from a poster reach to Wang posture."""

from pathlib import Path
import io
import subprocess
import sys

import h5py
import numpy as np
import opensim as osm
import pandas as pd
from scipy.signal import savgol_filter
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1e1dd51"
POSTER_PATH = "dataexp/centerout/center_out_g000_r08_spindles_coefseed0.npz"
WANG_IK = ROOT / "outputs/wang_grid_full_scale_refined/ik/r02_c10.npz"
MODEL_OSIM = ROOT / "MoBL-ARMSDynamicUpperLimb-latest/MoBL-ARMS Upper Extremity Model/Model/4.1/DefaultMOBL_ARMS_fixed_41.osim"
MODELS = ROOT / "trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints"
OUT = ROOT / "dataexp/poster_bias_investigation/poster_to_wang_posture_bridge"
COORDS = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
MUSCLES = ['CORB','DELT1','DELT2','DELT3','INFSP','LAT1','LAT2','LAT3','PECM1','PECM2','PECM3','SUBSC','SUPSP','TMAJ','TMIN','ANC','BIClong','BICshort','BRA','BRD','ECRL','PT','TRIlat','TRIlong','TRImed']
ALPHAS = np.linspace(0, 1, 21)
PRE, REACH, N = 396, 120, 816
PEAK = PRE + REACH - 1

sys.path.insert(0, str(ROOT))
from extract_data.generate_train_test_data import process_chunk
from inference.test_model_utils_new import load_model, parse_config_value
from train.new_spindle_dataset import SpindleDataset
from utils.spindle_FR_helper import normalize, load_coefficients, get_sampled_coefficients
from utils.visualize_sample import get_shoulder_elbow_wrist_loc


def git_npz(path):
    return np.load(io.BytesIO(subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)))


def make_trajectories():
    poster = git_npz(POSTER_PATH)
    base = poster["joint_angles"][:N].astype(np.float64)
    poster_endpoint = base[PEAK].copy()
    wang_endpoint = np.load(WANG_IK)["joint_angles"][-1].astype(np.float64)
    phase = np.linspace(0, 1, REACH)
    profile = 10*phase**3 - 15*phase**4 + 6*phase**5
    trajectories = []
    for alpha in ALPHAS:
        q = base.copy()
        delta = alpha * (wang_endpoint - poster_endpoint)
        q[PRE:PRE+REACH] += profile[:, None] * delta
        q[PRE+REACH:] = poster_endpoint + delta
        trajectories.append(q)
    return np.stack(trajectories), poster


def extract_lengths(trajectories):
    model = osm.Model(str(MODEL_OSIM)); state = model.initSystem()
    tmin = model.updMuscles().get("TMIN"); tmin.updGeometryPath().moveUpPathWrap(state, 2)
    coordinates = [model.getCoordinateSet().get(name) for name in COORDS]
    muscles = [model.getMuscles().get(name) for name in MUSCLES]
    result = np.empty((len(trajectories), N, 25), dtype=np.float32)
    cache = {}
    for trial, trajectory in enumerate(trajectories):
        for frame, angles in enumerate(trajectory):
            key = np.asarray(angles, dtype=np.float32).tobytes()
            if key not in cache:
                for coordinate, degrees in zip(coordinates, angles):
                    coordinate.setValue(state, np.deg2rad(float(degrees)))
                model.equilibrateMuscles(state)
                cache[key] = np.array([muscle.getFiberLength(state)*1000 for muscle in muscles], dtype=np.float32)
            result[trial, frame] = cache[key]
        print("geometry", trial + 1, "/", len(trajectories), flush=True)
    return result


def rates_from_lengths(lengths, coefficient_seed):
    config = yaml.safe_load((ROOT / "extract_data/configs/train_test_data_spindles_extended.yaml").read_text())
    config["seed"] = coefficient_seed
    for key in ("i_a_coeff_path","ii_coeff_path","i_a_sampled_coeff_path","ii_sampled_coeff_path"):
        if config.get(key) is not None: config[key] = str(ROOT / config[key])
    coefficients = {key: load_coefficients(config[key + "_coeff_path"]) for key in ("i_a","ii")}
    counts = [config["num_i_a"], config["num_ii"]]
    sampled = get_sampled_coefficients(config, counts, config["muscles"], coefficients)
    data = lengths.transpose(0,2,1)
    dt = 1/240
    velocity = savgol_filter(np.gradient(data, dt, axis=2), 31, 1, axis=2)
    acceleration = np.gradient(velocity, dt, axis=2).astype(np.float32)
    normalized = normalize(data, velocity, acceleration, config["optimal_lengths"])
    return process_chunk(normalized, coefficients, counts, config["muscles"], len(data), sampled).astype(np.float32)


def labels_from_trajectories(trajectories):
    labels = np.zeros((len(trajectories), N, 7), dtype=np.float32)
    labels[:,:,3:7] = trajectories[:,:,:4]
    for index in range(len(labels)):
        labels[index,:,:3] = get_shoulder_elbow_wrist_loc(labels[index])[2]
    return labels


def infer(rates, labels, coefficient_seed, training_seed, condition_values=None):
    temp = OUT / f"_tmp_{coefficient_seed}_{training_seed}.hdf5"
    with h5py.File(temp, "w") as stream:
        stream.create_dataset("data", data=rates); stream.create_dataset("labels", data=labels)
    name = f"{coefficient_seed}_{training_seed}"; model = MODELS / f"spatiotemporal_4_8-8-32-64_7171_{name}"
    config = yaml.safe_load((model / "config.yaml").read_text()); config = {k:parse_config_value(v) for k,v in config.items()}; config["padding_mode"]="zeros"
    dataset = SpindleDataset(str(temp), dataset_type="test", key="spindle_info", task="letter_reconstruction_joints", aclass=None, need_muscles=False, new_size=config["input_shape"][-1])
    tester = load_model(config, str(model), "letter_reconstruction_joints", torch.device("cpu"), dataset, causal=True, save_dir=str(ROOT)); tester.batch_size=min(40, len(labels))
    pred,_=tester.get_predictions(); pred=pred.detach().cpu().numpy(); temp.unlink()
    if condition_values is None:
        condition_values = ALPHAS
    rows=[]
    for index, condition_value in enumerate(condition_values):
        raw=pred[index,PEAK,:3]-labels[index,PEAK,:3]
        rows.append({"model":name,"coefficient_seed":coefficient_seed,"training_seed":training_seed,
                     "condition":condition_value,
                     "alpha":condition_value if np.isscalar(condition_value) and not isinstance(condition_value, str) else np.nan,
                     "true_x":labels[index,PEAK,0],"true_y":labels[index,PEAK,1],"true_z":labels[index,PEAK,2],
                     "outward_cm":-raw[0],"forward_cm":-raw[1],"up_cm":raw[2],"l2_cm":np.linalg.norm(raw)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trajectories, poster = make_trajectories(); labels=labels_from_trajectories(trajectories)
    lengths_path=OUT/"corrected_muscle_lengths.npy"
    lengths=np.load(lengths_path) if lengths_path.exists() else extract_lengths(trajectories)
    if not lengths_path.exists(): np.save(lengths_path,lengths)
    all_rows=[]; integrity=[]
    for coefficient in range(5):
        rates=rates_from_lengths(lengths,coefficient)
        archived=git_npz(POSTER_PATH.replace("coefseed0",f"coefseed{coefficient}"))["firing_rates"][0,:,:,:N]
        non_tmin=np.arange(25)!=14
        integrity.append({"coefficient_seed":coefficient,"alpha0_max_non_tmin_rate_difference":float(np.max(np.abs(rates[0][:,non_tmin]-archived[:,non_tmin]))),
                          "alpha0_tmin_rate_difference":float(np.max(np.abs(rates[0,:,14]-archived[:,14]))),
                          "tmin_length_min_mm":float(lengths[:,:,14].min()),"tmin_length_max_mm":float(lengths[:,:,14].max())})
        for training in (0,1,2,9): all_rows.extend(infer(rates,labels,coefficient,training)); print("model",coefficient,training,flush=True)
    result=pd.DataFrame(all_rows);result.to_csv(OUT/"model_bridge_results.csv",index=False)
    pd.DataFrame(integrity).to_csv(OUT/"integrity.csv",index=False)
    summary=result.groupby("alpha").agg(true_x=("true_x","mean"),true_y=("true_y","mean"),true_z=("true_z","mean"),mean_outward_cm=("outward_cm","mean"),std_outward_cm=("outward_cm","std"),fraction_outward=("outward_cm",lambda x:float((x>0).mean()))).reset_index()
    summary.to_csv(OUT/"ensemble_bridge_summary.csv",index=False);print(summary.to_string(index=False))


if __name__=="__main__": main()

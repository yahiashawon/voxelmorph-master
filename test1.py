# test.py
import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.interpolate import interpn
from keras.backend.tensorflow_backend import set_session

# Add medipy-lib to path
sys.path.append('C:/Users/yhsha/Desktop/voxelmorph-master/voxelmorph-master/voxelmorph-master/ext/medipy-lib')
import medipy
from medipy.metrics import dice
import networks
import datagenerators

def test(model_name, iter_num, gpu_id, 
         vol_size=(160,192,224), 
         nf_enc=[16,32,32,32], 
         nf_dec=[32,32,32,32,32,16,16,3]):
    
    print("\n[1/6] Initializing test...")
    
    # Device configuration
    device = '/cpu:0' if gpu_id == -1 else f'/gpu:{gpu_id}'
    print(f"Using device: {device}")

    # Load anatomical labels
    try:
        labels = sio.loadmat('./data/labels.mat')['labels'][0]
        print(f"[2/6] Loaded {len(labels)} labels")
    except Exception as e:
        print(f"Error loading labels: {str(e)}")
        return

    # Load atlas data
    try:
        atlas = np.load('./data/atlas_norm.npz')
        atlas_vol = np.reshape(atlas['vol'], (1,) + atlas['vol'].shape + (1,))
        atlas_seg = atlas['seg']
        print("[3/6] Loaded atlas data")
    except Exception as e:
        print(f"Error loading atlas: {str(e)}")
        return

    # Configure TensorFlow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.compat.v1.Session(config=config))

    # Load model
    try:
        print("[4/6] Loading model...")
        model_path = f'./models/{model_name}.h5'
        with tf.device(device):
            net = networks.unet(vol_size, nf_enc, nf_dec)
            net.load_weights(model_path)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return

    # Generate grid
    grid = np.stack(np.meshgrid(
        np.arange(vol_size[1]),  # x
        np.arange(vol_size[0]),  # y 
        np.arange(vol_size[2]),  # z
    ), axis=0)

    # Load test data
    try:
        print("[5/6] Loading test data...")
        X_vol, X_seg = datagenerators.load_example_by_name(
            './data/test_vol.npz',
            './data/test_seg.npz'
        )
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return

    # Run prediction
    try:
        print("[6/6] Running prediction...")
        with tf.device(device):
            pred = net.predict([X_vol, atlas_vol], batch_size=1)
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return

    # Warp segments
    flow = pred[1][0]
    sample = flow + grid.transpose(1,2,3,0)
    sample = sample[..., [1,0,2]]  # reorder coordinates
    
    try:
        warp_seg = interpn(
            (np.arange(vol_size[0]),  # y
            np.arange(vol_size[1]),  # x
            np.arange(vol_size[2])),  # z
            X_seg[0,...,0], 
            sample, 
            method='nearest', 
            bounds_error=False, 
            fill_value=0
        )
    except Exception as e:
        print(f"Warping failed: {str(e)}")
        return

    # Calculate Dice scores
    try:
        vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
        print("\n=== RESULTS ===")
        print(f"Dice score: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        print("===============")
    except Exception as e:
        print(f"Dice calculation failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., vm2_cc)")
    parser.add_argument("--iter", type=int, default=0, help="Iteration number (unused)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    
    test(
        model_name=args.model,
        iter_num=args.iter,
        gpu_id=-1 if args.device == "cpu" else 0
    )
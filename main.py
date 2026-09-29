#
# FuME Compression Tool
# Author: Port Christman
#
import time
import argparse
import sys
import dataBreakOut
import dataRegressor
import numpy as np
import verif
import os

parser = argparse.ArgumentParser(
    description="FuME Compression Tool",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("input_file", help="Path to input image/video.")
parser.add_argument("-v", "--verbose",      action="store_true",     help="Enable verbose output.")
parser.add_argument("--batch_iterations",   type=int, default=40,    help="PySR iterations per batch.")
parser.add_argument("--max_total_iterations", type=int, default=1500, help="Max total PySR iterations.")
parser.add_argument("--training_res",       type=int, default=256,
                    help="Downscale image to this max dimension for regression training.")
parser.add_argument("--fourier_freqs",      type=int, default=5,
                    help="Number of Fourier octave bands to pre-compute as input features. "
                         "More bands = higher-frequency patterns learnable, but larger search space. "
                         "4 = 18 total features; 5 = 22 features.")
parser.add_argument("--parsimony",          type=float, default=1e-6, help="complexity cost.")
parser.add_argument("--maxsize",            type=int,   default=256,   help="Max equation size (nodes).")
parser.add_argument("--processes",          type=int,   default=os.cpu_count(),
                    help="Number of parallel processes/threads for PySR.")
 
args = parser.parse_args()

def regress_channel(df, feature_cols, channel_name):
    """
    Fit f(x0, x1, …, xN) → pixel intensity for one colour channel.
 
    feature_cols gives the ordered list of input column names; this must match
    the order used by verif.py when reconstructing features from (x, y).
    """
    if df is None or df.empty:
        return None
 
    # All columns except the target
    X = df[feature_cols].values.astype(np.float64)
    y = df['y'].values.astype(np.float64)
 
    print(f"\n--- Regressing {channel_name} Channel "
          f"({X.shape[0]} pixels × {X.shape[1]} features) ---")
 
    results = dataRegressor.run_pysr_regression(
        X=X,
        y=y,
        iterations=args.batch_iterations,
        verbose=args.verbose,
        extra_pysr_params={
            'parsimony':    args.parsimony,
            'maxsize':      args.maxsize,
            'niterations':  args.max_total_iterations,
            'procs':        args.processes,
            'parallelism':  'multithreading',
        }
    )
 
    output = {'equation': 'None', 'complexity': 0}
 
    if results['success'] and results['equation_found']:
        output['equation'] = results['best_equation_str']
        if results.get('best_equation_details'):
            output['complexity'] = results['best_equation_details'].get('complexity', 0)
        print(f"  [{channel_name}] Eq (complexity {output['complexity']}): {output['equation']}")
    else:
        print(f"  [{channel_name}] Failed to find equation.")
 
    return output

def main():
    print("FuME Compression Tool")
    
    startTime = time.time()
    
    try:
        channel_dfs = dataBreakOut.process_media(
            input_path=args.input_file,
            return_data=True,
            verbose=args.verbose,
            max_dimension=args.training_res 
        )
    except Exception as e:
        print(f"Error processing media: {e}")
        sys.exit(1)

    # results dictionary
    equation_results = {}
    
    channels_to_process = ['r', 'g', 'b']
    if 'a' in channel_dfs and channel_dfs['a'] is not None:
         if channel_dfs['a']['y'].std() > 0:
            channels_to_process.append('a')
            
    for channel in channels_to_process:
        if channel in channel_dfs:
            feature_cols = [c for c in channel_dfs[channel].columns if c != 'y']
            result = regress_channel(channel_dfs[channel], feature_cols, channel.upper())
            if result:
                equation_results[channel.upper()] = result
    
    endTime = time.time()
    
    # We pass the equation_results to verify, which calculates stats on the full image
    print("\nCalculated Regression. Verifying against full resolution...")
    verification_stats = verif.verify_and_reconstruct(args.input_file, equation_results)

    print("\n" + "="*80)
    print(f"FuME COMPRESSION RESULTS | Time: {(endTime-startTime)/60:.2f} min")
    print("="*80)
    print(f"{'CHANNEL':<8} | {'COMPLEXITY':<10} | {'PEAK DEV':<10} | {'MEDIAN DEV':<10} | {'EQUATION'}")
    print("-" * 80)
    
    for chan in ['R', 'G', 'B', 'A']:
        if chan in equation_results:
            eq_data = equation_results[chan]
            stats = verification_stats.get(chan, {})
            
            eq_str = eq_data.get('equation', 'N/A')
            comp = eq_data.get('complexity', 0)
            peak = stats.get('peak', 0)
            median = stats.get('median', 0)
            
            # Truncate equation for display if too long
            display_eq = (eq_str[:45] + '..') if len(eq_str) > 45 else eq_str
            
            print(f"{chan:<8} | {comp:<10} | {peak:<10.0f} | {median:<10.0f} | {display_eq}")
    print("="*80)

if __name__ == "__main__":
    main()

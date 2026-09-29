import os
import csv
import time
import argparse
import numpy as np
import cv2
from pysr import PySRRegressor
import dataBreakOut
import dataRegressor
import verif

def main():
    parser = argparse.ArgumentParser(description="Batch process images and output CSV for Excel.")
    parser.add_argument("input_folder", help="Folder in root containing images to test.")
    parser.add_argument("output_csv", help="Output CSV file path (e.g., results.csv).")
    parser.add_argument("--batch_iterations", type=int, default=40)
    parser.add_argument("--max_total_iterations", type=int, default=1500)
    parser.add_argument("--training_res", type=int, default=256)
    parser.add_argument("--parsimony", type=float, default=1e-6)
    parser.add_argument("--maxsize", type=int, default=256)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--fourier_freqs", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Folder '{args.input_folder}' not found.")
        return
    
    global_pysr_params = {
    'parsimony': args.parsimony, 
    'maxsize': args.maxsize, 
    'niterations': args.max_total_iterations, 
    'procs': args.processes, 
    'parallelism': 'multithreading'
    }
    
    shared_model = PySRRegressor(
    model_selection="accuracy",
    binary_operators=["+", "*", "-", "/", "max", "min"],
    unary_operators=["square", "abs", "sqrt", "sin", "cos", "tanh", "exp"],
    batching=True,
    batch_size=512,
    turbo=True,
    temp_equation_file=True,
    delete_tempfiles=True,
    **global_pysr_params
    )
    
    TimeStartTotal = time.time()
    print(f"Starting batch processing at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TimeStartTotal))}")
    
    # Filter for standard image formats
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [
        f for f in os.listdir(args.input_folder) 
        if f.lower().endswith(image_extensions) 
        and not f.startswith("reconstructed_") 
        and not f.endswith("_error.png")
    ]

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Filename', 'Channel', 'Complexity', 'MAE', 'PSNR', 
            'Peak Dev', 'Median Dev', 'Time (s)', 'Equation'
        ])

        for file in files:
            filepath = os.path.join(args.input_folder, file)
            timeStartFile = time.time()
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if(img is not None):
                pixelCount = img.size
            else:
                print(f"Warning: Unable to read image {file}. Skipping.")
                continue
            print(f"Starting processing [{file}] at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TimeStartTotal))}")
            try:
                # Pass the fourier argument into breakout
                channel_dfs = dataBreakOut.process_media(
                    input_path=filepath, return_data=True, 
                    verbose=args.verbose, max_dimension=args.training_res,
                    fourier_freqs=args.fourier_freqs
                )
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                continue

            # Guard against process_media returning None or empty data
            if not channel_dfs:
                print(f"Warning: Could not extract channel data for [{file}]. Skipping.")
                continue

            equation_results = {}
            channels_to_process = ['r', 'g', 'b']
            
            # Identify if alpha channel data exists and contains variations
            if 'a' in channel_dfs and channel_dfs['a'] is not None:
                if channel_dfs['a']['y'].std() > 0:
                    channels_to_process.append('a')
                
            channel_times = {}
            for channel in channels_to_process:
                if channel in channel_dfs:
                    df = channel_dfs[channel]
                    feature_cols = [c for c in df.columns if c != 'y']
                    
                    X = df[feature_cols].values.astype(np.float64)
                    y = df['y'].values.astype(np.float64)
                    
                    reg_start = time.time()
                    res = dataRegressor.run_pysr_regression(
                        X=X, y=y, 
                        model=shared_model, # Pass the hot model here
                        iterations=args.batch_iterations, 
                        verbose=args.verbose
                    )
                    
                    output = {'equation': 'None', 'complexity': 0}
                    if res['success'] and res['equation_found']:
                        output['equation'] = res['best_equation_str']
                        if res.get('best_equation_details'):
                            output['complexity'] = res['best_equation_details'].get('complexity', 0)
                    
                    equation_results[channel.upper()] = output
                    channel_times[channel.upper()] = time.time() - reg_start
            
            # Generate reconstructions and error maps in the source folder
            output_stem = os.path.join(args.input_folder, f"reconstructed_{os.path.splitext(file)[0]}")
            # Pass the fourier argument into verif
            verification_stats = verif.verify_and_reconstruct(
                filepath, 
                equation_results, 
                output_stem=output_stem,
                fourier_freqs=args.fourier_freqs
            )
            time_elapsed = time.time() - timeStartFile
            
            print(f"Finished processing {file} in {time_elapsed:.2f} seconds.")
            
            # Map statistical outcomes to CSV structure
            for chan in ['R', 'G', 'B', 'A']:
                if chan in equation_results:
                    eq = equation_results[chan].get('equation', 'None')
                    comp = equation_results[chan].get('complexity', 0)
                    stats = verification_stats.get(chan, {})
                    
                    writer.writerow([
                        file,
                        pixelCount, 
                        chan, 
                        comp, 
                        f"{stats.get('mae', 0):.4f}", 
                        f"{stats.get('psnr', 0):.4f}", 
                        f"{stats.get('peak', 0):.4f}", 
                        f"{stats.get('median', 0):.4f}", 
                        f"{channel_times.get(chan, 0):.2f}",
                        eq
                    ])
                    # Force write to disk immediately so graphing can begin mid-batch
                    csvfile.flush()
                    
    print(f"\nBatch testing complete. Graphical data successfully saved to {args.output_csv}")
    print(f"Total time taken: {time.time() - TimeStartTotal:.2f} seconds")

if __name__ == "__main__":
    main()
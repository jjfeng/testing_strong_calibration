"""
a script for wrapping around my python scripts
"""
import os
import time
import sys
from pathlib import Path
import argparse
import logging
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
        description="run code on the cluster or locally"
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=300
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default='local',
        choices=['local', 'cluster']
    )
    parser.add_argument(
        "--target-template-files",
        type=str,
    )
    parser.add_argument(
        "--run-line",
        type=str,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
    )
    args = parser.parse_args()
    args.target_template_files = args.target_template_files.split(",")
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    # figure out missing seeds
    missing_seeds = []
    for seed_idx in range(1, args.num_seeds + 1):
        seed_completed = True
        for target_template_file in args.target_template_files:
            target_file = target_template_file.replace("SEED", str(seed_idx))
            print("checking for target file", target_file)
            if not os.path.exists(target_file):
                seed_completed = False
                break
        if not seed_completed:
            missing_seeds.append(seed_idx)
    print("missing seeds", missing_seeds)

    if args.cluster == 'local':
        for i in missing_seeds:
            run_cmd = "python %s --seed %d" % (args.run_line, i)
            print(run_cmd)
            subprocess.check_output(
                run_cmd, stderr=subprocess.STDOUT, shell=True
            )
    else:
        if len(missing_seeds) == 0:
            print("dont need to run qsub")
        elif (max(missing_seeds) - min(missing_seeds)) != (len(missing_seeds) - 1):
            # if the missing seeds does not correspond to a straightforward sequence
            for seed_idx in missing_seeds:
                qsub_cmd = ("qsub -t %d-%d run_script.sh %s" % (seed_idx, seed_idx, args.run_line))
                print(qsub_cmd)
                output = subprocess.check_output(
                    qsub_cmd,
                    stderr=subprocess.STDOUT,
                    shell=True,
                )
                print("QSUB DONE", output)
        else:
            qsub_cmd = ("qsub -t %d-%d -tc %d run_script.sh %s" % (min(missing_seeds),
                max(missing_seeds), args.num_parallel, args.run_line))
            print(qsub_cmd)
            output = subprocess.check_output(
                qsub_cmd,
                stderr=subprocess.STDOUT,
                shell=True,
            )
            print("QSUB DONE", output)

    # Check that the desired files are in the file system.
    wait_iters = 0
    for t in range(20 * args.num_seeds):
        seeds_completed = []
        for seed_idx in missing_seeds:
            job_completed = True
            for target_template_file in args.target_template_files:
                target_file = target_template_file.replace("SEED", str(seed_idx))
                print("checking for target file", target_file)
                if not os.path.exists(target_file):
                    job_completed = False
                    break
            if job_completed:
                seeds_completed.append(seed_idx)

        do_finish = (len(seeds_completed) == len(missing_seeds))
        if len(seeds_completed) > 0.9 * len(missing_seeds):
            wait_iters += 1
            if wait_iters > 6:
                do_finish = True
        if do_finish:
            # If I have been waiting and the number of results hasn't
            # increased, just consider the job completed and move onwards.
            # Create files to indicate to scons that the job is done
            for target_template_file in args.target_template_files:
                Path(target_template_file).touch()
            break
        else:
            time.sleep(30)

    time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:])



import argparse
import json
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to copy and setup checkpoints')
    parser.add_argument('--uname',      default="kpius", help="user name")
    parser.add_argument('--server',     default="tik42x.ethz.ch", help="server adress")
    parser.add_argument('--configs_dir',default="src_v2/configs", help="server adress")
    parser.add_argument('--to_dir',     default="checkpoints", help="local directory where files are saved")
    parser.add_argument('--from_dir',   default=None, help="path to where the run is located")
    args = parser.parse_args()

    # copy checkpoints and files
    from_dirs_path_split = args.from_dir.split("/")
    server_base_dir = f"{args.uname}@{args.server}:{args.from_dir}"
    server_dirs = [
        os.path.join(server_base_dir, "checkpoint_*"),
        os.path.join(server_base_dir, "params.json"),
        os.path.join(server_base_dir, "result.json"),
    ]
    local_dir = os.path.join(args.to_dir, from_dirs_path_split[-2])
    
    print()
    print("==== COPY CHECKPOINT SCRIPT ====")
    print(f"fetch run {args.from_dir}")
    print(f"  from {local_dir}")
    print()

    # fetch from sever
    for fp in server_dirs:
        subprocess.run(['scp', '-r', fp, local_dir], check=True, text=True)
    print()

    # get config used
    params_dir = os.path.join(local_dir, "params.json")
    with open(params_dir, 'r') as f:
        data = json.load(f)
        config_file = data["model"]["custom_model_config"]["info"]["env_config"]
        print(f"copy environment configuration file {config_file}")
        config_file_from = os.path.join(args.configs_dir, config_file)
        config_file_to = os.path.join(local_dir, config_file)
        subprocess.run(['cp', '-r', config_file_from, config_file_to], check=True, text=True)

    print()
    print("ready to run with python run.py")

    
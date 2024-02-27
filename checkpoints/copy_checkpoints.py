

import argparse
import json
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to copy and setup checkpoints')
    parser.add_argument('--uname',      default="kpius", help="user name")
    parser.add_argument('--server',     default="tik42x.ethz.ch", help="server adress")
    parser.add_argument('--configs_dir',default="src_v3/configs", help="server adress")
    parser.add_argument('--to_dir',     default="checkpoints", help="local directory where files are saved")
    parser.add_argument('--group',      default=None, help="group name, something like marl-env-202402-13-10-1855")
    parser.add_argument('--run',        default=None, help="run number")
    args = parser.parse_args()

    # copy checkpoints and files
    group_dir = os.path.join("/itet-stor", args.uname, "net_scratch", "si_bees", "log", args.group)
    ssh_command = [
        "ssh",
        f"{args.uname}@{args.server}",
        "ls",
        group_dir
    ]
    # Execute the SSH command and capture the output
    result = subprocess.run(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run_names = result.stdout.strip().split('\n')
    run_name = [d for d in run_names if f"{'{:05d}'.format(int(args.run))}_{int(args.run)}" in d]
    if not run_name:
        raise Exception(f"couldn find run {args.run} in group {args.group}")
    
    run_dir = os.path.join(group_dir, run_name[0])
    server_base_dir = f"{args.uname}@{args.server}:{run_dir}"
    server_dirs = [
        os.path.join(server_base_dir, "checkpoint_*"),
        os.path.join(server_base_dir, "params.*"),
        os.path.join(server_base_dir, "result.json"),
    ]
    local_dir = os.path.join(args.to_dir, f"{args.group}-r{args.run}")
    
    print()
    print("==== COPY CHECKPOINT SCRIPT ====")
    print(f"fetch run {run_dir}")
    print(f"  to {local_dir}")
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
    print("-> create notes file")
    run_command = f"python src_v3/run.py --checkpoint_name {args.group}-r{args.run} --task_level 0"
    with open(os.path.join(local_dir, "notes.txt"), 'w') as file:
            file.write(run_command)

    print("-> ready to run with python run.py")
    print(f"  {run_command}")

    
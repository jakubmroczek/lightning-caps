import sys, os

if __name__ == '__main__':
    EXPERIMENT_PREFIX = '--experiment'
    
    # Input format: --experiment=example from STDOUT
    # We handle only 1 experiment at the time
    assert len(sys.argv) == 3
    assert EXPERIMENT_PREFIX in sys.argv[1]

    experiment_name = sys.argv[2]

    print(f"Executing: train.py experiment={experiment_name}")
    os.system(f"python3 src/train.py experiment={experiment_name}")
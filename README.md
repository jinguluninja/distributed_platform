# distributed_platform

The following instructions apply to each institution

PREREQUISITES
- Linux machine (preferably Ubuntu 16.04 or later) with configured GPU (e.g. must have
  CUDA installed)
- python 3.7 with virtualenv
- required data organization (no other folders/files than the ones listed below):
  data_dir/
  	labels.csv with following format in each line:
  		filename.npy,label(int from 0 to num_classes-1)
  	train/
  		filename.npy files for train
  	val/
  		filename.npy files for val
  	test/
  		filename.npy files for test
- required data format: preprocessed .npy files, each with array of shape 
  (height, width, channels)
- I will refer to this repo in this instructions as "code_repo"
- new empty github repository (outside "code_repo") with commit privileges for each
  institution (this empty repo will be used internally for model file transfers)
  I will refer to this new empty repo in this instructions as "file_repo"

SETUP
- create python3.7 virtual environment: python3.7 -m venv /path/to/env_name
- activate virtual env (must be activate whenever running code): 
    source /path/to/env_name/bin/activate
- install required libraries: pip install -r /path/to/code_repo/requirements.txt

INSTRUCTIONS TO RUN
- for custom model architecture, fill in custom_model function in nets_classficication.py
- create sh_file_name.sh file (placed in file_repo) with call to classification.py (see
  classification.py for documentation of arguments and see example.sh for
  an example .sh file)
- run .sh file from within file_repo: sh sh_file_name.sh
- at each call to classification.py, you may be prompted to enter your github credentials
- during training, each institution must be running classification.py with apppropriate
  arguments simultaneously
- can look at training progress at other institutions in log files
- combined validation and testing results will be published in the log file of the last 
  institution (***IMPORTANT THAT EACH INSTITUTION HAS A DIFFERENT NAME FOR THEIR LOG
  FILE***)
- can access current saved models as well as best saved models (lowest val loss) in 
  file_repo (best model will be saved into [saved_model_name]_best.tar.gz where 
  [save_model_name] is an argument passed to classification.py, can untar this file 
  with the command: tar xvzf [saved_model_name]_best.tar.gz)
- should pass "[saved_model_name]" as argument to --load in classification.py
  during testing
- enter the following command to uncache github credentials after running code:
  git credential-cache exit



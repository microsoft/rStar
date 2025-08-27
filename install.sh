SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd -P)

cd $SCRIPT_DIR
git submodule init
git submodule update

# install verl
pip install "torch<2.8"
pip install -r verl/requirements_sglang.txt
pip install -e verl

# install code judge
pip install -r code-judge/requirements.txt
pip install -e code-judge

# install rstar2_agent
pip install -e .

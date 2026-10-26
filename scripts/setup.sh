# check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Miniforge3 or Miniconda and ensure it's in your PATH."
    exit 1
fi

# install venv for inference and benchmarking
conda create --name rr python=3.12 -y
conda activate rr
pip install -r requirements.txt

echo "Setup of virtual environment `rr` completed."

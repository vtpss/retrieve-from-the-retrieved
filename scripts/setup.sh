# check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Miniforge3 or Anaconda and ensure it's in your PATH."
    exit 1
fi

# install venv for inference and benchmarking
conda create --name benchmark python=3.12.12 -y
conda activate benchmark

pip install datasets dspy-ai 'litellm[proxy]' google-genai tiktoken rlms
pip install tabulate lxml python-dateutil rich matplotlib
pip install pymediawiki ddgs trafilatura markdownify func-timeout 
pip install rapidfuzz "bm25s[core]" PyStemmer "pylate[eval]" transformers==4.56.2
pip install flash-attn --no-build-isolation --no-cache-dir

git clone https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
pip install qwen_omni_utils faiss-cpu

# install openserp for web search. ddgs is not stable enough for scaling.
wget https://go.dev/dl/go1.26.3.linux-amd64.tar.gz # https://go.dev/dl/
tar -xzf go1.26.3.linux-amd64.tar.gz
export LOCAL_GO_PATH=$(pwd)
alias go="$LOCAL_GO_PATH/go/bin/go"
git clone https://github.com/karust/openserp.git
cd openserp
go build -o openserp .
# ./openserp serve -p $PORT: Run to serve

cd ..
echo "Setup of virtual environments completed."

# install openserp for web search. ddgs is not stable enough for scaling.
cd ..
echo "This script install go for Linux x86_64. If you are using other OS, please edit this script accordingly."
curl -O https://dl.google.com/go/go1.27.3.linux-amd64.tar.gz # https://go.dev/dl/
tar -xzf go1.27.3.linux-amd64.tar.gz
export LOCAL_GO_PATH=$(pwd)
alias go="$LOCAL_GO_PATH/go/bin/go"
git clone https://github.com/karust/openserp.git
cd openserp
git checkout 471e5ad7fe044c61ddf1401b6fed38675629a962
go build -o openserp .
# ./openserp serve -p $PORT: Run to serve

cd ../retrieve-from-the-retrieved

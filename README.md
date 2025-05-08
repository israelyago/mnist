# MNIST classification

## Data download

Can be downloaded at [Hugging Face](https://huggingface.co/datasets/ylecun/mnist)

## Running locally with conda/mamba

1. Create a new conda/mamba env: `mamba env create -n mnist -f environment.yaml`
1. Activate the environment with `mamba activate mnist`
1. Start the MLflow tracking server with `./local_mlflow_server.sh`

In a new terminal:

1. Activate the environment with `mamba activate mnist`
1. Run `python src/main.py` to debug or `python src/main.py -r` for release/full mode

## Running locally with container

First, run MLflow UI server with your container technology:

1. `docker build -t mlflow-server:local -f mlflow.Dockerfile .`
1. `docker run -d -p 5000:5000 mlflow-server:local`

Then, run the training code with:

1. `docker build -t mnist:local -f Dockerfile .`
1. `docker run -it --rm mnist:local sh`

## Development

When changing the dependencies of mamba:

1. Run: `mamba env export --from-history > environment.yaml`
1. Edit `environment.yaml` (Remove the prefix field and change the name field to "base")

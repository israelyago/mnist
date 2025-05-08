# MNIST classification

## Data download

Can be downloaded at [Hugging Face](https://huggingface.co/datasets/ylecun/mnist)

## Running locally with conda/mamba

1. Create a new conda/mamba env: `mamba env create -n mnist -f environment.yaml`
1. Activate the environment with `mamba activate mnist`
1. Run `python src/main.py` to debug or `python src/main.py -r` for release/full mode

## Development

When changing the dependencies of mamba:

1. Run: `mamba env export --from-history > environment.yaml`
1. Edit `environment.yaml` (Remove the prefix field and change the name field to "base")

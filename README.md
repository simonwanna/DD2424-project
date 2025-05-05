# DD2424-project - Transfer Learning

# Setup :rocket:

1) Clone repo: `git clone git@github.com:simonwanna/DD2424-project.git && cd DD2424-project` 
2) Create env: `conda env create -f environment.yaml && conda activate dd2424`
3) Get the data:

```bash
mkdir data && cd data
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
tar -xvzf images.tar.gz
tar -xvzf annotations.tar.gz
rm images.tar.gz annotations.tar.gz
```
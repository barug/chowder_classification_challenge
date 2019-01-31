# Owkin_challenge

## setup

```sh
python3 -m venv venv_owkin_challenge
source venv_owkin_challenge/bin/activate
pip install -r requirements.txt
```

## usage

```sh
python pytorch_implem.py --data_dir data/ --conf conf.yaml --save_mod_dir models/ --save_res_to results.p
```

options:
- data_dir : path of the data directory. required
- conf : path of the configuration file. required
- save_mod_dir : path of the directory where to save the models. optional
- save_res_to : path of the result save file. optional

# Owkin_challenge

## setup

```sh
python3 -m venv venv_owkin_challenge
source venv_owkin_challenge/bin/activate
pip install -r requirements.txt
```
to create an ipython kernel named owkin_challenge in your virtualenv, run from your virtualenv :
```sh
ipython kernel install --user --name=owkin_challenge
```

## usage

first run the model

```sh
python main.py --data_dir data/ --conf conf.yaml --save_mod_dir models/ --save_res_to results.p
```

options:
- data_dir : path of the data directory. required
- conf : path of the configuration file. required
- save_mod_dir : path of the directory where to save the models. optional
- save_res_to : path of the result save file. optional

After generating the result file, run the jupyter notebook named challenge_notebook.ipynb with the generated ipython kernel. This notebook contains a few cells that visualize the input and output data of the algorithm.

## conclusion

This implementation of the Chowder algorithm is fairly unstable. The average accuracy is of 0.6 and doesn't go much further than that. This value is also the ratio of positive to negative labels example in the dataset. This is why sometime the model is able to reach 0.6 accuracy by only classifying negative examples. When the model really does learn, it can also reach around 0.6 accuracy. I have not been able to detect the cause of these low performances with the time I had available to work on the project. Nonetheless, when using a good model that really did learned, looking at the scores computed for the tiles show some real differences, that when compared with the results of the Chowder paper, sometime seems to follow the same visual features as diseased or healthy tiles. This show that this implementation is able to differentiate defining features in the tiles and has the potential to be improved to be used for disease localization.

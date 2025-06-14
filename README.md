# Neural Operator Arena: a collection of most of the latest Neural Operators!

## Project structure:
### Data: 
in `./data/`, prepare the dataloaders of the datasets.

in `./scripts/datasets.py`, prepare the argument parser for the dataset.

Currently supported datasets: Burgers, Darcy, TorusLi, TorusVisForce

### Model: 
in ./models, prepare the models.

in `./scripts/models.py`, prepare the argument parser for the model.

Currently supported models: FNO, T-FNO, LSM

### Training task
in  `./scripts/`, you can define different settings of training, (e.g. `train.py`, `train_multitask.py`). for each setting, you may need some different arguments. set them in `./scripts/get_parser.py` 's `add_base_args()` function. In addition, you may customize some LightningModules and Callbacks in `./lightning_modules/`.

The process of each training scripts is:

First, in the training script, you should intialize a list of dataset parsers and model parsers that fits the kind of pipeline, say, $n$ datasets and $m$ models. Then the fetcher would create a parser of $ n\cdot m $ subparsers, correspondent to all the possible choices.

Don't forget to pass the type of task to the Fetcher().

Then one may customize his own training script concisely.

### Running configs
for linux environments, prepare your scripts in `./shell_scripts/`.

To extract logs, run:
```
tb-reducer runs/YOUR_FOLDER -o OUTPUT_PATH/BLABLABLA.csv --handle-dup-steps mean
```
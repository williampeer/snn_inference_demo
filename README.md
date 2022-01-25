# snn_inference

Currently a development/alpha-version.

## Usage
`python main.py -h`

See Models/ for models that are already implemented and their implementations.

Test/ for tests that you may want to run just to figure out how the Models can be used outside of the experiments context (or what happens under the hood).

exp_suite.py should give you a gist of what running an experiment does

### Sample usage

To run the first version of the framework integration, you may run `python main.py -h`, e.g.
```
python main.py -et SanityCheck -lr 0.05 -ti 10 -N 1 -bs 400 -tvr 10.0 -rpti 4000 -optim Adam -ipr 10.0 -mt LIF -lfn frd
```

Alternatively, for a much faster integration and experiment loop, please see:
```
python main_fast_SNN_GD_Synthetic_SGIF.py
```
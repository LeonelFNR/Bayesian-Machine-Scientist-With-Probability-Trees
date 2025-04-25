# Bayesian Machine Scientist

This repository contains the code of the paper:

Guimera, R, Reichardt, I, Aguilar-Mogas, A, Massucci, FA, Miranda, M, Pallares, J, Sales-Pardo, M. [A Bayesian machine scientist to aid in the solution of challenging scientific problems](http://dx.doi.org/10.1126/sciadv.aav6971), Sci. Adv. 6 (5) , eaav6971 (2020).

# Tutorial 

This tutorial illustrates how to program a Bayesian machine scientist, using the code provided here. The tutorial assumes general knowledge of Python programming. We start by importing all necessary Python modules:


```python
import sys
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from copy import deepcopy
from ipywidgets import IntProgress
from IPython.display import display

sys.path.append('./')
sys.path.append('./Prior/')
from mcmc import *
from parallel import *
from fit_prior import read_prior_par
```

## Loading and preparing the data 

We then load the data. In this particular case, we load the salmon stocks data. The features (independent variables) are loaded into a Pandas `DataFrame` named `x`, whereas the target (dependent) variable is loaded into a Pandas `Series` named `y`. Data should **always** be loaded in these formats to avoid problems. 


```python
XLABS = [
    'eff',
    'D_max',
    'D_apr',
    'D_may',
    'D_jun',
    'ET_apr',
    'ET_may',
    'ET_jun',
    'PT_apr',
    'PT_may',
    'PT_jun',
    'PT_jul',
    'PDO_win',
]
raw_data = pd.read_csv('Validation/LogYe/data/seymour.csv')
x, y = raw_data[XLABS], np.log(raw_data['rec'])
x.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eff</th>
      <th>D_max</th>
      <th>D_apr</th>
      <th>D_may</th>
      <th>D_jun</th>
      <th>ET_apr</th>
      <th>ET_may</th>
      <th>ET_jun</th>
      <th>PT_apr</th>
      <th>PT_may</th>
      <th>PT_jun</th>
      <th>PT_jul</th>
      <th>PDO_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.004697</td>
      <td>12500</td>
      <td>952</td>
      <td>4160</td>
      <td>8880</td>
      <td>7.8</td>
      <td>10.6</td>
      <td>14.5</td>
      <td>6.7</td>
      <td>7.3</td>
      <td>8.6</td>
      <td>9.7</td>
      <td>-1.544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.011504</td>
      <td>8040</td>
      <td>1650</td>
      <td>6040</td>
      <td>6020</td>
      <td>9.1</td>
      <td>12.4</td>
      <td>14.5</td>
      <td>7.2</td>
      <td>8.2</td>
      <td>8.9</td>
      <td>9.8</td>
      <td>-1.012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002780</td>
      <td>8330</td>
      <td>1700</td>
      <td>5670</td>
      <td>6790</td>
      <td>8.4</td>
      <td>11.4</td>
      <td>13.5</td>
      <td>7.1</td>
      <td>8.0</td>
      <td>8.6</td>
      <td>9.3</td>
      <td>-0.496</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002907</td>
      <td>7220</td>
      <td>920</td>
      <td>4960</td>
      <td>6020</td>
      <td>9.1</td>
      <td>12.2</td>
      <td>14.4</td>
      <td>7.6</td>
      <td>8.5</td>
      <td>9.1</td>
      <td>9.9</td>
      <td>-0.682</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.012463</td>
      <td>9060</td>
      <td>796</td>
      <td>4100</td>
      <td>7600</td>
      <td>8.4</td>
      <td>12.3</td>
      <td>13.2</td>
      <td>7.5</td>
      <td>8.3</td>
      <td>8.8</td>
      <td>9.2</td>
      <td>-0.472</td>
    </tr>
  </tbody>
</table>
</div>



## Initializing the Bayesian machine scienstist 

We start by initializing the machine scientist. This involves three steps:
- **Reading the prior hyperparameters.** The values of the hyperparameters depend on the number of variables `nv` and parameters `np`considered during the search. Many combinations of `nv` and `np` have hyperparameters calculated in the `Prior` directory. Otherwise, the hyperparameters should be fit. 
- **Setting the "temperatures" for the parallel tempering.** If you don't know what parallel tempering is, you can read it in the Methods section of the paper, or just leave it as is in the code. In general, more temperatures (here 20) lead to better sampling of the expression space (we use a maximum of 100 different temperatures)
- **Initializing the (parallel) scientist.**


```python
# Read the hyperparameters for the prior
prior_par = read_prior_par('./Prior/final_prior_param_sq.named_equations.nv13.np13.2016-09-01 17:05:57.196882.dat')

# Set the temperatures for the parallel tempering
Ts = [1] + [1.04**k for k in range(1, 20)]

# Initialize the parallel machine scientist
pms = Parallel(
    Ts,
    variables=XLABS,
    parameters=['a%d' % i for i in range(13)],
    x=x, y=y,
    prior_par=prior_par,
)
```

## Sampling expressions with the Bayesian machine scientist 

We are now ready to start sampling expressions with the Bayesian machine scientist, using MCMC. In its simplest form, one just needs to run the `mcmc_step()` and the `tree_swap()` methods as many times as necessary. `mcmc_step()` performs an MCMC update at each of the temperatures of the parallel tempering, whereas `tree_swap()` attempts to swap the expressions at two consecutive temperatures.


```python
# Number of MCMC steps
nstep = 100

# Draw a progress bar to keep track of the MCMC progress
f = IntProgress(min=0, max=nstep, description='Running:') # instantiate the bar
display(f)

# MCMC
for i in range(nstep):
    # MCMC update
    pms.mcmc_step() # MCMC step within each T
    pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
    # Update the progress bar
    f.value += 1
```


    IntProgress(value=0, description='Running:')


Typically, of course, one wants to do something other than just generate expressions. For example, one may want to keep track of the most plausible (or, equivalently, the minimum description length) model visited so far by the MCMC, or to keep a trace of some of the properties of the sampled expressions. The example below keeps the best model, as well as a trace of all the description lengths visited. Note that, in `Parallel` objects, the relevant expression is stored in the `t1` attribute (which stands for temperature 1).


```python
# Number of MCMC steps
nstep = 3000

# Draw a progress bar to keep track of the MCMC progress
f = IntProgress(min=0, max=nstep, description='Running:') # instantiate the bar
display(f)

# MCMC
description_lengths, mdl, mdl_model = [], np.inf, None
for i in range(nstep):
    # MCMC update
    pms.mcmc_step() # MCMC step within each T
    pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
    # Add the description length to the trace
    description_lengths.append(pms.t1.E)
    # Check if this is the MDL expression so far
    if pms.t1.E < mdl:
        mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)
    # Update the progress bar
    f.value += 1
```


    IntProgress(value=0, description='Running:', max=3000)


So let's take a look at the objects we stored. Here is the best model sampled by the machine scientist:


```python
print('Best model:\t', mdl_model)
print('Desc. length:\t', mdl)
```

    Best model:	 (log(eff) + _a3_)
    Desc. length:	 92.0026314728859


And here is the trace of the description length:


```python
plt.figure(figsize=(15, 5))
plt.plot(description_lengths)
plt.xlabel('MCMC step', fontsize=14)
plt.ylabel('Description length', fontsize=14)
plt.title('MDL model: $%s$' % mdl_model.latex())
plt.show()
```


    
![png](https://bitbucket.org/rguimera/machine-scientist/raw/1499d134ea5655db08c7750f605094c0be8aca52/Images/output_17_0.png)
    


## Making predictions with the Bayesian machine scientist 

Finally, we typically want to make predictions with models. In this regard, the interface of the machine scientist is similar to those in Scikit Learn: to make a prediction we call the `predict(x)` method, with an argument that has the same format as the training `x`, that is, a Pandas `DataFrame` with the exact same columns.


```python
plt.figure(figsize=(6, 6))
plt.scatter(mdl_model.predict(x), y)
plt.plot((-6, 0), (-6, 0))
plt.xlabel('MDL model predictions', fontsize=14)
plt.ylabel('Actual values', fontsize=14)
plt.show()
```


    
![png](https://bitbucket.org/rguimera/machine-scientist/raw/1499d134ea5655db08c7750f605094c0be8aca52/Images/output_20_0.png)
    


## Further capabilities of the Bayesian machine scientist

### Reading an expression into the machine scientist

Rather than sampling models, we can directly build a machine scientist from a given expression. This is useful when we want to evaluate models that have been previously saved, or to analyze specific predefined models. For this we do not use `Parallel` machine scientists but rather a single `Tree` machine scientist; we instantiate the `Tree` with the `from_string` keyword. Additionally, in these situations we typically want to set the values of the model parameters, because trying to fit them from scratch may fail. All in all, we proceed as follows:


```python
# Define the string (NOTE this MUST be in the same format as the code write models!)
model_string = "(log((eff / (D_apr * _a0_)))"
# Define parameter values
model_parameters = {
   '_a0_': 6.655292653177647e-05,
}

# Instantiate a Tree from the desired string
my_model = Tree(
    variables=XLABS,
    parameters=['a%d' % i for i in range(13)],
    x=x, y=y,
    prior_par=prior_par,
    from_string=model_string,
)

# Set the parameter values
my_model.set_par_values(model_parameters)
```

Now, this model works as any other model. For example, you can use it to make predictions:


```python
plt.figure(figsize=(6, 6))
plt.scatter(my_model.predict(x), y)
plt.plot((-6, 0), (-6, 0))
plt.xlabel('My predefined model predictions', fontsize=14)
plt.ylabel('Actual values', fontsize=14)
plt.show()
```


    
![png](https://bitbucket.org/rguimera/machine-scientist/raw/b1551cd0677df9668c31a515376aa20c6a95dc36/Images/output_25_0.png)
    


### Finding models for multiple data sets at the same time

The machine scientist can model multiple data sets at the same time, using the same models for all data sets, but allowing for different parameter values for each data set. This approach makes sense when one expects the mechanisms to be common accross data sets, but idiosyncrasies of each data set preclude from putting all data together. This approach has been used, for example, to [model human mobility across States within the U.S.](https://arxiv.org/abs/2312.11281), or to [model friction in turbulent flows across different experimental conditions](https://link.aps.org/doi/10.1103/PhysRevLett.124.084503).

To achieve this, and assuming that `x1` and `x2` are feature dataframes like `x` above, and that `y1` and `y2` are target dataframes like `y` above, we just need to build the following dictionaries:  


```python
xmulti = {'dataset1' : x1, 'dataset2' : x2}
ymulti = {'dataset1' : y1, 'dataset2' : y2}
```

and feed `x=xmulti` and `y=ymulti` as keyword arguments when instantiating the `Parallel` object, that is:


```python
# Initialize the parallel machine scientist with multiple data sets
pms = Parallel(
    Ts,
    variables=XLABS,
    parameters=['a%d' % i for i in range(13)],
    x=xmulti, y=ymulti,
    prior_par=prior_par,
)
```

If initialized in this way, `pms.t1.par_values` will automatically be a dictionary with the same keys as `xmulti` and `ymulti` (which, for obvious reasons, have to be the same!), and whose values will be the parameter values for each of the data sets.

### Fixing a term to the model

Sometimes, for whatever reason, part of the expression we are looking for is known. For example, we may know that the model we are looking for contains a given factor multiplied by something else that we do not know and wish to model. In such cases, it is possible to specify a fixed term to the model. For example, in the example we have been using in this tutorial, we may want to specify that

$$\log({\rm rec}) = c_1 * \log ({\rm eff}) * f({\rm features})$$

and limit our search to $f$. This is achieved by using the `fixed_term` and `fixed_term_op` keyword arguments at the time of the creation of the (single or parallel) machine scientist. 

There are several ways one may want to use the `fixed_term`, depending on whether the machine scientist needs to use or not in $f$ the features and parameters in the fixed term. If the fixed factor uses some features and parameters that can be used also in $f$, then it can be simply initialized as follows

```python
# Initialize the parallel machine scientist with an extra factor _a0_*log(eff) that uses
# parameters (_a0_) and features (eff)  that are also potentially used in the rest
# of the expression
pms = Parallel(
    Ts,
    variables=[xl for xl in XLABS],
    parameters=['a%d' % i for i in range(13)],
    x=x, y=y,
    prior_par=prior_par,
    fixed_term='(_a0_ * (log(eff)))', # Note that _a0_ is in parameters and eff is in variables
    fixed_term_op='*', # The fixed term will be multiplied to f
)
```

More often, however, we want the parameters and/or features in the fixed term to be excluded from $f$. If we want to use a parameter that cannot be used in $f$, it is enough to use a name for the parameter that is **not** included in `parameters`. (Note, however, that parameter names **must** start and end with `_`.) For example:


```python
# Initialize the parallel machine scientist with an extra factor _b0_*log(eff) that uses
# a *new* parameter (_b0_), and a feature (eff) that is potentially used in the rest
# of the expression
pms = Parallel(
    Ts,
    variables=[xl for xl in XLABS],
    parameters=['a%d' % i for i in range(13)],
    x=x, y=y,
    prior_par=prior_par,
    fixed_term='(_b0_ * (log(eff)))', # Note that _b0_ is NOT in parameters. An extra parameter,
                                      # not usable outside the factor, is created automatically
    fixed_term_op='*',
)
```

Finally, if we want the features in the prefactor **not** to be usable in the rest of the expression, we need to exclude them from `variables` and explicitly add them as a list using the `extra_variables` keyword argument. For example:


```python
# Initialize the parallel machine scientist with an extra factor _b0_*log(eff) that uses
# a *new* parameter (_b0_) and a *new* feature (eff) which will *not* appear in the rest
# of the expression
pms = Parallel(
    Ts,
    variables=[xl for xl in XLABS if xl != 'eff'], # exclude eff from the list of features
    parameters=['a%d' % i for i in range(13)],
    x=x, y=y,
    prior_par=prior_par,
    fixed_term='(_b0_ * (log(eff)))',
    fixed_term_op='*',
    extra_variables=['eff'], # add eff as an extra feature, only used in the fixed factor
)
```

In the last example, note that both `_b0_` and `eff` will only be used in the fixed term, but not in the rest of the expression. This is, perhaps, the most common situation. 

**Importantly**, note that the description lengths of models obtained with a fixed factor are not directly comparable to models obtained without restriction, as the operations in the fixed term are not taken into account in the calculation of the description length.

## Further refinements 

The examples above are only intended to illustrate how a basic MCMC would be implemented. In practice, there are other considerations that we kept in mind in all the experiments reported in the manuscriot, and that anyone using the code should too:

* **Equilibration**: One should not start sampling until the MCMC has converged to the stationary distribution. Although determining when a sample is in equilibrium, a necessary condition is that the description length is not increasing or, more typically, decreasing. The trace of the description length should be flat (except for fluctuations) before we start collecting samples.
* **Thinning**: MCMC samples should be thinned, so only one in, say, 100 samples are kept for the trace. Otherwise, one is getting highly correlated samples, which may lead to, for example, erroneous estimates of confidence intervals.
* **Getting trapped**: Despite the parallel tempering, the MCMC can get trapped in local minima of the description length. For this, we typically keep track of the number of steps since the last `tree_swap()` move was accepted for each temperature. If a particular temperature has *not* accepted swaps in a long time, then we anneal the whole system, that is, we increase all temperatures and decrease them slowly back to equilibrium so as to escape the local minima. Using several restarts of the MCMC and comparing the results is also a convenient check.
* **Memory issues**: By default, the machine scientist keeps a cache of all visited models, so as to avoid duplicates of previously considered models, as well as to speed up the process of obtaining the maximum likelihood estimators of the model parameters. For long MCMC chains this becomes memory intensive, so it may be convenient to periodically clean this cache (or, at least, old models in this cache) by reinitializing the `fit_pat` and `representative` attributes of the `Parallel` instance.

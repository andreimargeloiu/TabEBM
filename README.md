# TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models


### Content
1. `TabEBM.py` contains (i) the implementation of TabEBM, and (ii) a helper function `plot_TabEBM_energy_contour` to show the energy contour (or unnormalized probability) approximated by TabEBM
2. `TabEBM_approximated_density.ipynb` shows the TabEBM approximation of the density of the real data distribution
3. `TabEBM_generate_data.ipynb` shows how to generate data using TabEBM


### Installation
* Create `conda` environment
```
conda create -n tabebm python=3.10.12
conda activate tabebm
```

* Install dependencies
```
pip install --no-cache-dir -r requirements.txt
```

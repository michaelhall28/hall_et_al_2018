# clone-competition-simulation
Code to run simulations of clone competition in epithelia during ongoing mutagenesis.
Cells in a hexagonal grid acquire fitness-altering mutations and with compete with neighbouring cells to divide.
Tracks mutant clone sizes over time.


### Hall et al 2019
The notebooks can reproduce the figures published in Hall et al, "Relating evolutionary selection and mutant clonal dynamics in normal epithelia", Royal Society Interface, 2019.

Some of the original figures required many hours of simulation time. Some parameters in the notebooks have been changed so that the simulations run in a reasonable amount of time.
By adjusting the parameters (instructions in the notebooks) the original figures can be recreated.

`Multiple Simulation Figures.ipynb` contains code to recreate Figures 2 and 3.
`Single Simulation Figures.ipynb` produces Figure 5 and Figure 6a,b
`Oesophagus Data.ipynb` produces Figure 4 and Figure 6c,d
`Supplementary Simulation Figures.ipynb` produces Supplementary Figure 1


### Requirements
Python 3.6  
matplotlib >=2.0.0  
numpy >=1.12.1  
scipy >=0.19.1
seaborn >= 0.9.0
scikit-learn >= 0.20.2

# A-Model-Based-Approach-To-Assess-Epidemic-Risk
### Hugo Dolan, Riccardo Rastelli
## Abstract
We study how international flights can facilitate the spread of an
epidemic to a worldwide scale. We combine an infrastructure network of flight
connections with a population density dataset to derive the mobility network,
and then define an epidemic framework to model the spread of disease. Our approach combines a compartmental SEIRS model with a graph diffusion model
to capture the clusteredness of the distribution of the population. The resulting model is characterised by the dynamics of a metapopulation SEIRS, with
amplification or reduction of the infection rate which is determined by the
mobility of individuals. Then, we use simulations to characterise and study
a variety of realistic scenarios that resemble the recent spread of COVID-19.
Crucially, we define a formal framework that can be used to design epidemic
mitigation strategies: we propose an optimisation approach based on genetic
algorithms that can be used to identify an optimal airport closure strategy,
and that can be employed to aid decision making for the mitigation of the
epidemic, in a timely manner.

## About this repository
This repository contains copies of the original datasets used in this project in the "/datasets" folder.
It also contains 2 python files, which contain the entire simulation environment (environment.py) and also a script for
percolating networks by various factors (percolation.py).
Additionally there are 3 notebooks which contain examples of how this code is used in practice as well as the majority of
the data processing pipepline, many experimental results and our genetic algorithm procedure.
This repository is accompanied by the paper which is found in pdf form and also can be found at https://arxiv.org/abs/2009.12964

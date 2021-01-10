# SIRD Model

This repo contains code which was used for simulating a SIR/SIRD model, as part of my SRE (research project) under [Prof. Sharayu Moharir](https://sites.google.com/site/sharayumoharir/).

Please consult the [report](https://methi1999.github.io/assets/pdf/sre_sird.pdf) for more details.

# Directory Structure

* config.yaml: store hyperparameters
* graph.py: Returns a graph model which is used as an underlying network topology.
* sir_simple.py: Use a random-mixing model for predicting the evolution of the system.
* sird.py: Predict the evolution of the system with an underlying network topology.
* plot.py: Plot the evolution of the system.

Subdirectories *img/* and *pickle/* are created for storing the plots, gifs and intermediate results.



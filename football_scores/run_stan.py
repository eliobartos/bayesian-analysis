import numpy as np
import stan
import pickle

def main():

    with open('model.stan', 'r') as f:
        model = f.read()

    data = {
        "N": 5,
        "M": 3,
        "total_players": 7,
        "y": [0, 1, 2, -1, -2],
        "team1": [[1, 2, 3], [1, 2, 7], [1, 4, 5], [7, 2, 3], [5, 2, 3]],
        "team2": [[4, 5, 6], [4, 5, 6], [2, 7, 3], [1, 5, 6], [4, 1, 6]],
        "p_skills": [5, 4],
        "p_beta": 1.0,
        "p_nu": 1.0/29,
        "p_theta": 1.0/0.2
    }

    posterior = stan.build(model, data=data, random_seed=1)
    fit = posterior.sample(num_chains=3, num_samples=10000)

    with open('fit.pckl', 'wb') as f:
        pickle.dump(fit, f)

if __name__ == '__main__':
    main()
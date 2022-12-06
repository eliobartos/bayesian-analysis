import numpy as np
import stan
import pickle

def main():

    with open('model.stan', 'r') as f:
        model = f.read()

    data = {
        "N": 2,
        "M": 6,
        #"winners": [[1, 2], [1, 3], [1, 4]],
        #"losers": [[3, 4], [2, 4], [2, 3]]
        "winners": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]],
        "losers": [[7, 8, 9, 10, 11, 12], [4, 5, 6, 10, 11, 12]]
    }

    posterior = stan.build(model, data=data, random_seed=1)
    fit = posterior.sample(num_chains=3, num_samples=10000)

    with open('fit.pckl', 'wb') as f:
        pickle.dump(fit, f)

if __name__ == '__main__':
    main()
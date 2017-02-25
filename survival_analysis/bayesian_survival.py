import numpy as np
import pymc3 as pm
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels import datasets
from theano import tensor as T


# Modifiying this code: http://austinrochford.com/posts/2015-10-05-bayes-survival.html


def load_data():
    data = datasets.get_rdataset('mastectomy', 'HSAUR', cache=True).data
    data.event = data.event.astype(np.int64)
    data.metastized = (data.metastized == 'yes').astype(np.int64)

    n_patients = data.shape[0]
    patients = np.arange(n_patients)

    print("data:{}, patients:{}, number_patients:{}".format(data.head(), patients, n_patients))
    print("censored observations{}".format(1 - data.event.mean()))

    return data, patients


def visualize_data(data, patients):
    n_patients = len(patients)

    # The column metastized represents whether the cancer had metastized prior to surgery.
    fig, ax = plt.subplots(figsize=(8, 6))
    blue, _, red = sns.color_palette()[:3]
    ax.hlines(patients[data.event.values == 0], 0, data[data.event.values == 0].time,
              color=blue, label='Censored')
    ax.hlines(patients[data.event.values == 1], 0, data[data.event.values == 1].time,
              color=red, label='Uncensored')
    ax.scatter(data[data.metastized.values == 1].time, patients[data.metastized.values == 1],
               color='k', zorder=10, label='Metastized')
    ax.set_xlim(left=0)
    ax.set_xlabel('Months since mastectomy')
    ax.set_ylim(-0.25, n_patients + 0.25)
    ax.legend(loc='center right')
    plt.show()

    interval_bounds, intervals, _ = create_intervals(data)
    print("intervals:{}, interval_bounds:{}".format(intervals, interval_bounds))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data[data.event == 1].time.values, bins=interval_bounds,
            color=red, alpha=0.5, lw=0,
            label='Uncensored')
    ax.hist(data[data.event == 0].time.values, bins=interval_bounds,
            color=blue, alpha=0.5, lw=0,
            label='Censored')
    ax.set_xlim(0, interval_bounds[-1])
    ax.set_xlabel('Months since mastectomy')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel('Number of observations')
    ax.legend()
    plt.show()


def create_intervals(data):
    interval_length = 3
    interval_bounds = np.arange(0, data.time.max() + interval_length + 1, interval_length)
    n_intervals = interval_bounds.size - 1
    intervals = np.arange(n_intervals)
    return interval_bounds, intervals, interval_length


def build_model(data, patients):
    interval_bounds, intervals, interval_length = create_intervals(data)
    n_intervals = len(intervals)

    last_period = np.floor((data.time - 0.01) / interval_length)
    print("data time:{}, last_period:{}, n_intervals:{}".format(data.time, last_period, n_intervals))

    # define dij = {1, 0} to define if patient i died in the interval
    death = np.zeros((len(patients), n_intervals))
    death[patients, last_period] = data.event

    # ti,j to be the amount of time the i-th subject was at risk in the j-th interval.
    exposure = np.greater_equal.outer(data.time, interval_bounds[:-1]) * interval_length
    exposure[patients, last_period] = data.time - interval_bounds[last_period]
    # print("death:{}, exposure:{}, last_period:{}".format(death[0:5, :], exposure[0:5, :], last_period[0:5]))


    # the risk incurred by the i-th subject in the j-th interval as λi,j=λjexp⁡(xiβ).
    # approximate di,j with a Possion random variable with mean ti,j λi,j

    trace = mcmc(data, death, exposure, n_intervals)
    base_hazard = trace['lambda0']
    met_hazard = trace['lambda0'] * np.exp(np.atleast_2d(trace['beta']).T)

    plot_harzard(base_hazard, interval_bounds, met_hazard, data)


def plot_harzard(base_hazard, interval_bounds, met_hazard, data):
    blue, _, red = sns.color_palette()[:3]
    fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))
    plot_with_hpd(interval_bounds[:-1], base_hazard, cum_hazard,
                  hazard_ax, color=blue, label='Had not metastized')
    plot_with_hpd(interval_bounds[:-1], met_hazard, cum_hazard,
                  hazard_ax, color=red, label='Metastized')
    hazard_ax.set_xlim(0, data.time.max())
    hazard_ax.set_xlabel('Months since mastectomy')
    hazard_ax.set_ylabel(r'Cumulative hazard $\Lambda(t)$')
    hazard_ax.legend(loc=2)
    plot_with_hpd(interval_bounds[:-1], base_hazard, survival,
                  surv_ax, color=blue)
    plot_with_hpd(interval_bounds[:-1], met_hazard, survival,
                  surv_ax, color=red)
    surv_ax.set_xlim(0, data.time.max())
    surv_ax.set_xlabel('Months since mastectomy')
    surv_ax.set_ylabel('Survival function $S(t)$')
    fig.suptitle('Bayesian survival model')
    plt.show()


def plot_with_hpd(x, hazard, f, ax, color=None, label=None, alpha=0.05):
    mean = f(hazard.mean(axis=0))

    percentiles = 100 * np.array([alpha / 2., 1. - alpha / 2.])
    hpd = np.percentile(f(hazard), percentiles, axis=0)

    ax.fill_between(x, hpd[0], hpd[1], color=color, alpha=0.25)
    ax.step(x, mean, color=color, label=label)


def cum_hazard(hazard):
    return (3 * hazard).cumsum(axis=-1)


def survival(hazard):
    return np.exp(-cum_hazard(hazard))


def mcmc(data, death, exposure, n_intervals):
    SEED = 5078864
    with pm.Model() as model:
        lambda0 = pm.Gamma('lambda0', 0.01, 0.01, shape=n_intervals)

        sigma = pm.Uniform('sigma', 0., 10.)
        tau = pm.Deterministic('tau', sigma ** -2)
        mu_beta = pm.Normal('mu_beta', 0., 10 ** -2)

        beta = pm.Normal('beta', mu_beta, tau)

        lambda_ = pm.Deterministic('lambda_', T.outer(T.exp(beta * data.metastized), lambda0))
        mu = pm.Deterministic('mu', exposure * lambda_)

        obs = pm.Poisson('obs', mu, observed=death)

        n_samples = 40000
        burn = 20000
        thin = 20

        with model:
            step = pm.Metropolis()
            trace_ = pm.sample(n_samples, step, random_seed=SEED)
            trace = trace_[burn::thin]
            print("trace:{}".format(trace))
            print("beta mean:{}".format(np.exp(trace['beta'].mean())))

            pm.traceplot(trace, ['beta'])
            pm.autocorrplot(trace, ['beta'])

    return trace


if __name__ == '__main__':
    df, pat = load_data()
    visualize_data(df, pat)
    build_model(df, pat)

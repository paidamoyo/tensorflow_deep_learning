import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels import datasets


def load_data():
    data = datasets.get_rdataset('bladder', 'survival', cache=True).data
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


if __name__ == '__main__':
    df, pat = load_data()
    # visualize_data(df, pat)
    build_model(df, pat)

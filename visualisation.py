from keras.layers import Add
from matplotlib import cm


def categoryVsSample(train):
    df_grouped = train.groupby(['label', 'manually_verified']).count().drop(['freesound_id', 'license'], axis=1)

    cmap = cm.get_cmap('viridis')

    plot = df_grouped.unstack().reindex(df_grouped.unstack().sum(axis=1).sort_values().index) \
        .plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16, 9), colormap=cmap)

    plot.legend(['Unverified', 'Verified']);
    plot.set_xlabel("Category")
    plot.set_ylabel("No. of Samples")

    plot.figure.savefig("figures/Category_vs_No.samples.png", bbox_inches="tight", dpi=100)
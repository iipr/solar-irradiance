import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn import linear_model

class Plotter():

    def __init__(self, lw=1.75, # linewidth
                 title_fs=30, tick_fs=20, legend_fs=25, # fontsize
                ):
        self.title_fs = title_fs
        self.tick_fs = tick_fs
        self.legend_fs = legend_fs
        self.lw = lw

    def plot_series(self, series_d, labels, dims=None,
                    date_ticker=None, int_ticker=False,
                    style='-', grid=True, scale=1, out_path='tmp.png'):
        if not dims:
            dims = (len(series_d), 1)
        fig, axs = plt.subplots(*dims)
        axs = axs.ravel() if dims != (1, 1) else [axs]
        fig.set_size_inches(15, 5 * (dims[0] if dims[0] > 1 else 2))
        for idx, (title, sub_d) in enumerate(series_d.items()):
            for s_name, series in sub_d.items():
                if date_ticker is not None:
                    axs[idx].plot(date_ticker, series, style, linewidth=self.lw, label=s_name)
                else:
                    axs[idx].plot(series, style, linewidth=self.lw, label=s_name)
            self.__format_plot(axs[idx], title, labels, grid, int_ticker=int_ticker, 
                               date_ticker=date_ticker, scale=scale)
        fig.tight_layout()
        # Save to file
        self.__save_plot(fig, out_path)

    def plot_scatters(self, series_d, labels, min_max=None, dims=None, 
                      style='g.', grid=True, scale=1, out_path='tmp.png'):
        if not dims:
            dims = (len(series_d), 1)
        fig, axs = plt.subplots(*dims)
        axs = axs.ravel() if dims != (1, 1) else [axs]
        fig.set_size_inches(15, 5 * (dims[0] if dims[0] > 1 else 2))
        for idx, (title, sub_d) in enumerate(series_d.items()):
            (s1_name, s1), (s2_name, s2) = sub_d.items()
            axs[idx].plot(s1, s2, style, label='{} vs {}'.format(s1_name, s2_name))
            if min_max:
                # Create linear regression object
                regr = linear_model.LinearRegression()
                # Train the model using s2
                regr.fit(X=s2.reshape(-1, 1), y=s1)
                # Make predictions using the testing set
                line = regr.predict(s2.reshape(-1, 1))
                axs[idx].plot(s2, line, color='royalblue', label='LR of {} vs {}'.format(s1_name, s2_name))
                axs[idx].plot(min_max, min_max, color='r', label='y(x) = x')
            self.__format_plot(axs[idx], title, labels, grid, min_max=min_max, scale=scale)
        if len(axs) >= len(series_d):
            axs[-1].set_axis_off()
        fig.tight_layout()
        # Save to file
        self.__save_plot(fig, out_path)

    def plot_boxplot(self, data, x_labels, labels, title,
                     flierprops={'markerfacecolor':'g', 'marker': 'o'},
                     grid=True, scale=1, out_path='tmp.png'):
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        ax.boxplot(data, labels=x_labels, flierprops=flierprops)
        self.__format_plot(ax, title, labels, grid,
                           legend=False, scale=scale)
        # Save to file
        self.__save_plot(fig, out_path)

    def __format_plot(self, ax, title, labels, grid, legend=True,
                      int_ticker=False, date_ticker=None,
                      min_max=False, loc='best', scale=1):
        ax.set_title(title, fontsize=int(self.title_fs * scale))
        ax.set_xlabel(labels[0], fontsize=int(self.title_fs * scale))
        ax.set_ylabel(labels[1], fontsize=int(self.title_fs * scale))
        ax.xaxis.set_tick_params(labelsize=int(self.tick_fs * scale))
        ax.yaxis.set_tick_params(labelsize=int(self.tick_fs * scale))
        if legend:
            ax.legend(loc=loc, fontsize=int(self.legend_fs * scale))
        if grid:
            ax.grid()
        if int_ticker:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x+1)))
        if date_ticker is not None:
            #ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(6,21,2)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        if min_max:
            ax.set_aspect('equal')

    def __save_plot(self, fig, out_path):
        fig.savefig(out_path, bbox_inches='tight')
        plt.close()
        print('\n\tPlot saved to {}'.format(out_path))

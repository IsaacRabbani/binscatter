"""Monkey-patch Matplotlib to add an 'ax.binscatter' method."""
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
import numpy as np
import numpy.typing as npt

def _get_bins(n_elements: int, n_bins: int) -> List[slice]:
    bin_edges = np.linspace(0, n_elements, n_bins + 1).astype(int)
    bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    return bins


def get_binscatter_objects(
    x: pd.Series,
    y: pd.Series,
    controls: Union[pd.Series, pd.DataFrame] = None,
    weights: pd.Series = None,
    plot_fit = False,
    degree = 1,
    n_bins = 20,
    recenter_x = True,
    recenter_y = True,
    bins = None
):

    x = np.asarray(x)
    y = np.asarray(y)
    if weights is None:
        weights = [1] * len(x)
    weights = np.asarray(weights)

    if controls is None:
        if np.any(np.diff(x) < 0):
            argsort = np.argsort(x)
            x = x[argsort]
            y = y[argsort]
            weights = weights[argsort]
        x_data = x
        y_data = y
    else:
        if np.ndim(controls) == 1:
            controls = np.expand_dims(controls, 1)

        demeaning_y_reg = linear_model.LinearRegression().fit(controls, y, weights)
        y_data = y - demeaning_y_reg.predict(controls)

        demeaning_x_reg = linear_model.LinearRegression().fit(controls, x, weights)
        x_data = x - demeaning_x_reg.predict(controls)
        argsort = np.argsort(x_data)
        x_data = x_data[argsort]
        y_data = y_data[argsort]
        weights = weights[argsort]

        if recenter_y:
            y_data += np.average(y, weights = weights)
        if recenter_x:
            x_data += np.average(x, weights = weights)

    # this shouldn't be necessary
    if x_data.ndim == 1:
        x_data = x_data[:, None]
    if y_data.ndim == 1:
        y_data = y_data[:, None]
    if weights.ndim == 1:
        wts_reshape = weights[:, None]
    
    if bins is None:
        bins = get_bins(len(y), n_bins)

    x_means = [np.average(x_data[bin_i], weights = wts_reshape[bin_i]) for bin_i in bins]
    y_means = [np.average(y_data[bin_i], weights = wts_reshape[bin_i]) for bin_i in bins]

    if plot_fit:
        
        X_poly = PolynomialFeatures(degree = degree).fit_transform(x_data)
        if controls is None:
            X_mat = X_poly
        else:
            X_mat = np.concatenate((X_poly, controls), axis = 1)
        
        reg = linear_model.LinearRegression().fit(X_mat, y_data, weights)
        reg_coefs = [reg.intercept_[0]] + reg.coef_[0, 1:degree+1].tolist()
        
        x_curve = np.linspace(min(x_means), max(x_means))
        y_curve = np.sum([reg_coefs[n] * np.power(x_curve, n) for n in range(degree + 1)], axis = 0)
        
        return {
            'x_means' : x_means,
            'y_means' : y_means,
            'reg_coefs' : reg_coefs,
            'x_curve' : x_curve,
            'y_curve' : y_curve
        }

    else:
        return {
            'x_means' : x_means,
            'y_means' : y_means,
        }


def binscatter(
    self,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    controls=None,
    n_bins=20,
    line_kwargs: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
    recenter_x: bool = False,
    recenter_y: bool = True,
    # TODO: make 'bins' consistent with functions in other libraries, as in pd.cut
    bins: Optional[Iterable[slice]] = None,
    fit_reg: Optional[bool] = True,
) -> Tuple[List[float], List[float], float, float]:
    """
    :param self: matplotlib.axes.Axes object.
        i.e., fig, axes = plt.subplots(3)
              axes[0].binscatter(x, y)

    :param y: Numpy ArrayLike, such as numpy.ndarray or pandas.Series; must be 1d
    :param x: Numpy ArrayLike, such as numpy.ndarray or pandas.Series
    :param controls: Optional, {array-like, sparse matrix}; whatever can be passed to
        sklearn.linear_model.LinearRegression
    :param n_bins: int, default 20
    :param line_kwargs: keyword arguments passed to the line in the
    :param scatter_kwargs: dict
    :param recenter_y: If true, recenter y-tilde so its mean is the mean of y
    :param recenter_x: If true, recenter y-tilde so its mean is the mean of y
    :param bins: Indices of each bin. By default, if you leave 'bins' as None,
        binscatter constructs equal sized bins;
        if you don't like that, use this parameter to construct your own.
    :param fit_reg: Whether to plot a regression line.
    """
    if line_kwargs is None:
        line_kwargs = {}
    elif not fit_reg:
        warnings.warn("Both fit_reg=False and non-None line_kwargs were passed.")
    if scatter_kwargs is None:
        scatter_kwargs = {}

    x_means, y_means, intercept, coef = get_binscatter_objects(
        np.asarray(y), np.asarray(x), controls, n_bins, recenter_x, recenter_y, bins
    )

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    if fit_reg:
        self.plot(
            x_range,
            intercept + x_range * coef,
            label="beta=" + str(round(coef, 3)),
            **line_kwargs
        )
    # If series were passed, might be able to label
    if hasattr(x, "name"):
        self.set_xlabel(x.name)
    if hasattr(y, "name"):
        self.set_ylabel(y.name)
    return x_means, y_means, intercept, coef


matplotlib.axes.Axes.binscatter = binscatter

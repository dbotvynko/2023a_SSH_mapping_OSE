import torch
from datetime import timedelta
from glob import glob
import numpy as np
import datetime
import xarray as xr
import os
import pandas as pd
import logging
import pyinterp

def restrict_time_alongtrack(time_alongtrack, time_rec, days_offset=0.5):
    # Define the allowed timedelta in seconds
    allowed_timedelta_seconds = timedelta(days=days_offset).total_seconds()

    # Convert numpy.datetime64 to seconds since epoch
    time_alongtrack_seconds = time_alongtrack.astype('int64') // 10**9
    time_rec_seconds = time_rec.astype('int64') // 10**9

    # Move data to GPU as torch tensors
    time_alongtrack_tensor = torch.tensor(time_alongtrack_seconds, dtype=torch.float32)
    time_rec_tensor = torch.tensor(time_rec_seconds, dtype=torch.float32)

    # Expand tensors to compare all combinations
    time_alongtrack_expanded = time_alongtrack_tensor.unsqueeze(1)  # Shape: (N, 1)
    time_rec_expanded = time_rec_tensor.unsqueeze(0)  # Shape: (1, M)

    # Compute the time differences (broadcasting for all combinations)
    time_differences = time_rec_expanded - time_alongtrack_expanded  # Shape: (N, M)

    # Apply the allowed timedelta mask
    mask = (time_differences.abs() <= allowed_timedelta_seconds)  # Shape: (N, M)

    # Reduce along the time_rec dimension to check if any match exists
    valid_mask = mask.any(dim=1)  # Shape: (N,)

    # Filter time_alongtrack using the mask
    filtered_datetimes = time_alongtrack[valid_mask.numpy()]

    return filtered_datetimes

class TimeSeries:
    """
    Manage a time series composed of a grid stack.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the time series data.

    Attributes
    ----------
    ds : xarray.Dataset
        Input dataset containing the time series data.
    series : pandas.Series
        Time series data loaded from the dataset.
    dt : datetime.timedelta
        Time step duration between consecutive data points in the series.

    Methods
    -------
    _is_sorted(array)
        Check if an array is sorted.
    _load_ts()
        Load the time series data into memory.
    _load_dataset(self, varname, start, end)
        Loading the time series into memory for the defined period.
    """

    def __init__(self, ds):
        """
        Initialize a TimeSeries object.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing the time series data.
        """
        
        self.ds = ds
        self.series, self.dt = self._load_ts()

    @staticmethod
    def _is_sorted(array):
        """
        Check if an array is sorted.

        Parameters
        ----------
        array : numpy.ndarray
            Input array to check.

        Returns
        -------
        bool
            True if the array is sorted, False otherwise.
        """
        
        indices = np.argsort(array)
        return np.all(indices == np.arange(len(indices)))

    def _load_ts(self):
        """
        Load the time series data into memory.

        Returns
        -------
        pandas.Series
            Loaded time series data.
        datetime.timedelta
            Time step duration between consecutive data points in the series.
        """
        time = self.ds.time
        assert self._is_sorted(time)

        series = pd.Series(time)
        frequency = set(
            np.diff(series.values.astype("datetime64[s]")).astype("int64"))
        if len(frequency) != 1:
            raise RuntimeError(
                "Time series does not have a constant step between two "
                f"grids: {frequency} seconds")
        #return series, datetime.timedelta(seconds=float(frequency.pop()))
        return series, timedelta(seconds=float(frequency.pop()))

    def _load_dataset(self, varname, start, end):
        """
        Loading the time series into memory for the defined period.
        
        Parameters
        ----------
        varname: str
            Name of the variable to be loaded into memory.
        start: datetime.datetime
               Date of the first map to be loaded.
        end: datetime.datetime
               Date of the last map to be loaded.
        
        Returns
        -------  
        pyinterp.backends.xarray.Grid3D: 
                The interpolator handling the interpolation of the grid series.
        """
        
        if start < self.series.min():
            start = self.series.min()
        if end > self.series.max():
            end = self.series.max()
        
        #if start < self.series.min() or end > self.series.max():
        #    raise IndexError(
        #        f"period [{start}, {end}] out of range [{self.series.min()}, "
        #        f"{self.series.max()}]")
            
        first = start - self.dt
        last = end + self.dt

        selected = self.series[(self.series >= first) & (self.series < last)]
        logging.info("fetch data from %s to %s",selected.min(), selected.max())

        data_array = self.ds[varname].isel(time=selected.index)
        return pyinterp.backends.xarray.Grid3D(data_array, increasing_axes=True)
    
    
def periods(df, time_series, var_name="sla_unfiltered", frequency='W'):
    """
    Return the list of periods covering the time series loaded in memory.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time series data.
    time_series : TimeSeries
        Time series data and properties.
    var_name : str, optional
        Name of the variable to consider, by default "sla_unfiltered".
    frequency : str, optional
        Frequency for period grouping, by default 'W' (weekly).

    Yields
    ------
    tuple
        A tuple containing the start and end timestamps of each period.
    """
    period_start = df.groupby(
        df.index.to_period(frequency))[var_name].count().index

    for start, end in zip(period_start, period_start[1:]):
        start = start.to_timestamp()
        if start < time_series.series[0]:
            start = time_series.series[0]
        end = end.to_timestamp()
        yield start, end
    yield end, df.index[-1] + time_series.dt

def interpolate_plus(df, time_series, start, end, var='ssh', out_var='ssh_interpolated', forecast_interval=False, **kwargs):
    """
    Interpolate the time series over the defined period.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time series data.
    time_series : TimeSeries
        Time series data and properties.
    start : pandas.Timestamp
        Start timestamp of the interpolation period.
    end : pandas.Timestamp
        End timestamp of the interpolation period.
    """

    additionnal_args = {}
    if forecast_interval:
        additionnal_args['z_method'] = 'nearest'
    if 'num_threads' in kwargs:
        num_threads = kwargs.pop('num_threads')
    else:
        num_threads=0

    interpolator = time_series._load_dataset(var, start, end)
    mask = (df.index >= start) & (df.index < end)
    selected = df.loc[mask, ["longitude", "latitude"]]
    df.loc[mask, [out_var]] = interpolator.trivariate(
        dict(longitude=selected["longitude"].values,
             latitude=selected["latitude"].values,
             time=selected.index.values),
        #interpolator="inverse_distance_weighting",
        interpolator="bilinear",
        **additionnal_args,
        num_threads=num_threads)
    
    
def run_interpolation_plus(ds_maps, ds_alongtrack, frequency='M', var_alongtrack='ssh', var_rec='ssh', out_var='ssh_interpolated', **kwargs):
    """
    Interpolate time series data over specified periods.

    Parameters
    ----------
    ds_maps : xarray.Dataset
        Input dataset containing maps data.
    ds_alongtrack : xarray.Dataset
        Input dataset containing along-track data.
    frequency : str, optional
        Frequency for period grouping, by default 'M' (monthly).

    Returns
    -------
    xarray.Dataset
        Interpolated dataset.
    """
    
    time_series = TimeSeries(ds_maps)
    
    df = ds_alongtrack.to_dataframe()

    for start, end in periods(df, time_series, frequency=frequency, var_name=var_alongtrack):
        interpolate_plus(df, time_series, start, end, var_rec, out_var, **kwargs)
        
    ds = df.to_xarray()
        
    return ds

def get_preprocessed_rec_mercator_forecast(folders_glob_pattern, leadtime_index=0):
    forecast_folders = glob(folders_glob_pattern)

    lat = np.linspace(-80, 90, 2041)
    lon = np.linspace(-180, 180, 4320)

    data_arrays = []

    for forecast_folder in forecast_folders:
        # folder name of pattern R%year%month%day
        R_folder_name = forecast_folder.split('/')[-2]
        R_year = R_folder_name[1:5]
        R_month = R_folder_name[5:7]
        R_day = R_folder_name[7:9]
        date = datetime.datetime.strptime(R_year+R_month+R_day, '%Y%m%d')
        date_plus_x = date + datetime.timedelta(days=leadtime_index, hours=12)

        R_date_plus_x = datetime.datetime.strftime(date_plus_x, '%Y%m%d')
        
        ds = xr.open_dataset(os.path.join(forecast_folder, 'glo12_rg_1d-m_'+R_date_plus_x+'-'+R_date_plus_x+'_fcst_'+R_folder_name+'.nc'))
        da = ds.isel(time=0)['zos']

        da = da.assign_coords({'latitude': lat, 'longitude': lon, 'time':pd.to_datetime(date_plus_x.strftime('%Y%m%d'), format='%Y%m%d')}).expand_dims('time')
        data_arrays.append(da)

    concat_da = xr.concat(data_arrays, dim='time').sortby('time')
    concat_ds = concat_da.to_dataset(name='zos')
    leadtime = concat_ds.rename({'zos':'out', 'latitude':'lat', 'longitude':'lon'})
    return leadtime
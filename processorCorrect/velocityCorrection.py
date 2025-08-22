"""
Contains the staggered PRT and dual PRF correction method.

Both methods can be called through the errorCorrect function.

Requires scipy, numpy, datetime, and sys.
"""

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
from scipy.stats import norm
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def smooth_vel(vel, nyq, spatial, filter_size):
    """
    Smooth velocity field and compute difference from raw field.
    
    Parameters
    ----------
    vel : np.ndarray
        The observed Doppler velocity field
    nyq : float
        Nyquist velocity
    spatial : np.ndarray
        Either the radius or azimuth data associated with the shape of vel
    filter_size : int
        The radial or azimuthal filter window
    
    Returns
    -------
    tuple
        Smoothed velocity field and difference from observed velocity field
    """
    # Pre-compute complex exponential
    complex_vel = np.ma.exp(1j * vel * np.pi / nyq)
    
    # Linear interpolation along radial - vectorized where possible
    vel_interp = np.empty(vel.shape, dtype=complex)
    
    for i, ray in enumerate(complex_vel):
        valid_mask = ~ray.mask
        valid_count = valid_mask.sum()
        
        if valid_count > 5:
            spatial_valid = spatial[valid_mask]
            values = ray.data[valid_mask]
            order = np.argsort(spatial_valid)
            vel_interp[i] = np.interp(spatial, spatial_valid[order], values[order])
        else:
            vel_interp[i] = ray

    # Smooth interpolated field - combine real and imaginary operations
    vel_smooth_complex = savgol_filter(vel_interp.real, filter_size, 3, mode='interp', axis=1) + \
                        1j * savgol_filter(vel_interp.imag, filter_size, 3, mode='interp', axis=1)
    
    vel_smooth = np.angle(vel_smooth_complex) * nyq / np.pi
    
    # Replace invalid values with NaN
    vel_smooth[vel_smooth < -1000] = np.nan
    
    # Create difference field
    diff = vel.filled(fill_value=np.nan) - vel_smooth
    
    return vel_smooth, diff


def _validate_filters(radial_filters, azimuthal_filters, method):
    """Validate and return filter configurations."""
    if radial_filters is not None and azimuthal_filters is not None:
        if len(radial_filters) != len(azimuthal_filters):
            raise ValueError("The lengths of the radial and azimuthal filter lists must be the same.")
        return list(zip(radial_filters, azimuthal_filters))
    
    # Default filters based on method
    if method == 'staggered':
        return [(11, 5), (21, 9), (5, 5), (51, 21), (71, 71), (5, 5)]
    else:  # dual
        return [(71, 71), (11, 5), (21, 9), (5, 5), (51, 21), (71, 71), (5, 5)]


def _compute_possible_solutions_staggered(vel_new, vel_smooth_mean, diff_az, diff_rad, diff_mean, 
                                         nyq_l, nyq_h, fnyq, mask):
    """Compute possible solutions for staggered PRT correction."""
    shape = vel_new.shape
    possible_solutions = np.empty((16, *shape))
    possible_solutions[0] = vel_new.copy()
    
    count = 1
    # Pre-compute error conditions
    valid_mask = ~(np.isnan(diff_az) | np.isnan(diff_rad))
    
    for n1 in range(4):
        for n2 in range(4):
            if n1 == 0 and n2 == 0:
                continue
                
            nyq = n1 * nyq_l + n2 * nyq_h
            bound = fnyq
            limit = max(0, nyq - bound)
            
            vel_possible = vel_new.copy()
            
            # Vectorized error detection and correction
            pos_mask = valid_mask & (diff_mean > limit) & (diff_mean < nyq + bound)
            neg_mask = valid_mask & (diff_mean < -limit) & (diff_mean > -(nyq + bound))
            
            vel_possible[pos_mask] -= nyq
            vel_possible[neg_mask] += nyq
            mask[pos_mask | neg_mask] = 1
            
            possible_solutions[count] = vel_possible
            count += 1
    
    return possible_solutions, mask


def _compute_possible_solutions_dual(vel_new, vel_smooth_mean, diff_az, diff_rad, diff_mean,
                                    nyq_l, nyq_h, mask):
    """Compute possible solutions for dual PRF correction."""
    shape = vel_new.shape
    possible_solutions = np.empty((7, *shape))
    differences = np.empty((7, *shape))
    possible_solutions[0] = vel_new.copy()
    
    count = 1
    valid_mask = ~(np.isnan(diff_az) | np.isnan(diff_rad))
    
    for n1 in range(1, 4):
        for nyq in [2 * n1 * nyq_l, 2 * n1 * nyq_h]:
            bound = nyq
            limit = max(0, nyq - bound)
            
            vel_possible = vel_new.copy()
            
            # Vectorized error detection and correction
            pos_mask = valid_mask & (diff_mean > limit) & (diff_mean < nyq + bound)
            neg_mask = valid_mask & (diff_mean < -limit) & (diff_mean > -(nyq + bound))
            
            vel_possible[pos_mask] -= nyq
            vel_possible[neg_mask] += nyq
            mask[pos_mask | neg_mask] = 1
            
            possible_solutions[count] = vel_possible
            differences[count] = vel_possible - vel_smooth_mean
            count += 1
    
    return possible_solutions, differences, mask


def error_correct(radar, vel_field='VT', fnyq=0, nyq_l=0, nyq_h=0, method='staggered',
                 plot_stats=False, name='figure', radial_filters=None, azimuthal_filters=None,
                 determine_nyqs=False):
    """
    Correct errors related to staggered-PRT processing.
    
    Parameters
    ----------
    radar : object
        The radar object from Py-ART
    vel_field : str
        The velocity field name (should already be dealiased)
    fnyq : float
        Sum of the low and high PRF nyquists
    nyq_l : float
        The low PRF nyquist in m/s
    nyq_h : float
        The high PRF nyquist in m/s
    method : str
        Either 'staggered' or 'dual' for correction method
    plot_stats : bool
        True to plot histogram before and after correction
    name : str
        Name for figure output if plot_stats is True
    radial_filters : list or None
        List of int filter lengths for radial dimension
    azimuthal_filters : list or None
        List of int filter lengths for azimuthal dimension
    determine_nyqs : bool
        If True, compute Nyquist velocities from radar object
    
    Returns
    -------
    object
        The radar object with corrected velocity field
    """
    if method not in ['staggered', 'dual']:
        raise ValueError("Method must be 'staggered' or 'dual'")
    
    filters = _validate_filters(radial_filters, azimuthal_filters, method)
    
    time_start = datetime.now()
    points_caught = 0
    
    for rad_filter, az_filter in filters:
        for sweep_slice, sweep_num in zip(radar.iter_slice(), range(radar.nsweeps)):
            
            if determine_nyqs:
                nyq_l, nyq_h, fnyq = retrieve_nyqs(radar, sweep_slice, sweep_num)
                if nyq_l == np.inf:
                    print(f"Skipping sweep {sweep_num}. Sweep determined to be single PRF sweep.")
                    continue
                print(f"Nyquists for sweep {sweep_num}: Low={nyq_l:.1f}, High={nyq_h:.1f}")
            
            nyq_dual_prf = round(nyq_l / (nyq_h - nyq_l)) * nyq_h
            
            # Process velocity field
            vel = radar.fields[vel_field]['data'][sweep_slice].copy()
            vel = np.ma.masked_outside(vel, -500, 500)
            
            # Radial smoothing
            vel_smooth_rad, diff_rad = smooth_vel(vel, nyq_dual_prf, radar.range['data'], rad_filter)
            
            # Azimuthal smoothing (transpose for processing)
            vel_t = vel.T
            vel_smooth_az, diff_az = smooth_vel(vel_t, nyq_dual_prf, 
                                               radar.azimuth['data'][sweep_slice], az_filter)
            vel_smooth_az, diff_az = vel_smooth_az.T, diff_az.T
            
            # Compute mean smoothed field
            vel_smooth_mean = np.nanmean([vel_smooth_az, vel_smooth_rad], axis=0)
            diff_mean = vel.filled(fill_value=np.nan) - vel_smooth_mean
            
            vel_new = vel.copy()
            
            # Compute standard deviations
            _, std_az = norm.fit(np.ma.masked_invalid(diff_az).compressed())
            _, std_rad = norm.fit(np.ma.masked_invalid(diff_rad).compressed())
            
            # Determine masking thresholds
            max_nyq = max(nyq_l, nyq_h)
            rad_mask = min(nyq_l, nyq_h) if max_nyq < 3 * std_rad else 3 * std_rad
            az_mask = min(nyq_l, nyq_h) if max_nyq < 3 * std_az else 3 * std_az
            
            # Apply masks
            diff_az[np.abs(diff_az) < az_mask] = np.nan
            diff_rad[np.abs(diff_rad) < rad_mask] = np.nan
            
            # Plot statistics if requested
            if points_caught == 0 and plot_stats:
                _plot_histogram(diff_mean, name, 'before')
            
            # Compute possible solutions
            mask = np.zeros(vel_new.shape)
            
            if method == 'staggered':
                possible_solutions, mask = _compute_possible_solutions_staggered(
                    vel_new, vel_smooth_mean, diff_az, diff_rad, diff_mean, 
                    nyq_l, nyq_h, fnyq, mask)
            else:  # dual
                possible_solutions, differences, mask = _compute_possible_solutions_dual(
                    vel_new, vel_smooth_mean, diff_az, diff_rad, diff_mean,
                    nyq_l, nyq_h, mask)
            
            # Recompute smoothed field excluding corrected points
            masked_vel = np.ma.masked_where(mask == 1, vel_new)
            vel_smooth_recompute = np.nanmean([
                smooth_vel(masked_vel, nyq_dual_prf, radar.range['data'], rad_filter)[0],
                smooth_vel(masked_vel.T, nyq_dual_prf, radar.azimuth['data'][sweep_slice], az_filter)[0].T
            ], axis=0)
            
            # Compute differences for all solutions
            differences = np.abs([vel_poss - vel_smooth_recompute for vel_poss in possible_solutions])
            differences = np.nan_to_num(differences, nan=0.0)
            
            # Find best solution
            best_indices = np.nanargmin(differences, axis=0)
            points_caught += np.sum(best_indices != 0)
            
            # Apply final solution
            final_solution = np.take_along_axis(possible_solutions, 
                                               best_indices[np.newaxis, ...], axis=0)[0]
            
            radar.fields[vel_field]['data'][sweep_slice] = np.ma.masked_where(
                radar.fields[vel_field]['data'][sweep_slice].mask, final_solution)
    
    print(f"TOTAL TIME {(datetime.now() - time_start).total_seconds():.2f} seconds")
    
    if plot_stats:
        _plot_histogram(diff_mean, name, 'after')
    
    return radar


def _plot_histogram(data, name, suffix):
    """Helper function to plot histogram."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(data.flatten(), bins=range(-40, 41, 1), alpha=0.5, 
            histtype='bar', ec='black', align='mid', log=True)
    ax.set_xlim(-40, 40)
    ax.set_ylim(1e0, 1e5)
    ax.set_xlabel('Velocity Difference from Mean Difference (m s$^{-1}$)')
    fig.savefig(f'{name}{suffix}.png', dpi=300)
    plt.close(fig)


def retrieve_nyqs(radar, sweep_slice, sweep_num):
    """
    Retrieve Nyquist velocities from radar object.
    
    Parameters
    ----------
    radar : object
        Radar object
    sweep_slice : slice
        Slice for current sweep
    sweep_num : int
        Sweep number
    
    Returns
    -------
    tuple
        Low Nyquist, High Nyquist, and sum of both
    """
    nyq_slice = radar.instrument_parameters['nyquist_velocity']['data'][sweep_slice]
    v_high = nyq_slice.max()
    v_low = nyq_slice.min()
    
    if v_high == v_low:
        # Check for prt_ratio field
        if 'prt_ratio' in radar.instrument_parameters:
            prt_ratio = np.nanmean(radar.instrument_parameters['prt_ratio']['data'][sweep_slice])
            
            if prt_ratio == np.inf:
                return np.inf, np.inf, np.inf
            
            # Determine integer components of ratio
            if prt_ratio > 1:
                m = int(1 / (prt_ratio - 1))
                v_low, v_high = v_high / (m + 1), v_high / m
            elif prt_ratio < 1:
                m = int(1 / (1 - prt_ratio))
                v_low, v_high = v_high / m, v_high / (m - 1)
            else:
                return np.inf, np.inf, np.inf
        
        # Check for prt field
        elif 'prt' in radar.instrument_parameters:
            prt = radar.instrument_parameters['prt']['data'][sweep_slice][0]
            freq = radar.instrument_parameters['frequency']['data'][0]
            wavelength = 3e8 / freq
            v_high = wavelength / (4 * prt)
            
            v_prf = v_low
            v_low = v_prf / round(v_prf / v_high + 1)
            
            # Validate Nyquist velocities
            if not _check_nyquist(v_low, v_high):
                v_low = v_high
                v_high = v_prf / round(v_prf / v_low - 1)
                
                if not _check_nyquist(v_low, v_high):
                    raise ValueError("Cannot determine Nyquist velocities from prt.")
    
    return v_low, v_high, v_low + v_high


def _check_nyquist(v_low, v_high):
    """Check if Nyquist velocities make sense."""
    dv = v_high - v_low
    a = round(v_low / dv)
    b = round(v_high / dv)
    return (a, b) in [(4, 5), (3, 4)]
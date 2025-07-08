#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from numba import njit


# In[17]:
@njit
def get_statistic(window_values):
    idx = np.arange(len(window_values))
    valid_mask = ~np.isnan(window_values)
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    valid_values = window_values[valid_mask]
    median_val = np.median(valid_values)
    
    index_median = np.median(idx)
    index_diff = np.abs(idx - index_median)
    
    median_diff = np.abs(window_values - median_val)
    min_diff = np.nanmin(median_diff)
    
    closest_idx = np.where(median_diff == min_diff)[0]
    if len(closest_idx) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    chosen_idx = closest_idx[np.argmin(index_diff[closest_idx])]
    std_dev = np.nanstd(window_values)
    mad = np.nanmedian(np.abs(valid_values - median_val)) * 1.4826
    val_range = np.nanmax(valid_values) - np.nanmin(valid_values)

    return median_val, index_diff[chosen_idx], std_dev, mad, val_range

@njit
def moving_median_extend_asymmetrical_core(value_array, n, max_window, use_sd, ideal_valid_per_side):
    total_len = len(value_array)
    median_list = np.full(total_len, np.nan)
    count_list = np.zeros(total_len)
    window_list = np.zeros(total_len)
    lside_list = np.zeros(total_len)
    rside_list = np.zeros(total_len)
    spread_list = np.full(total_len, np.nan)
    range_list = np.full(total_len, np.nan)
    distance_list = np.full(total_len, np.nan)

    for i in range(50, n - 50):
        found = False
        converged = False
        prev_median = np.nan
        prev_mad = np.nan

        for left_len in range(1, max_window):
            for right_len in range(1, max_window):
                total_len_local = left_len + right_len + 1
                if total_len_local > 90:
                    continue
                if min(left_len, right_len) > 30 or max(left_len, right_len) > 60:
                    continue
                ratio = max(left_len, right_len) / min(left_len, right_len)
                if ratio > 2:
                    continue

                window_start = i - left_len
                window_end = i + right_len + 1
                window_vals = value_array[window_start:window_end]
                valid_mask = ~np.isnan(window_vals)
                split_idx = i - window_start
                valid_before = np.sum(valid_mask[:split_idx])
                valid_after = np.sum(valid_mask[split_idx + 1:])

                if valid_before < ideal_valid_per_side or valid_after < ideal_valid_per_side or valid_before != valid_after:
                    continue

                median_val, idx_diff, std_dev, mad, val_range = get_statistic(window_vals)

                if np.isnan(prev_median):
                    prev_median = median_val
                    prev_mad = mad
                else:
                    if mad > 0.0 and np.abs(median_val - prev_median) <= 0.5 * mad:
                        # Check for Convergence
                        median_list[i] = median_val
                        count_list[i] = np.sum(valid_mask)
                        window_list[i] = total_len_local
                        lside_list[i] = left_len
                        rside_list[i] = right_len
                        spread_list[i] = std_dev if use_sd else mad
                        range_list[i] = val_range
                        distance_list[i] = idx_diff
                        found = True
                        converged = True
                        break

                    # Update for next loop
                    prev_median = median_val
                    prev_mad = mad

            if converged:
                break

        # Fallback 90 day span window
        if not found:
            left_len = 45
            right_len = 44
            window_start = i - left_len
            window_end = i + right_len + 1
            window_vals = value_array[window_start:window_end]
            valid_mask = ~np.isnan(window_vals)

            if np.any(valid_mask):
                median_val, idx_diff, std_dev, mad, val_range = get_statistic(window_vals)

                median_list[i] = median_val
                count_list[i] = np.sum(valid_mask)
                window_list[i] = 90
                lside_list[i] = left_len
                rside_list[i] = right_len
                spread_list[i] = std_dev if use_sd else mad
                range_list[i] = val_range
                distance_list[i] = idx_diff

    return (median_list[50:50 + n], count_list[50:50 + n], window_list[50:50 + n],
            lside_list[50:50 + n], rside_list[50:50 + n], spread_list[50:50 + n],
            range_list[50:50 + n], distance_list[50:50 + n])


def moving_median_extend_asymmetrical(value_list, max_window=61, use_sd=True, ideal_valid_per_side=1):
    value_array = np.array([np.nan]*50 + list(value_list) + [np.nan]*50, dtype=np.float64)
    n = len(value_list)

    median, count, win, left, right, spread, rng, lag = moving_median_extend_asymmetrical_core(
        value_array, n, max_window, use_sd, ideal_valid_per_side
    )

    return pd.DataFrame({
        "Median": median,
        "Count": count,
        "WindowSize": win,
        "LeftSize": left,
        "RightSize": right,
        "Spread": spread,
        "Range": rng,
        "TempLag": lag
    }).reset_index(drop=True)


def get_median_all_thres_asymmetrical(value_list, max_valid=11, initial_i=1, max_window=61, use_sd=True):
    one_clm = moving_median_extend_asymmetrical(value_list, max_window, use_sd, initial_i)
    one_clm.columns = [f"{col}" for col in one_clm.columns]
    return one_clm

def get_median_change(df, get_value=True):
    df = df.filter(regex=r"^Median_w\d+$")
    change_cols = {}
    for i in range(df.shape[1] - 1):
        diff_col = f'diff_{i+1}_{i}'
        change_cols[diff_col] = abs(df.iloc[:, i+1] - df.iloc[:, i])
    change_df = pd.DataFrame(change_cols)
    return df if get_value else change_df

def iammf(value_list, max_window=61, max_valid=11, use_sd=True, initial_i=1, change_thres='MAD', threshold_scale=0.5):
    full_df = get_median_all_thres_asymmetrical(value_list, max_valid, initial_i, max_window, use_sd)
    return full_df


def fix_gap_image(df):
    df = df.sort_values('Date').reset_index(drop=True)
    new_rows = []
    for i in range(len(df) - 1):
        curr_date = df.loc[i, 'Date']
        next_date = df.loc[i + 1, 'Date']
        date_diff = (next_date - curr_date).days
        if date_diff > 1:
            for j in range(1, date_diff):
                new_date = curr_date + timedelta(days=j)
                new_rows.append({
                    'Date': new_date,
                    'JlnDt': new_date.timetuple().tm_yday,
                    'Year': new_date.year,
                    'Month': f"{new_date.month:02d}"
                })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df


# In[18]:


#DATA PROCESSING
df = pd.read_csv("OneGlacierDemo.csv")[["Albedo", "Date"]]
df['Albedo'] = df['Albedo'].replace("NA", np.nan) * 0.01
df['Date'] = pd.to_datetime(df['Date'])
df['JulianDate'] = df['Date'].dt.dayofyear
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df = df.drop_duplicates(subset='Date')

#GLACIER PROCESSING
sub = df.copy()
sub = fix_gap_image(sub)
result = iammf(sub['Albedo'].values, max_window=61, max_valid=21, use_sd=False,
                   initial_i=1, change_thres='MAD', threshold_scale=0.5)
sub = sub.iloc[:len(result)].copy()
sub['Albedo_IAMMF'] = result["Median"]

#sub['Albedo_IAMMF_interp'] = sub['Albedo_IAMMF'].interpolate(method = 'linear')

#GRAPHING
plt.figure(figsize=(11, 4))
plt.plot(sub['Date'], sub['Albedo'], label="Original Albedo", color="dodgerblue", alpha=0.4)
plt.plot(sub['Date'], sub['Albedo_IAMMF'], label="IAMMF Smoothed", color="orangered", alpha=0.8)

#plt.plot(sub['Date'], sub['Albedo_IAMMF_interp'], label="IAMMF (Interpolated)", color="orangered", alpha=0.8)

plt.xlabel("Date")
plt.ylabel("Albedo")
plt.legend()
plt.tight_layout()
plt.show()


#Download as new file
# final_df = df.copy()
# final_df["Albedo_IAMMF"] = sub["Albedo_IAMMF"] 
# col = 'IAMMF_org' 
# final_df = final_df[[c for c in final_df.columns if c != col] + [col]]
# final_df.to_csv("Testing.csv", index=False)


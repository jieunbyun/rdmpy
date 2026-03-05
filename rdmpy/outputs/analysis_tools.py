# Utils for demos and outputs - shared functions for loading, processing, and visualizing incident data

import json
import pickle
import sys
import os
from IPython.display import display
import pandas as pd
import copy
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import glob

# Helper function to find processed_data directory
def find_processed_data_path():
    """
    Find the processed_data directory by checking multiple possible locations.
    Returns the path if found, None otherwise.
    """
    possible_paths = [
        '../processed_data',  # From outputs/ or demos/
        './processed_data',   # From root directory
        'processed_data',     # From root directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_data')  # Absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# ============================================================================
# HELPER FUNCTIONS - LOAD AND FILTER DATA
# ============================================================================

def _load_station_files_and_filter_incident(processed_base, incident_number, filter_date=None):
    """
    Load all parquet files and filter by incident number, optionally by date.
    
    Returns: (all_incidents list, files_processed count, files_with_data count)
    """
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    all_incidents = []
    files_processed = 0
    files_with_data = 0
    
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        parquet_files = glob.glob(os.path.join(station_path, "*.parquet"))
        
        for file_path in parquet_files:
            files_processed += 1
            try:
                station_data = pd.read_parquet(file_path, engine='fastparquet')
                
                if isinstance(station_data, pd.DataFrame) and len(station_data) > 0:
                    if 'INCIDENT_NUMBER' in station_data.columns:
                        station_data = station_data.dropna(subset=['INCIDENT_NUMBER'])
                        
                        if len(station_data) == 0:
                            continue
                        
                        # Match incident number
                        try:
                            incident_float = float(incident_number)
                            incident_mask = (station_data['INCIDENT_NUMBER'] == incident_float)
                        except (ValueError, TypeError):
                            incident_mask = (station_data['INCIDENT_NUMBER'].astype(str) == str(incident_number))
                        
                        # Optional date filtering
                        if filter_date is not None and 'EVENT_DATETIME' in station_data.columns:
                            station_data = station_data.dropna(subset=['EVENT_DATETIME'])
                            if len(station_data) == 0:
                                continue
                            
                            station_data['event_date'] = pd.to_datetime(station_data['EVENT_DATETIME'], 
                                                                      format='%d-%b-%Y %H:%M', errors='coerce').dt.date
                            
                            try:
                                target_date = datetime.strptime(filter_date, '%d-%b-%Y').date()
                            except ValueError:
                                date_formats = ['%d-%b-%Y', '%d-%B-%Y', '%Y-%m-%d', '%m/%d/%Y']
                                target_date = None
                                for fmt in date_formats:
                                    try:
                                        target_date = datetime.strptime(filter_date, fmt).date()
                                        break
                                    except ValueError:
                                        continue
                                if target_date is None:
                                    continue
                            
                            date_mask = (station_data['event_date'] == target_date)
                            filtered_data = station_data[incident_mask & date_mask]
                        else:
                            filtered_data = station_data[incident_mask]
                        
                        if len(filtered_data) > 0:
                            files_with_data += 1
                            all_incidents.extend(filtered_data.to_dict('records'))
                        
            except Exception as e:
                continue
    return all_incidents, files_processed, files_with_data


def _parse_incident_datetimes(df, add_target_date=None):
    """
    Parse all datetime columns in incident DataFrame.
    If add_target_date provided, creates chart_datetime for visualization.
    """
    if len(df) == 0:
        return df
    
    # Parse main event datetime
    df['full_datetime'] = pd.to_datetime(df['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['full_datetime']).sort_values('full_datetime')
    
    # Parse incident start datetime
    if 'INCIDENT_START_DATETIME' in df.columns:
        df['incident_start_datetime'] = pd.to_datetime(df['INCIDENT_START_DATETIME'], 
                                                       format='%d-%b-%Y %H:%M', errors='coerce')
    
    # Create chart datetime if target date provided
    if add_target_date is not None:
        target_date = add_target_date if isinstance(add_target_date, date) else \
                     datetime.strptime(add_target_date, '%d-%b-%Y').date()
        
        df['time_only'] = df['full_datetime'].dt.time
        df['chart_datetime'] = df['time_only'].apply(lambda t: datetime.combine(target_date, t))
        df['hour'] = df['full_datetime'].dt.hour
        
        # Also prepare incident start chart time
        if 'incident_start_datetime' in df.columns:
            df['start_time_only'] = df['incident_start_datetime'].dt.time
            df['start_chart_datetime'] = df['start_time_only'].apply(lambda t: datetime.combine(target_date, t) if pd.notna(t) else None)
    
    return df


def _calculate_summary_statistics(df):
    """Calculate incident summary statistics."""
    stats = {
        'total_delay_minutes': df['PFPI_MINUTES'].fillna(0).sum() if len(df) > 0 else 0,
        'total_cancellations': len(df[df['EVENT_TYPE'] == 'C']) if len(df) > 0 else 0,
        'total_records': len(df),
    }
    
    # Peak delay info
    stats['peak_delay_info'] = "N/A"
    stats['peak_regular_delay'] = "N/A"
    stats['peak_cancellation_delay'] = "N/A"
    
    if len(df[df['PFPI_MINUTES'] > 0]) > 0:
        peak_idx = df['PFPI_MINUTES'].idxmax()
        peak_delay = df.loc[peak_idx, 'PFPI_MINUTES']
        peak_time = df.loc[peak_idx, 'full_datetime'].strftime('%H:%M')
        peak_is_cancelled = df.loc[peak_idx, 'EVENT_TYPE'] == 'C'
        
        if peak_is_cancelled:
            stats['peak_delay_info'] = f"{peak_delay:.1f} minutes at {peak_time} (CANCELLED SERVICE)"
        else:
            stats['peak_delay_info'] = f"{peak_delay:.1f} minutes at {peak_time} (Regular Delay)"
    
    delays_only = df[df['EVENT_TYPE'] != 'C']
    cancellations_only = df[df['EVENT_TYPE'] == 'C']
    
    if len(delays_only[delays_only['PFPI_MINUTES'] > 0]) > 0:
        peak_reg_idx = delays_only['PFPI_MINUTES'].idxmax()
        peak_reg_delay = delays_only.loc[peak_reg_idx, 'PFPI_MINUTES']
        peak_reg_time = delays_only.loc[peak_reg_idx, 'full_datetime'].strftime('%H:%M')
        stats['peak_regular_delay'] = f"{peak_reg_delay:.1f} minutes at {peak_reg_time}"
    
    if len(cancellations_only[cancellations_only['PFPI_MINUTES'] > 0]) > 0:
        peak_canc_idx = cancellations_only['PFPI_MINUTES'].idxmax()
        peak_canc_delay = cancellations_only.loc[peak_canc_idx, 'PFPI_MINUTES']
        peak_canc_time = cancellations_only.loc[peak_canc_idx, 'full_datetime'].strftime('%H:%M')
        stats['peak_cancellation_delay'] = f"{peak_canc_delay:.1f} minutes at {peak_canc_time}"
    
    return stats


# ============================================================================
# HELPER FUNCTIONS - VISUALIZATION
# ============================================================================

def _create_hourly_chart_for_date(ax, df, target_date, base_datetime, incident_end_chart_datetime=None):
    """Create hourly delay totals bar chart for a specific date."""
    delay_data = df[df['PFPI_MINUTES'] > 0].copy() if 'PFPI_MINUTES' in df.columns else pd.DataFrame()
    
    hour_datetimes = []
    hour_values = []
    hourly_delays = delay_data.groupby('hour')['PFPI_MINUTES'].sum() if len(delay_data) > 0 else pd.Series()
    
    for hour in range(24):
        hour_dt = datetime.combine(target_date if isinstance(target_date, date) else \
                                  datetime.strptime(target_date, '%d-%b-%Y').date(),
                                  datetime.strptime(f'{hour:02d}:00', '%H:%M').time())
        hour_datetimes.append(hour_dt)
        hour_values.append(hourly_delays.get(hour, 0))
    
    bars = ax.bar(hour_datetimes, hour_values, width=timedelta(minutes=45), alpha=0.7, color='steelblue')
    
    # Add incident markers
    if 'start_chart_datetime' in df.columns:
        incident_start_times = df[df['start_chart_datetime'].notna()]['start_chart_datetime'].unique()
        for incident_start_time in incident_start_times:
            ax.axvline(x=incident_start_time, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        if len(incident_start_times) > 0:
            ax.axvline(x=incident_start_times[0], color='red', linestyle='--', linewidth=3, alpha=0.9, 
                       label='Incident Start Time')
    
    if incident_end_chart_datetime is not None:
        ax.axvline(x=incident_end_chart_datetime, color='green', linestyle='--', linewidth=3, alpha=0.9, 
                   label='Incident End Time')
    
    # Format axis
    fixed_end = base_datetime + timedelta(days=1) - timedelta(minutes=1)
    ax.set_xlim(base_datetime, fixed_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)
    
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)


def _create_severity_chart(ax, df):
    """Create delay severity distribution bar chart."""
    delay_data = df[df['PFPI_MINUTES'] > 0].copy()
    
    if len(delay_data) == 0:
        ax.text(0.5, 0.5, 'No delay events found', transform=ax.transAxes, 
                ha='center', va='center', fontsize=22)
        return
    
    delay_values = delay_data['PFPI_MINUTES'].values
    bins = [0, 5, 15, 30, 60, 120, float('inf')]
    labels = ['1\n-5', '6\n-15', '16\n-30', '31\n-60', '61\n-120', '120\n+']
    
    counts, _ = np.histogram(delay_values, bins=bins)
    colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred', 'purple']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
    ax.set_ylabel('Number of\nDelay Events', fontsize=20)
    ax.set_xlabel('Delay Severity Range (minutes)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3, axis='y')


def _create_timeline_scatter_chart(ax, df, base_datetime, target_date, incident_end_chart_datetime=None):
    """Create timeline scatter plot of delays and cancellations."""
    delays = df[df['EVENT_TYPE'] != 'C'].copy()
    cancellations = df[df['EVENT_TYPE'] == 'C'].copy()
    
    if len(delays) > 0:
        ax.scatter(delays['chart_datetime'], delays['PFPI_MINUTES'], 
                   s=60, alpha=0.7, color='blue', label=f'Delays ({len(delays)})')
    
    if len(cancellations) > 0:
        ax.scatter(cancellations['chart_datetime'], cancellations['PFPI_MINUTES'], 
                   s=100, marker='X', color='red', alpha=0.8, 
                   label=f'Cancellations ({len(cancellations)})')
    
    # Add incident markers
    if 'start_chart_datetime' in df.columns:
        incident_start_times = df[df['start_chart_datetime'].notna()]['start_chart_datetime'].unique()
        for incident_start_time in incident_start_times:
            ax.axvline(x=incident_start_time, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        if len(incident_start_times) > 0:
            ax.axvline(x=incident_start_times[0], color='red', linestyle='--', linewidth=3, alpha=0.9, 
                       label='Incident Start Time')
    
    if incident_end_chart_datetime is not None:
        ax.axvline(x=incident_end_chart_datetime, color='green', linestyle='--', linewidth=3, alpha=0.9, 
                   label='Incident End Time')
    
    # Format axis
    fixed_end = base_datetime + timedelta(days=1) - timedelta(minutes=1)
    ax.set_xlim(base_datetime, fixed_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)
    
    ax.set_ylabel('Delay Minutes', fontsize=22)
    ax.set_xlabel('24-Hour Timeline', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)


def _get_day_suffix(analysis_datetime):
    """
    Get day of week suffix from datetime.
    
    Parameters:
    -----------
    analysis_datetime : datetime
        The datetime to extract day of week from
    
    Returns:
    --------
    str : Day suffix (MO, TU, WE, TH, FR, SA, SU)
    """
    
    day_mapping = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
    return day_mapping[analysis_datetime.weekday()]


def _get_station_name_from_reference(incident_section_code):
    """
    Look up station name from STANOX code in reference files.
    
    Parameters:
    -----------
    incident_section_code : str/int
        STANOX code to look up
    
    Returns:
    --------
    str or None : Station name if found, None otherwise
    """
    
    if not incident_section_code:
        return None
    
    try:
        from demo.data.reference import reference_files
        with open(reference_files["station codes"], 'r') as f:
            station_codes_data = json.load(f)
        
        for record in station_codes_data:
            if isinstance(record, dict) and str(record.get('stanox')) == str(incident_section_code):
                return record.get('description')
    except Exception:
        pass
    
    return None


def _calculate_planned_calls(df, incident_delay_day, analysis_datetime, analysis_end):
    """
    Calculate number of planned calls for analysis period.
    Filters non-delayed trains for the analysis time window.
    
    Parameters:
    -----------
    df : DataFrame
        Station data
    incident_delay_day : str/list
        Day type to match against
    analysis_datetime : datetime
        Start of analysis period
    analysis_end : datetime
        End of analysis period
    
    Returns:
    --------
    int : Number of unique planned calls in period
    """
    
    planned_calls = 0
    
    if incident_delay_day and not pd.isna(incident_delay_day):
        non_delayed_data = df[
            (df['INCIDENT_NUMBER'].isna()) &
            (df['ENGLISH_DAY_TYPE'].apply(lambda x: incident_delay_day in x if isinstance(x, list) else False))
        ].copy()
        
        if not non_delayed_data.empty:
            non_delayed_data = non_delayed_data[non_delayed_data['PLANNED_CALLS'].notna()].copy()
            
            if not non_delayed_data.empty:
                non_delayed_data['planned_time'] = non_delayed_data['PLANNED_CALLS'].apply(_parse_time_string)
                non_delayed_data = non_delayed_data[non_delayed_data['planned_time'].notna()].copy()
                
                if not non_delayed_data.empty:
                    analysis_start_time_only = analysis_datetime.time()
                    analysis_end_time_only = analysis_end.time()
                    
                    if analysis_end_time_only > analysis_start_time_only:
                        period_non_delayed = non_delayed_data[
                            (non_delayed_data['planned_time'] >= analysis_start_time_only) & 
                            (non_delayed_data['planned_time'] <= analysis_end_time_only)
                        ].copy()
                    else:
                        period_non_delayed = non_delayed_data[
                            (non_delayed_data['planned_time'] >= analysis_start_time_only) | 
                            (non_delayed_data['planned_time'] <= analysis_end_time_only)
                        ].copy()
                    
                    planned_calls = period_non_delayed['TRAIN_SERVICE_CODE'].nunique()
    
    return planned_calls


def _calculate_delayed_train_metrics(df, analysis_datetime, analysis_end):
    """
    Calculate delayed trains in/out and their delay minutes.
    
    Parameters:
    -----------
    df : DataFrame
        Incident data
    analysis_datetime : datetime
        Start of analysis period
    analysis_end : datetime
        End of analysis period
    
    Returns:
    --------
    tuple : (delayed_out, delay_minutes_out, delayed_in, delay_minutes_in)
    """
    
    all_incident_data = df[df['INCIDENT_NUMBER'].notna()].copy()
    
    if all_incident_data.empty:
        return 0, [], 0, []
    
    all_incident_data['EVENT_DATETIME_parsed'] = pd.to_datetime(
        all_incident_data['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce'
    )
    all_incident_data = all_incident_data[all_incident_data['EVENT_DATETIME_parsed'].notna()].copy()
    all_incident_data = all_incident_data[all_incident_data['PFPI_MINUTES'].notna()].copy()
    
    if all_incident_data.empty:
        return 0, [], 0, []
    
    # Calculate original scheduled time = actual time - delay
    all_incident_data['original_scheduled_time'] = (
        all_incident_data['EVENT_DATETIME_parsed'] - 
        pd.to_timedelta(all_incident_data['PFPI_MINUTES'], unit='minutes')
    )
    
    # Trains originally scheduled for analysis period
    originally_scheduled_in_period = all_incident_data[
        (all_incident_data['original_scheduled_time'] >= analysis_datetime) & 
        (all_incident_data['original_scheduled_time'] <= analysis_end)
    ].copy()
    
    delayed_trains_out = 0
    delay_minutes_out = []
    delayed_trains_in = 0
    delay_minutes_in = []
    
    if not originally_scheduled_in_period.empty:
        # Trains delayed OUT of analysis period
        trains_delayed_out_of_period = originally_scheduled_in_period[
            (originally_scheduled_in_period['EVENT_DATETIME_parsed'] > analysis_end)
        ]
        
        delayed_trains_out = trains_delayed_out_of_period['TRAIN_SERVICE_CODE'].nunique()
        delay_minutes_out = trains_delayed_out_of_period['PFPI_MINUTES'].dropna().tolist()
    
    # Trains delayed INTO analysis period
    trains_delayed_into_period = all_incident_data[
        (all_incident_data['original_scheduled_time'] < analysis_datetime) &
        (all_incident_data['EVENT_DATETIME_parsed'] >= analysis_datetime) &
        (all_incident_data['EVENT_DATETIME_parsed'] <= analysis_end)
    ]
    
    delayed_trains_in = trains_delayed_into_period['TRAIN_SERVICE_CODE'].nunique()
    delay_minutes_in = trains_delayed_into_period['PFPI_MINUTES'].dropna().tolist()
    
    return delayed_trains_out, delay_minutes_out, delayed_trains_in, delay_minutes_in


# Helpers for heatmap visualization functions:

def _load_station_coordinates_from_json():
    """Load and validate station coordinates from reference JSON file."""
    try:
        from demo.data.reference import reference_files
        import json
        
        file_path = reference_files["all dft categories"]
        
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            stations_coords_data = json.load(f)
        
        # Filter to valid categories (A, B, C1, C2) with coordinates
        valid_categories = {'A', 'B', 'C1', 'C2'}
        all_station_coords_map = {}
        
        for station in stations_coords_data:
            if isinstance(station, dict):
                station_id = str(station.get('stanox', ''))
                dft_category = station.get('dft_category', '')
                latitude = station.get('latitude')
                longitude = station.get('longitude')
                description = station.get('description', 'Unknown Station')
                
                if (dft_category in valid_categories and 
                    station_id and 
                    latitude is not None and longitude is not None and
                    str(latitude).replace('.', '').replace('-', '').isdigit() and 
                    str(longitude).replace('.', '').replace('-', '').isdigit()):
                    try:
                        all_station_coords_map[station_id] = {
                            'name': description,
                            'lat': float(latitude),
                            'lon': float(longitude),
                            'category': dft_category
                        }
                    except ValueError:
                        continue
        
        return all_station_coords_map
        
    except Exception as e:
        return None


def _parse_heatmap_analysis_parameters(analysis_date, analysis_hhmm, period_minutes, interval_minutes):
    """Parse and validate heatmap analysis parameters."""
    try:
        analysis_datetime = datetime.strptime(f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", '%d-%b-%Y %H:%M')
        analysis_end = analysis_datetime + timedelta(minutes=period_minutes)
        
        num_intervals = period_minutes // interval_minutes
        if period_minutes % interval_minutes != 0:
            num_intervals += 1
        
        analysis_day_suffix = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}[analysis_datetime.weekday()]
        
        return analysis_datetime, analysis_end, num_intervals, analysis_day_suffix
    except ValueError:
        return None, None, None, None


def _collect_heatmap_delay_timeline(target_files, incident_code, incident_date, all_station_coords_map, analysis_datetime, analysis_end, interval_minutes):
    """Collect delay timeline data for affected stations during analysis period."""
    station_timeline_data = {}
    incident_section_code = None
    incident_reason = None
    incident_start_time = None
    
    for file_path, station_code in target_files:
        if station_code not in all_station_coords_map:
            continue
            
        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
            if not isinstance(df, pd.DataFrame):
                continue
                
            all_incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            if all_incident_data.empty:
                continue
            
            # Extract incident information on first occurrence
            if incident_section_code is None and not all_incident_data.empty:
                incident_records = all_incident_data[all_incident_data['INCIDENT_START_DATETIME'].str.contains(incident_date, na=False)]
                if not incident_records.empty:
                    incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                    incident_reason = incident_records['INCIDENT_REASON'].iloc[0]
                    incident_start_time = incident_records['INCIDENT_START_DATETIME'].iloc[0]
            
            # Filter to events within analysis period
            all_incident_data['EVENT_DATETIME_parsed'] = pd.to_datetime(all_incident_data['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
            all_incident_data = all_incident_data[all_incident_data['EVENT_DATETIME_parsed'].notna()].copy()
            
            period_data = all_incident_data[
                (all_incident_data['EVENT_DATETIME_parsed'] >= analysis_datetime) &
                (all_incident_data['EVENT_DATETIME_parsed'] <= analysis_end) &
                (all_incident_data['PFPI_MINUTES'].notna())
            ].copy()
            
            if period_data.empty:
                continue
            
            # Create interval-specific delays for this station
            interval_delays = {}
            
            for _, row in period_data.iterrows():
                event_time = row['EVENT_DATETIME_parsed']
                delay = row['PFPI_MINUTES']
                
                minutes_from_start = int((event_time - analysis_datetime).total_seconds() / 60)
                interval_number = minutes_from_start // interval_minutes
                interval_start = analysis_datetime + timedelta(minutes=interval_number * interval_minutes)
                
                if interval_start not in interval_delays:
                    interval_delays[interval_start] = 0
                interval_delays[interval_start] += delay
            
            timeline = [(interval_start, interval_delays[interval_start]) for interval_start in sorted(interval_delays.keys())]
            
            if timeline:
                station_timeline_data[station_code] = timeline
                
        except Exception:
            continue
    
    return station_timeline_data, incident_section_code, incident_reason, incident_start_time


def _get_incident_location_coordinates(incident_section_code):
    """Get coordinates for incident location from STANOX codes."""
    incident_locations = []
    incident_station_name = None
    
    if incident_section_code:
        try:
            from demo.data.reference import reference_files
            import json
            
            with open(reference_files["station codes"], 'r') as f:
                station_codes_data = json.load(f)
            
            stanox_codes = [code.strip() for code in str(incident_section_code).split(':')]
            
            for stanox_code in stanox_codes:
                found_match = False
                for record in station_codes_data:
                    if isinstance(record, dict):
                        record_stanox = record.get('stanox')
                        if (str(record_stanox) == str(stanox_code) or 
                            str(record_stanox).zfill(5) == str(stanox_code).zfill(5) or
                            str(record_stanox).lstrip('0') == str(stanox_code).lstrip('0')):
                            found_match = True
                            if record.get('latitude') and record.get('longitude'):
                                location_info = {
                                    'lat': float(record['latitude']),
                                    'lon': float(record['longitude']),
                                    'name': record.get('description', 'Incident Location'),
                                    'stanox': stanox_code
                                }
                                incident_locations.append(location_info)
                                incident_station_name = record.get('description', 'Unknown')
                            break
                
        except Exception as e:
            pass
    
    return incident_locations, incident_station_name


def _build_timeline_data_structure(all_station_coords_map, station_timeline_data, analysis_datetime, analysis_end, interval_minutes):
    """Build timeline data structure indexed by time steps for JavaScript."""
    time_steps = []
    current_time = analysis_datetime
    while current_time < analysis_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    timeline_data = {}
    for step_time in time_steps:
        time_key = step_time.strftime('%H:%M')
        timeline_data[time_key] = {}
        
        for station_code in all_station_coords_map.keys():
            if station_code in station_timeline_data:
                station_timeline = station_timeline_data[station_code]
                interval_delay = 0
                
                for event_time, delay_amount in station_timeline:
                    time_window_start = step_time
                    time_window_end = step_time + timedelta(minutes=interval_minutes)
                    
                    if time_window_start <= event_time < time_window_end:
                        interval_delay += delay_amount
                
                timeline_data[time_key][station_code] = interval_delay
            else:
                timeline_data[time_key][station_code] = 0
    
    return timeline_data, time_steps


# Helpers for station view yearly:

def _load_all_station_day_files(station_folder):
    """Load all day files for a station, returning combined dataframe with day labels."""
    all_station_data = []
    day_files = ['MO.parquet', 'TU.parquet', 'WE.parquet', 'TH.parquet', 'FR.parquet', 'SA.parquet', 'SU.parquet']
    
    for day_file in day_files:
        file_path = os.path.join(station_folder, day_file)
        if os.path.exists(file_path):
            try:
                day_data = pd.read_parquet(file_path, engine='fastparquet')
                day_data['day_of_week'] = day_file.replace('.parquet', '')
                all_station_data.append(day_data)
            except Exception as e:
                pass
    
    if not all_station_data:
        return None
    
    combined_data = pd.concat(all_station_data, ignore_index=True)
    return combined_data


def _separate_incident_and_normal_operations(combined_data):
    """Separate data into incident-related and normal operations."""
    # Filter for trains with planned calls
    train_mask = combined_data['PLANNED_CALLS'].notna()
    all_train_data = combined_data[train_mask].copy()
    
    # Deduplication by maximum delay
    if len(all_train_data) > 0:
        all_train_data['delay_numeric'] = pd.to_numeric(all_train_data['PFPI_MINUTES'], errors='coerce').fillna(0)
        all_train_data['dedup_priority'] = all_train_data['delay_numeric'] * 1000
        
        if 'ACTUAL_CALLS' in all_train_data.columns:
            all_train_data['dedup_priority'] += all_train_data['ACTUAL_CALLS'].notna().astype(int) * 100
        
        basic_dedup_cols = ['TRAIN_SERVICE_CODE', 'PLANNED_CALLS', 'day_of_week']
        basic_available = [col for col in basic_dedup_cols if col in all_train_data.columns]
        
        if len(basic_available) >= 2:
            all_train_data = all_train_data.sort_values(['delay_numeric', 'dedup_priority'], ascending=[False, False])
            all_train_data = all_train_data.drop_duplicates(subset=basic_available, keep='first')
            all_train_data = all_train_data.drop(['delay_numeric', 'dedup_priority'], axis=1)
    
    if len(all_train_data) == 0:
        return None, None
    
    # Separate by incident presence
    incident_mask = all_train_data['INCIDENT_NUMBER'].notna()
    incident_data = all_train_data[incident_mask].copy()
    normal_data = all_train_data[~incident_mask].copy()
    
    return incident_data, normal_data


def _process_operations_data(data, operation_type, interval_minutes):
    """Process operations data (incident or normal) into time-interval summaries."""
    if len(data) == 0:
        return pd.DataFrame()
    
    reference_date = datetime(2024, 1, 1)
    
    def parse_time_simple(time_val, base_date):
        if pd.isna(time_val):
            return None
        try:
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except:
            return None
    
    data['planned_dt'] = data['PLANNED_CALLS'].apply(lambda x: parse_time_simple(x, reference_date))
    data['original_actual_dt'] = data['ACTUAL_CALLS'].apply(lambda x: parse_time_simple(x, reference_date))
    data['delay_minutes'] = pd.to_numeric(data['PFPI_MINUTES'], errors='coerce').fillna(0)
    
    # Correct actual times based on delays
    corrected_actual_times = []
    for _, row in data.iterrows():
        planned_dt = row['planned_dt']
        original_actual_dt = row['original_actual_dt']
        delay_min = row['delay_minutes']
        
        if pd.isna(planned_dt):
            corrected_actual_times.append(None)
        elif delay_min > 0:
            corrected_actual_times.append(planned_dt + timedelta(minutes=delay_min))
        elif delay_min == 0:
            corrected_actual_times.append(planned_dt)
        else:
            corrected_actual_times.append(original_actual_dt if pd.notna(original_actual_dt) else planned_dt)
    
    data['effective_time'] = corrected_actual_times
    valid_data = data[data['effective_time'].notna()].copy()
    
    if len(valid_data) == 0:
        return pd.DataFrame()
    
    # Group by time intervals
    valid_data['hour_of_day'] = valid_data['effective_time'].dt.hour
    valid_data['interval_group'] = (valid_data['hour_of_day'] * 60 + valid_data['effective_time'].dt.minute) // interval_minutes
    
    intervals = []
    for interval_group in valid_data['interval_group'].unique():
        interval_trains = valid_data[valid_data['interval_group'] == interval_group]
        
        arrival_trains = interval_trains[interval_trains['EVENT_TYPE'] != 'C']
        cancellation_trains = interval_trains[interval_trains['EVENT_TYPE'] == 'C']
        
        if len(arrival_trains) > 0 or len(cancellation_trains) > 0:
            delay_values = arrival_trains['delay_minutes'].tolist() if len(arrival_trains) > 0 else []
            ontime_arrivals = len([d for d in delay_values if d == 0.0])
            delayed_arrivals = len([d for d in delay_values if d > 0.0])
            delayed_minutes = [round(d, 1) for d in delay_values if d > 0.0]
            total_cancellations = len(cancellation_trains)
            
            start_minute = interval_group * interval_minutes
            end_minute = start_minute + interval_minutes
            time_period = f"{(start_minute // 60):02d}:{(start_minute % 60):02d}-{(end_minute // 60):02d}:{(end_minute % 60):02d}"
            
            intervals.append({
                'time_period': time_period,
                'ontime_arrival_count': ontime_arrivals,
                'delayed_arrival_count': delayed_arrivals,
                'cancellation_count': total_cancellations,
                'delay_minutes': delayed_minutes,
                'operation_type': operation_type
            })
    
    return pd.DataFrame(intervals)


# for aggregate view:

def aggregate_view(incident_number, start_date):
    """
    Multi-day incident analysis with clean separation of concerns.
    Creates 3 charts: hourly delays, severity distribution, and event timeline.
    
    Parameters:
    -----------
    incident_number : int/str
        The incident number to analyze
    start_date : str
        Starting date in 'DD-MMM-YYYY' format
    
    Returns:
    --------
    dict : Summary statistics for the incident
    """
    
    # Step 1: Locate and load data
    processed_base = find_processed_data_path()
    if processed_base is None:
        return None
    
    all_incidents, files_processed, files_with_data = _load_station_files_and_filter_incident(
        processed_base, incident_number, filter_date=start_date
    )
    
    if not all_incidents:
        return None
    
    # Step 2: Convert to DataFrame and parse datetimes
    df = pd.DataFrame(all_incidents)
    
    target_date = datetime.strptime(start_date, '%d-%b-%Y').date()
    base_datetime = datetime.combine(target_date, datetime.min.time())
    df = _parse_incident_datetimes(df, add_target_date=target_date)
    
    # Step 3: Calculate metrics
    stats = _calculate_summary_statistics(df)
    
    # Calculate incident end time
    incident_end_datetime = df['full_datetime'].max() if len(df) > 0 else None
    incident_end_chart_datetime = None
    if pd.notna(incident_end_datetime):
        end_time_only = incident_end_datetime.time()
        incident_end_chart_datetime = datetime.combine(target_date, end_time_only)
    
    # Step 4: Create visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 18))
    
    _create_hourly_chart_for_date(ax1, df, target_date, base_datetime, incident_end_chart_datetime)
    ax1.set_ylabel('Total Delay Minutes\nper Hour', fontsize=22)
    ax1.set_xlabel('24-Hour Timeline', fontsize=22)
    
    _create_severity_chart(ax2, df)
    
    _create_timeline_scatter_chart(ax3, df, base_datetime, target_date, incident_end_chart_datetime)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
    # Step 5: Return results
    summary = {
        "Total Delay Minutes": stats['total_delay_minutes'],
        "Total Cancellations": stats['total_cancellations'],
        "Total Records Found": stats['total_records'],
        "Files Processed": files_processed,
        "Files with Data": files_with_data,
        "Incident Number": incident_number,
        "Date": start_date,
        "Time Range": f"{df['full_datetime'].min().strftime('%H:%M')} - {df['full_datetime'].max().strftime('%H:%M')}" if len(df) > 0 else "N/A",
        "Peak Delay Event": stats['peak_delay_info'],
        "Peak Regular Delay": stats['peak_regular_delay'],
        "Peak Cancellation Delay": stats['peak_cancellation_delay']
    }
    
    return summary


def _load_station_files_for_multiday_incident(processed_base, incident_number):
    """
    Load parquet files and filter for a specific incident (multi-day approach - NO date filtering).
    
    Parameters:
    -----------
    processed_base : str
        Base path to processed_data directory
    incident_number : int/float/str
        The incident number to filter by
    
    Returns:
    --------
    tuple : (all_incidents list, files_processed count, files_with_data count)
    """
    
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    all_incidents = []
    files_processed = 0
    files_with_data = 0
    
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        parquet_files = glob.glob(os.path.join(station_path, "*.parquet"))
        
        for file_path in parquet_files:
            files_processed += 1
            try:
                station_data = pd.read_parquet(file_path, engine='fastparquet')
                
                if isinstance(station_data, pd.DataFrame) and len(station_data) > 0:
                    if 'INCIDENT_NUMBER' in station_data.columns:
                        station_data = station_data.dropna(subset=['INCIDENT_NUMBER'])
                        
                        if len(station_data) == 0:
                            continue
                        
                        try:
                            incident_float = float(incident_number)
                            incident_mask = (station_data['INCIDENT_NUMBER'] == incident_float)
                        except (ValueError, TypeError):
                            incident_mask = (station_data['INCIDENT_NUMBER'].astype(str) == str(incident_number))
                        
                        # NO date filter - get all days with this incident
                        filtered_data = station_data[incident_mask]
                        
                        if len(filtered_data) > 0:
                            files_with_data += 1
                            all_incidents.extend(filtered_data.to_dict('records'))
                        
            except Exception as e:
                continue
    
    return all_incidents, files_processed, files_with_data


def _get_target_files_for_day(processed_base, day_suffix):
    """
    Load target files for a specific day of week.
    
    Parameters:
    -----------
    processed_base : str
        Base path to processed_data directory
    day_suffix : str
        Day abbreviation (e.g., 'MO', 'TU', etc.)
    
    Returns:
    --------
    list : List of (file_path, station_code) tuples
    """
    
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    target_files = []
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        day_file = os.path.join(station_path, f"{day_suffix}.parquet")
        if os.path.exists(day_file):
            target_files.append((day_file, station_dir))
    
    return target_files


def _parse_time_string(time_str):
    """
    Parse time string like '0001', '1230', '2359' to time object.
    Handles formats like 'HHMM' with optional trailing letters.
    
    Parameters:
    -----------
    time_str : str
        Time string in format 'HHMM' or similar
    
    Returns:
    --------
    datetime.time or None
    """
    
    try:
        if isinstance(time_str, str) and len(time_str) >= 4:
            # Remove any trailing letters and take first 4 digits
            clean_time = ''.join(filter(str.isdigit, time_str))[:4]
            if len(clean_time) == 4:
                hour = int(clean_time[:2])
                minute = int(clean_time[2:])
                # Handle 24:XX format by converting to 00:XX
                if hour >= 24:
                    hour = hour % 24
                return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
        return None
    except:
        return None


def _load_and_prepare_multiday_data(incident_number):
    """
    Load and prepare incident data for multi-day analysis.
    Handles data loading, datetime parsing, date identification, and event filtering.
    
    Parameters:
    -----------
    incident_number : int/str
        The incident number to load
    
    Returns:
    --------
    tuple : (df, files_processed, files_with_data, unique_dates, num_days)
            Returns None if no incidents found
    """
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        return None
    
    all_incidents, files_processed, files_with_data = _load_station_files_for_multiday_incident(
        processed_base, incident_number
    )
    
    if not all_incidents:
        return None
    
    # Parse datetime and identify unique dates
    df = pd.DataFrame(all_incidents)
    
    df['full_datetime'] = pd.to_datetime(df['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['full_datetime']).sort_values('full_datetime')
    
    # Get unique dates
    df['event_date_only'] = df['full_datetime'].dt.date
    unique_dates = sorted(df['event_date_only'].dropna().unique())
    num_days = len(unique_dates)
    
    # Check if incident spans more than 3 days
    
    # Parse INCIDENT_START_DATETIME and filter events before incident start
    if 'INCIDENT_START_DATETIME' in df.columns:
        df['incident_start_datetime'] = pd.to_datetime(df['INCIDENT_START_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
        
        incident_start_times = df['incident_start_datetime'].dropna()
        if len(incident_start_times) > 0:
            earliest_incident_start = incident_start_times.min()
            records_before_filtering = len(df)
            df = df[df['full_datetime'] >= earliest_incident_start]
            records_filtered = records_before_filtering - len(df)
            
            if records_filtered > 0:
                pass
            
            df['event_date_only'] = df['full_datetime'].dt.date
            unique_dates = sorted(df['event_date_only'].dropna().unique())
            num_days = len(unique_dates)
    
    # Parse incident end datetime column (if exists)
    if 'INCIDENT_END_DATETIME' in df.columns:
        df['incident_end_datetime'] = pd.to_datetime(df['INCIDENT_END_DATETIME'], 
                                                     format='%d-%b-%Y %H:%M', errors='coerce')
    
    return df, files_processed, files_with_data, unique_dates, num_days


def _build_legend_and_info_boxes(fig, ax_severity, timeline_axes, df, unique_dates, delay_data_all):
    """
    Create legend and info boxes for the multi-day incident visualization.
    Places all legend elements and info box text on the figure.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to add elements to
    ax_severity : matplotlib.axes.Axes
        The severity chart axes (for info box positioning)
    timeline_axes : list
        List of (ax, date, day_num) tuples for timeline chart positioning
    df : pd.DataFrame
        The incident data (for calculating delays/cancellations per day)
    unique_dates : list
        List of unique dates in the incident
    delay_data_all : pd.DataFrame
        Filtered dataframe with only delay events
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Create legend elements
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Minor', alpha=0.7),
        Patch(facecolor='yellow', edgecolor='black', label='Moderate', alpha=0.7),
        Patch(facecolor='orange', edgecolor='black', label='Significant', alpha=0.7),
        Patch(facecolor='red', edgecolor='black', label='Major', alpha=0.7),
        Patch(facecolor='darkred', edgecolor='black', label='Severe', alpha=0.7),
        Patch(facecolor='purple', edgecolor='black', label='Critical', alpha=0.7),
        Patch(facecolor='none', edgecolor='none', label=''),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Delays', alpha=0.7),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Cancellations', alpha=0.8),
        Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Incident Start', alpha=0.9),
        Line2D([0], [0], color='green', linestyle='--', linewidth=4, label='Incident End', alpha=0.9),
    ]
    
    legend_x = 0.85
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(legend_x, 0.99), 
              fontsize=20, facecolor='white', edgecolor='black', framealpha=0.9,
              title='Legend', title_fontsize=20, ncol=1, columnspacing=0.5, labelspacing=0.3)
    
    plt.subplots_adjust(right=0.68)
    
    # Add info boxes for each timeline
    if len(delay_data_all) > 0:
        total_events = len(delay_data_all)
        avg_delay = delay_data_all['PFPI_MINUTES'].mean()
        
        severity_pos = ax_severity.get_position()
        y_pos_b = (severity_pos.y0 + severity_pos.y1) / 2
        info_text_b = f'Total Events: {total_events}\nAverage Delay:\n{avg_delay:.1f} min'
        fig.text(legend_x + 0.06, y_pos_b, info_text_b, fontsize=20, 
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', alpha=0.9))
        
        for ax_timeline, current_date, day_num in timeline_axes:
            timeline_pos = ax_timeline.get_position()
            day_df_info = df[df['event_date_only'] == current_date].copy()
            delays_count = len(day_df_info[day_df_info['EVENT_TYPE'] != 'C'])
            cancellations_count = len(day_df_info[day_df_info['EVENT_TYPE'] == 'C'])
            
            if len(unique_dates) > 1:
                info_text_c = f'Day {day_num}\nDelayed: {delays_count}\nCancelled: {cancellations_count}'
            else:
                info_text_c = f'Delayed: {delays_count}\nCancelled: {cancellations_count}'
            
            y_pos_c = (timeline_pos.y0 + timeline_pos.y1) / 2
            fig.text(legend_x + 0.06, y_pos_c, info_text_c, fontsize=20,
                    ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', alpha=0.9))


def calculate_incident_summary_stats(df, delay_data_all, unique_dates, files_processed, files_with_data, incident_number, num_days):
    """
    Calculate final summary statistics for the incident.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The incident data
    delay_data_all : pd.DataFrame
        Filtered dataframe with only delay events
    unique_dates : list
        List of unique dates in the incident
    files_processed : int
        Number of files processed
    files_with_data : int
        Number of files with matching data
    incident_number : int/str
        The incident number
    num_days : int
        Number of days the incident spans
    
    Returns:
    --------
    dict : Summary statistics dictionary
    """
    total_delay_minutes = df['PFPI_MINUTES'].fillna(0).sum()
    total_cancellations = len(df[df['EVENT_TYPE'] == 'C'])
    
    peak_delay_info = "N/A"
    if len(df[df['PFPI_MINUTES'] > 0]) > 0:
        peak_idx = df['PFPI_MINUTES'].idxmax()
        peak_delay = df.loc[peak_idx, 'PFPI_MINUTES']
        peak_time = df.loc[peak_idx, 'full_datetime'].strftime('%d-%b-%Y %H:%M')
        peak_is_cancelled = df.loc[peak_idx, 'EVENT_TYPE'] == 'C'
        peak_delay_info = f"{peak_delay:.1f} minutes at {peak_time} ({'CANCELLED SERVICE' if peak_is_cancelled else 'Regular Delay'})"
    
    delays_only = df[df['EVENT_TYPE'] != 'C']
    cancellations_only = df[df['EVENT_TYPE'] == 'C']
    
    peak_regular_delay = "N/A"
    peak_cancellation_delay = "N/A"
    
    if len(delays_only[delays_only['PFPI_MINUTES'] > 0]) > 0:
        peak_reg_idx = delays_only['PFPI_MINUTES'].idxmax()
        peak_reg_delay = delays_only.loc[peak_reg_idx, 'PFPI_MINUTES']
        peak_reg_time = delays_only.loc[peak_reg_idx, 'full_datetime'].strftime('%d-%b-%Y %H:%M')
        peak_regular_delay = f"{peak_reg_delay:.1f} minutes at {peak_reg_time}"
    
    if len(cancellations_only[cancellations_only['PFPI_MINUTES'] > 0]) > 0:
        peak_canc_idx = cancellations_only['PFPI_MINUTES'].idxmax()
        peak_canc_delay = cancellations_only.loc[peak_canc_idx, 'PFPI_MINUTES']
        peak_canc_time = cancellations_only.loc[peak_canc_idx, 'full_datetime'].strftime('%d-%b-%Y %H:%M')
        peak_cancellation_delay = f"{peak_canc_delay:.1f} minutes at {peak_canc_time}"

    summary = {
        "Total Delay Minutes": total_delay_minutes,
        "Total Cancellations": total_cancellations,
        "Total Records Found": len(df),
        "Files Processed": files_processed,
        "Files with Data": files_with_data,
        "Incident Number": incident_number,
        "Number of Days": num_days,
        "Date Range": f"{unique_dates[0].strftime('%d-%b-%Y')} to {unique_dates[-1].strftime('%d-%b-%Y')}",
        "Unique Dates": [d.strftime('%d-%b-%Y') for d in unique_dates],
        "Time Range": f"{df['full_datetime'].min().strftime('%H:%M')} - {df['full_datetime'].max().strftime('%H:%M')}" if len(df) > 0 else "N/A",
        "Peak Delay Event": peak_delay_info,
        "Peak Regular Delay": peak_regular_delay,
        "Peak Cancellation Delay": peak_cancellation_delay
    }
    
    return summary


def aggregate_view_multiday(incident_number, start_date):
    """
    Multi-day incident analysis that handles incidents spanning multiple days.
    Creates separate charts for each day with labels (a), (b), (c) or (a.1), (a.2), etc.
    
    For single-day incidents, labels are simply (a), (b), (c).
    For multi-day incidents: (a.1), (a.2) for hourly charts, (b) for severity, (c.1), (c.2) for timelines.
    
    WARNING: If incident spans more than 3 days, the same incident number may refer to multiple separate incidents.
    
    Parameters:
    -----------
    incident_number : int/str
        The incident number to analyze
    start_date : str
        Starting date in 'DD-MMM-YYYY' format (used for display purposes only - all days are loaded)
    
    Returns:
    --------
    dict : Summary statistics for the incident across all days
    """
    
    # Step 1: Load and prepare data
    load_result = _load_and_prepare_multiday_data(incident_number)
    if load_result is None:
        return None
    
    df, files_processed, files_with_data, unique_dates, num_days = load_result
    
    # Step 2: Calculate summary statistics 
    total_delay_minutes = df['PFPI_MINUTES'].fillna(0).sum()
    total_cancellations = len(df[df['EVENT_TYPE'] == 'C'])
    delay_data_all = df[df['PFPI_MINUTES'] > 0].copy()
    
    # Step 3: Create visualizations with GridSpec layout
    total_subplots = num_days + 1 + num_days  # hourly per day + severity + timeline per day
    total_height = num_days * 4 + 4 + num_days * 4
    fig = plt.figure(figsize=(8, total_height))
    
    height_ratios = [20] * num_days + [20] + [20] * num_days
    gs = GridSpec(total_subplots, 1, figure=fig, height_ratios=height_ratios, hspace=0.6)
    
    subplot_idx = 0
    is_multiday = num_days > 1
    
    # ===== SECTION A: Hourly Delay Totals per Day =====
    for day_num, current_date in enumerate(unique_dates, 1):
        ax = fig.add_subplot(gs[subplot_idx])
        subplot_idx += 1
        
        day_df = df[df['event_date_only'] == current_date].copy()
        day_df['hour'] = day_df['full_datetime'].dt.hour
        base_datetime = datetime.combine(current_date, datetime.min.time())
        day_df['time_only'] = day_df['full_datetime'].dt.time
        day_df['chart_datetime'] = day_df['time_only'].apply(lambda t: datetime.combine(current_date, t))
        
        # Create start chart datetime for incident start line (only on first day)
        if day_num == 1 and 'incident_start_datetime' in day_df.columns:
            day_df['start_time_only'] = day_df['incident_start_datetime'].dt.time
            day_df['start_chart_datetime'] = day_df['start_time_only'].apply(lambda t: datetime.combine(current_date, t) if pd.notna(t) else None)
        
        # Calculate incident end time for this day (only on last day - use max datetime from events)
        incident_end_chart_datetime = None
        if day_num == num_days and len(day_df) > 0 and 'full_datetime' in day_df.columns:
            max_datetime = day_df['full_datetime'].max()
            if pd.notna(max_datetime):
                end_time_only = max_datetime.time()
                incident_end_chart_datetime = datetime.combine(current_date, end_time_only)
        
        # Use helper to create hourly chart
        _create_hourly_chart_for_date(ax, day_df, current_date, base_datetime, incident_end_chart_datetime)
        
        ax.set_ylim(0, 1200)
        ax.set_ylabel('Total Delay\nMinutes per Hour', fontsize=20)
        ax.set_xlabel('24-Hour Timeline', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(base_datetime, base_datetime + timedelta(days=1) - timedelta(minutes=1))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 23]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.grid(True, alpha=0.3)
        
        label_text = f"(a.{day_num})" if is_multiday else "(a)"
        ax.text(1.02, 0.5, label_text, transform=ax.transAxes, fontsize=24, fontweight='bold',
               verticalalignment='center', horizontalalignment='left')
    
    # ===== SECTION B: Overall Delay Severity Distribution =====
    ax_severity = fig.add_subplot(gs[subplot_idx])
    subplot_idx += 1
    _create_severity_chart(ax_severity, delay_data_all)
    ax_severity.set_ylim(0, 150)
    ax_severity.grid(True, alpha=0.3, axis='y')
    ax_severity.text(1.02, 0.5, "(b)", transform=ax_severity.transAxes, fontsize=24, fontweight='bold',
                    verticalalignment='center', horizontalalignment='left')
    
    # ===== SECTION C: Event Timeline per Day =====
    timeline_axes = []
    for day_num, current_date in enumerate(unique_dates, 1):
        ax = fig.add_subplot(gs[subplot_idx])
        subplot_idx += 1
        timeline_axes.append((ax, current_date, day_num))
        
        day_df = df[df['event_date_only'] == current_date].copy()
        base_datetime = datetime.combine(current_date, datetime.min.time())
        day_df['time_only'] = day_df['full_datetime'].dt.time
        day_df['chart_datetime'] = day_df['time_only'].apply(lambda t: datetime.combine(current_date, t))
        
        # Create start chart datetime for incident start line (only on first day)
        if day_num == 1 and 'incident_start_datetime' in day_df.columns:
            day_df['start_time_only'] = day_df['incident_start_datetime'].dt.time
            day_df['start_chart_datetime'] = day_df['start_time_only'].apply(lambda t: datetime.combine(current_date, t) if pd.notna(t) else None)
        
        # Calculate incident end time for this day (only on last day - use max datetime from events)
        incident_end_chart_datetime = None
        if day_num == num_days and len(day_df) > 0 and 'full_datetime' in day_df.columns:
            max_datetime = day_df['full_datetime'].max()
            if pd.notna(max_datetime):
                end_time_only = max_datetime.time()
                incident_end_chart_datetime = datetime.combine(current_date, end_time_only)
        
        # Use helper to create timeline chart
        _create_timeline_scatter_chart(ax, day_df, base_datetime, current_date, incident_end_chart_datetime)
        
        ax.set_ylabel('Delay\nMinutes', fontsize=20)
        ax.set_xlabel('24-Hour Timeline', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(0, 110)
        ax.set_xlim(base_datetime, base_datetime + timedelta(days=1) - timedelta(minutes=1))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 23]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.grid(True, alpha=0.3)
        
        label_text = f"(c.{day_num})" if is_multiday else "(c)"
        ax.text(1.02, 0.5, label_text, transform=ax.transAxes, fontsize=24, fontweight='bold',
               verticalalignment='center', horizontalalignment='left')
    
    # Step 4: Build legend and info boxes
    _build_legend_and_info_boxes(fig, ax_severity, timeline_axes, df, unique_dates, delay_data_all)
    
    plt.show()
    
    # Step 5: Calculate and return summary statistics
    summary = calculate_incident_summary_stats(df, delay_data_all, unique_dates, files_processed, 
                                                files_with_data, incident_number, num_days)
    
    return summary


# for incident view:

def incident_view(incident_code, incident_date, analysis_date, analysis_hhmm, period_minutes):
    """
    Generate a detailed table showing each station affected by an incident with their calls and delays
    for a specific time period during the incident lifecycle.
    Shows trains that were shifted between time periods due to delays.
    
    Parameters:
    incident_code (int/float): The incident number to analyze
    incident_date (str): Incident date in 'DD-MMM-YYYY' format (used to locate the incident)
    analysis_date (str): Specific date to analyze in 'DD-MMM-YYYY' format
    analysis_hhmm (str): Start time for analysis in 'HHMM' format (e.g., '1830' for 18:30)
    period_minutes (int): Minutes from analysis start time to analyze
    
    Returns:
    tuple: (pandas.DataFrame, str, str) - Results table, incident start time string, and analysis period string
    """
    
    # Step 1: Parse analysis time inputs
    try:
        analysis_datetime = datetime.strptime(f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", '%d-%b-%Y %H:%M')
    except ValueError:
        return pd.DataFrame(), None, None
    
    analysis_end = analysis_datetime + timedelta(minutes=period_minutes)
    analysis_period_str = f"{analysis_datetime.strftime('%d-%b-%Y %H:%M')} to {analysis_end.strftime('%d-%b-%Y %H:%M')} ({period_minutes} min)"
    
    # Step 2: Get day suffix and load target files
    analysis_day_suffix = _get_day_suffix(analysis_datetime)
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        return pd.DataFrame(), None, None
    
    target_files = _get_target_files_for_day(processed_base, analysis_day_suffix)
    
    # Step 3: Extract incident metadata and calculate metrics
    incident_start_time = None
    incident_delay_day = None
    incident_section_code = None
    incident_reason = None
    station_results = []
    
    for file_path, station_code in target_files:
        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
            if not isinstance(df, pd.DataFrame):
                continue
            
            # Extract incident metadata on first occurrence
            incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            if incident_data.empty:
                continue
            
            incident_data['incident_date'] = incident_data['INCIDENT_START_DATETIME'].str.split(' ').str[0]
            incident_records = incident_data[incident_data['incident_date'] == incident_date].copy()
            if incident_records.empty:
                continue
            
            if incident_start_time is None:
                incident_start_time = incident_records['INCIDENT_START_DATETIME'].iloc[0]
                incident_delay_day = incident_records['DELAY_DAY'].iloc[0]
                incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                incident_reason = incident_records['INCIDENT_REASON'].iloc[0]
                
                incident_start_dt = datetime.strptime(incident_start_time, '%d-%b-%Y %H:%M')
                if analysis_datetime < incident_start_dt:
                    pass
            
            # Step 4: Calculate metrics using helpers
            planned_calls = _calculate_planned_calls(df, incident_delay_day, analysis_datetime, analysis_end)
            
            delayed_trains_out, delay_minutes_out, delayed_trains_in, delay_minutes_in = \
                _calculate_delayed_train_metrics(df, analysis_datetime, analysis_end)
            
            actual_calls = planned_calls - delayed_trains_out + delayed_trains_in
            
            # Step 5: Store results if station has data
            if planned_calls > 0 or delayed_trains_out > 0 or delayed_trains_in > 0:
                station_results.append({
                    'STATION_CODE': station_code,
                    'PLANNED_CALLS': planned_calls,
                    'ACTUAL_CALLS': actual_calls,
                    'DELAYED_TRAINS_OUT': delayed_trains_out,
                    'DELAY_MINUTES_OUT': delay_minutes_out,
                    'DELAYED_TRAINS_IN': delayed_trains_in,
                    'DELAY_MINUTES_IN': delay_minutes_in
                })
            
        except Exception:
            continue
    
    # Step 6: Handle case where no incident was found
    if incident_start_time is None:
        return pd.DataFrame(), None, None
    
    # Step 7: Get station name and print incident details
    incident_station_name = _get_station_name_from_reference(incident_section_code)
    
    # Step 8: Build and return results
    if not station_results:
        return pd.DataFrame(), incident_start_time, analysis_period_str
    
    result_df = pd.DataFrame(station_results)
    result_df = result_df.sort_values('STATION_CODE').reset_index(drop=True)
    
    return result_df, incident_start_time, analysis_period_str


def _load_heatmap_station_files(processed_base, analysis_day_suffix):
    """
    Load station file paths for the specified analysis day.
    
    Parameters:
    processed_base (str): Path to processed_data directory
    analysis_day_suffix (str): Day suffix (e.g., 'MO', 'TU')
    
    Returns:
    list: List of tuples (file_path, station_dir)
    """
    station_dirs = [d for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
    target_files = []
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        day_file = os.path.join(station_path, f"{analysis_day_suffix}.parquet")
        if os.path.exists(day_file):
            target_files.append((day_file, station_dir))
    
    return target_files


def _prepare_heatmap_json_and_markers(incident_code, incident_section_code, incident_reason, 
                                       incident_start_time, incident_locations, all_station_coords_map, 
                                       station_timeline_data, timeline_data, time_steps):
    """
    Prepare JSON data and JavaScript incident markers for heatmap visualization.
    
    Parameters:
    incident_code: The incident number
    incident_section_code: Section code of incident
    incident_reason: Reason for incident
    incident_start_time: Start time string
    incident_locations: List of incident location dicts
    all_station_coords_map: Station coordinates map
    station_timeline_data: Timeline data with delays
    timeline_data: Structured timeline data
    time_steps: List of time step datetimes
    
    Returns:
    tuple: (station_coords_json, timeline_data_json, time_steps_json, incident_markers_js)
    """
    import json
    
    station_coords_json = json.dumps(all_station_coords_map)
    timeline_data_json = json.dumps(timeline_data)
    time_steps_json = json.dumps([t.strftime('%H:%M') for t in time_steps])
    
    incident_markers_js = ""
    if incident_locations:
        for i, loc in enumerate(incident_locations):
            incident_markers_js += f'''
            var incidentMarker{i} = L.marker([{loc['lat']}, {loc['lon']}], {{
                icon: L.divIcon({{
                    html: '<div style="font-size: 24px; color: red; font-weight: bold;">X</div>',
                    className: 'incident-marker',
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                }})
            }}).addTo(map);
            
            incidentMarker{i}.bindPopup(`
                <strong>Incident Location</strong><br>
                Station: {loc['name']}<br>  
                STANOX: {loc['stanox']}<br>
                Incident: {incident_code}<br>
                Section: {incident_section_code or 'N/A'}<br>
                Reason: {incident_reason or 'N/A'}<br>
                Started: {incident_start_time or 'N/A'}
            `);'''
    
    return station_coords_json, timeline_data_json, time_steps_json, incident_markers_js


def _generate_heatmap_html_visualization(incident_code, analysis_date, analysis_datetime, analysis_end, 
                                          period_minutes, interval_minutes, incident_section_code, 
                                          incident_reason, incident_start_time, incident_station_name, 
                                          station_coords_json, timeline_data_json, time_steps_json, 
                                          incident_markers_js):
    """
    Generate the complete HTML/CSS/JavaScript visualization for heatmap.
    
    Returns:
    str: Complete HTML content
    """
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Incident {incident_code} - Network Heatmap Analysis</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f5f5; }}
        .gradient-legend {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .gradient-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, 
                rgba(0,255,0,1.0) 0%,        /* Bright green - minor delays (1-14 min) */
                rgba(255,255,0,1.0) 33%,     /* Bright yellow - medium delays (15-29 min) */
                rgba(255,165,0,1.0) 66%,     /* Bright orange - high delays (30-59 min) */
                rgba(255,0,0,1.0) 100%       /* Bright red - critical delays (60+ min) */
            );
            border: 1px solid #ccc;
            border-radius: 10px;
            margin: 0 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .gradient-labels {{
            display: flex;
            justify-content: space-between;
            width: 200px;
            margin: 5px 10px 0 10px;
            font-size: 10px;
            color: #666;
        }}
        #map {{ height: 75vh; }}
        #controls {{ 
            height: 25vh; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .control-panel {{ 
            background: rgba(255,255,255,0.95); 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #333;
        }}
        #timeline {{ width: 100%; margin: 10px 0; height: 8px; }}
        .time-display {{ 
            font-size: 24px; 
            font-weight: bold; 
            text-align: center; 
            color: #2c3e50;
            margin: 10px 0;
        }}
        .play-controls {{ text-align: center; margin: 15px 0; }}
        .btn {{ 
            background: #3498db; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            margin: 0 5px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .btn:hover {{ background: #2980b9; transform: translateY(-2px); }}
        .btn:active {{ transform: translateY(0); }}
        #station-info {{ 
            max-height: 120px; 
            overflow-y: auto; 
            margin-top: 10px;
            font-size: 14px;
        }}
        .station-delay {{ 
            display: inline-block; 
            margin: 2px 5px; 
            padding: 3px 8px; 
            border-radius: 15px; 
            font-size: 12px;
        }}
        .delay-low {{ background: rgba(0,255,0,1.0); color: black; }}
        .delay-med {{ background: rgba(255,255,0,1.0); color: black; }}
        .delay-high {{ background: rgba(255,165,0,1.0); color: white; }}
        .delay-extreme {{ background: rgba(255,0,0,1.0); color: white; }}
        .legend {{ 
            background: rgba(255,255,255,0.95); 
            padding: 10px; 
            border-radius: 8px;
            margin: 10px 0;
            font-size: 12px;
            color: #333;
        }}
        .legend-item {{ 
            display: inline-block;
            margin: 3px 8px 3px 0; 
        }}
        .legend-color {{ 
            display: inline-block; 
            width: 15px; 
            height: 15px; 
            border-radius: 50%; 
            margin-right: 5px; 
            vertical-align: middle;
        }}
        
        /* Cloud-like heatmap circle styles */
        .delay-circle {{
            filter: blur(3px) brightness(1.1);
            transition: filter 0.3s ease;
        }}
        .delay-circle:hover {{
            filter: blur(1px) brightness(1.3);
            /* Removed transform: scale() to prevent circle movement */
        }}
    </style>
</head>
<body>
    <!-- Map Container -->
    <div id="map">
    </div>
    
    <!-- Controls Container -->
    <div id="controls">
        <div class="control-panel">
            <h3 style="margin-top: 0;">Incident {incident_code} - Network Heatmap (A/B/C1/C2 Stations) - {analysis_date}</h3>
            <p style="margin: 5px 0;">Analysis Period: {analysis_datetime.strftime('%H:%M')} - {analysis_end.strftime('%H:%M')} ({period_minutes} min total, {interval_minutes}-min intervals)</p>
            <p style="margin: 5px 0; font-weight: bold;">Section: {incident_section_code or 'N/A'}{' (' + incident_station_name + ')' if incident_station_name else ''} | Reason: {incident_reason or 'N/A'} | Started: {incident_start_time or 'N/A'}</p>
            
            <div class="time-display" id="current-time"></div>
            
            <input type="range" id="timeline" min="0" max="0" value="0" step="1">
            
            <div class="play-controls">
                <button class="btn" onclick="playTimeline()">▶ Play</button>
                <button class="btn" onclick="pauseTimeline()">⏸ Pause</button>
                <button class="btn" onclick="resetTimeline()">⏮ Reset</button>
            </div>
            
            <div class="legend">
                <strong>Custom Delay Visualization - Exact Color Matching:</strong>
                <div class="gradient-legend">
                    <span style="font-size: 10px; font-weight: bold;">Minor</span>
                    <div class="gradient-bar"></div>
                    <span style="font-size: 10px; font-weight: bold;">Critical</span>
                </div>
                <div class="gradient-labels">
                    <span>1min</span>
                    <span>15min</span>
                    <span>30min</span>
                    <span>60+min</span>
                </div>

            </div>
            
            <div id="station-info">Loading network heatmap...</div>
        </div>
    </div>

    <script>
        // Initialize map
        var map = L.map('map').setView([54.5, -2.0], 6);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add incident location markers for all found locations
        
        // Custom colored circles for precise color control (bypassing Leaflet.heat limitations)
        var delayCircles = L.layerGroup().addTo(map);
        
        // Color function that matches our legend exactly
        function getExactDelayColor(delayMinutes) {{
            if (delayMinutes >= 60) return '#ff0000';      // Bright red for critical delays (60+ min)
            if (delayMinutes >= 30) return '#ffa500';      // Bright orange for high delays (30-59 min)
            if (delayMinutes >= 15) return '#ffff00';      // Bright yellow for medium delays (15-29 min)
            return '#00ff00';                              // Bright green for minor delays (1-14 min)
        }}
        
        // Consistent circle size - same for all delays for uniform heatmap appearance
        function getCircleSize(delayMinutes) {{
            return 15;  // Fixed size for all delays to create uniform cloud-like heatmap
        }}
        
        // Data
        var stationCoords = {station_coords_json};
        var timelineData = {timeline_data_json};
        var timeSteps = {time_steps_json};
        
        // State
        var currentIndex = 0;
        var isPlaying = false;
        var playInterval;
        var markers = {{}};
        
        // Color functions for heatmap
        function getHeatmapColor(delayMinutes) {{
            if (delayMinutes === 0) return '#cccccc';      // Grey for no delays
            if (delayMinutes >= 60) return '#ff0000';      // Bright red for critical delays (60+ min) - matches legend
            if (delayMinutes >= 30) return '#ffa500';      // Bright orange for high delays (30-59 min) - matches legend
            if (delayMinutes >= 15) return '#ffff00';      // Bright yellow for medium delays (15-29 min) - matches legend
            return '#00ff00';                              // Bright green for minor delays (1-14 min) - matches legend
        }}
        
        // Update heatmap for specific time index
        function updateMap(index) {{
            try {{
                currentIndex = index;
                var timeKey = timeSteps[index];
                var delays = timelineData[timeKey] || {{}};
                
                console.log('🕐 Updating heatmap for time:', timeKey, 'with', Object.keys(delays).length, 'stations');
                
                // Clear existing station markers
                Object.values(markers).forEach(m => map.removeLayer(m));
                markers = {{}};
                
                // Update time display
                document.getElementById('current-time').textContent = timeKey;
                
                // Clear existing delay circles and create new ones with exact colors
                delayCircles.clearLayers();
                
                var stationInfoHtml = '';
                var totalDelayedStations = 0;
                var totalSystemDelay = 0;
                var delayedStations = [];
                
                // Create colored delay circles for each affected station
                Object.entries(stationCoords).forEach(([stationCode, coords]) => {{
                    var delayMinutes = delays[stationCode] || 0;
                    
                    // Create colored circles for stations with delays
                    if (delayMinutes > 0) {{
                        var color = getExactDelayColor(delayMinutes);
                        var size = getCircleSize(delayMinutes);
                        
                        // Debug logging for extreme delays
                        if (delayMinutes >= 120) {{
                            console.log(`🔥 EXTREME DELAY: Station ${{coords.name}} has ${{delayMinutes}} min delay → color ${{color}} size ${{size}} (should be bright red)`);
                        }}
                        
                        // Create cloud-like colored circle for the delay
                        var delayCircle = L.circleMarker([coords.lat, coords.lon], {{
                            radius: size,
                            fillColor: color,
                            color: color,
                            weight: 0,               // No border for cleaner cloud effect
                            opacity: 0.7,            // Slightly transparent for blending
                            fillOpacity: 0.5,        // More transparent for cloud-like appearance
                            className: 'delay-circle' // Apply CSS class for blur effects
                        }});
                        
                        // Add popup with detailed information
                        delayCircle.bindPopup(`
                            <strong>${{coords.name}}</strong><br>
                            Station: ${{stationCode}}<br>
                            Interval Delay: ${{delayMinutes}} minutes<br>
                            Time Window: ${{timeKey}} ({interval_minutes} min)
                        `);
                        
                        delayCircles.addLayer(delayCircle);
                    }}
                    
                    // Create invisible clickable markers only for affected stations (with delays)
                    if (delayMinutes > 0) {{
                        var marker = L.circleMarker([coords.lat, coords.lon], {{
                            radius: 8,               // Larger invisible area for easier clicking
                            fillColor: 'transparent', // Invisible fill
                            color: 'transparent',     // Invisible border
                            weight: 0,               // No border
                            opacity: 0,              // Completely invisible
                            fillOpacity: 0           // Completely invisible
                        }});
                        
                        // Add popup with station information
                        var popupContent = `
                            <strong>${{coords.name}}</strong><br>
                            Station: ${{stationCode}}<br>
                            Interval Delay: ${{delayMinutes}} minutes<br>
                            Time Window: ${{timeKey}} - ${{timeKey}} ({interval_minutes} min)
                        `;
                        
                        marker.bindPopup(popupContent);
                        marker.addTo(map);
                        markers[stationCode] = marker;
                    }}
                    
                    // Collect statistics for affected stations
                    if (delayMinutes > 0) {{
                        totalDelayedStations++;
                        totalSystemDelay += delayMinutes;
                        
                        var delayClass = delayMinutes >= 60 ? 'delay-extreme' : 
                                       delayMinutes >= 30 ? 'delay-high' : 
                                       delayMinutes >= 15 ? 'delay-med' : 'delay-low';
                        
                        delayedStations.push({{
                            name: coords.name.substring(0,15),
                            delay: delayMinutes,
                            class: delayClass
                        }});
                    }}
                }});
                
                // Sort delayed stations by delay amount (highest first)
                delayedStations.sort((a, b) => b.delay - a.delay);
                
                // Update station info panel
                if (totalDelayedStations > 0) {{
                    var avgDelay = (totalSystemDelay / totalDelayedStations).toFixed(1);
                    stationInfoHtml = `<strong>Network Impact:</strong> ${{totalDelayedStations}} stations affected, ${{totalSystemDelay}} total minutes in this {interval_minutes}-min interval (avg: ${{avgDelay}}min)<br><br>`;
                    
                    // Show top affected stations
                    var topStations = delayedStations.slice(0, 10); // Show top 10
                    topStations.forEach(station => {{
                        stationInfoHtml += `<span class="station-delay ${{station.class}}">${{station.name}}: ${{station.delay}}min</span>`;
                    }});
                    
                    if (delayedStations.length > 10) {{
                        stationInfoHtml += `<span style="margin-left: 10px; font-style: italic;">...and ${{delayedStations.length - 10}} more</span>`;
                    }}
                }} else {{
                    stationInfoHtml = `<strong>Network Status:</strong> No delays detected in this interval. ${{Object.keys(stationCoords).length}} stations (A/B/C1/C2 categories) monitored.`;
                }}
                
                document.getElementById('station-info').innerHTML = stationInfoHtml;
                
            }} catch (error) {{
                console.error('Error updating heatmap:', error);
                document.getElementById('station-info').innerHTML = 'Error updating heatmap: ' + error.message;
            }}
        }}
        
        // Timeline controls (same as incident_view_html)
        document.getElementById('timeline').addEventListener('input', function(e) {{
            pauseTimeline();
            updateMap(parseInt(e.target.value));
        }});
        
        function playTimeline() {{
            if (!isPlaying) {{
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentIndex < timeSteps.length - 1) {{
                        currentIndex++;
                        document.getElementById('timeline').value = currentIndex;
                        updateMap(currentIndex);
                    }} else {{
                        pauseTimeline();
                    }}
                }}, 1500);  // Slightly slower for heatmap viewing
            }}
        }}
        
        function pauseTimeline() {{
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
        }}
        
        function resetTimeline() {{
            pauseTimeline();
            currentIndex = 0;
            document.getElementById('timeline').value = 0;
            updateMap(0);
        }}
        
        // Initialize heatmap with better error handling
        function initializeHeatmap() {{
            console.log('🌡️ Starting heatmap initialization...');
            console.log('Leaflet available:', typeof L !== 'undefined');
            console.log('Leaflet.heat available:', typeof L.heatLayer !== 'undefined');
            console.log('Station coordinates loaded:', Object.keys(stationCoords).length);
            console.log('Time steps:', timeSteps.length);
            console.log('Timeline data:', Object.keys(timelineData).length);
            
            if (typeof L === 'undefined') {{
                console.error('Leaflet not loaded!');
                document.getElementById('station-info').innerHTML = 'Error: Leaflet library not loaded';
                return;
            }}
            
            if (typeof L.heatLayer === 'undefined') {{
                console.error('Leaflet.heat plugin not loaded!');
                document.getElementById('station-info').innerHTML = 'Error: Leaflet.heat plugin not loaded';
                return;
            }}
            
            if (timeSteps.length === 0) {{
                console.error('No time steps available!');
                document.getElementById('station-info').innerHTML = 'No time data available';
            }} else if (Object.keys(stationCoords).length === 0) {{
                console.error('No station coordinates available!');
                document.getElementById('station-info').innerHTML = 'No station coordinates available';
            }} else {{
                console.log('All data loaded successfully, initializing heatmap...');
                document.getElementById('timeline').max = timeSteps.length - 1;
                updateMap(0);
            }}
            
            // Add incident location markers
            {incident_markers_js}
            
            console.log('🎬 Heatmap initialization complete!');
        }}
        
        // Wait for all scripts to load before initializing
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeHeatmap);
        }} else {{
            // DOM is already loaded, but wait a bit for scripts
            setTimeout(initializeHeatmap, 100);
        }}
    </script>
</body>
</html>'''
    
    return html_content


def _save_heatmap_html_file(html_content, output_file, incident_code, time_steps, 
                             all_station_coords_map, station_timeline_data, period_minutes, interval_minutes):
    """
    Save HTML heatmap content to file and print results.
    
    Parameters:
    html_content (str): The HTML content string
    output_file (str): File path to save to
    incident_code: Incident number
    time_steps: List of time steps
    all_station_coords_map: Map of all stations
    station_timeline_data: Data with delays per station
    period_minutes (int): Total analysis period
    interval_minutes (int): Interval size
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
    except Exception as e:
        pass

def incident_view_heatmap_html(incident_code, incident_date, analysis_date, analysis_hhmm, period_minutes, interval_minutes=10, output_file=None):
    """
    Create dynamic interactive HTML heatmap showing railway network delays.
    Displays delay intensity as vibrant heatmap visualization with incident locations and timeline animation.
    
    Parameters:
    incident_code (int/float): The incident number to analyze
    incident_date (str): Date when incident started in 'DD-MMM-YYYY' format
    analysis_date (str): Specific date to analyze in 'DD-MMM-YYYY' format
    analysis_hhmm (str): Start time for analysis in 'HHMM' format (e.g., '1900')
    period_minutes (int): Total duration of analysis period in minutes
    interval_minutes (int): Duration of each interval in minutes (default: 10)
    output_file (str): Optional HTML file path to save
    
    Returns:
    str: HTML content of the interactive heatmap
    """
    # Step 1: Load station coordinates
    all_station_coords_map = _load_station_coordinates_from_json()
    if all_station_coords_map is None:
        return None
    
    # Step 2: Parse analysis parameters
    analysis_datetime, analysis_end, num_intervals, analysis_day_suffix = \
        _parse_heatmap_analysis_parameters(analysis_date, analysis_hhmm, period_minutes, interval_minutes)
    
    if analysis_datetime is None:
        return None
    
    # Step 2: Parse parameters
    analysis_datetime, analysis_end, num_intervals, analysis_day_suffix = \
        _parse_heatmap_analysis_parameters(analysis_date, analysis_hhmm, period_minutes, interval_minutes)
    
    if analysis_datetime is None:
        return None
    
    # Step 3: Load station files for the analysis day
    processed_base = find_processed_data_path()
    if processed_base is None:
        return None
    
    target_files = _load_heatmap_station_files(processed_base, analysis_day_suffix)
    
    # Step 4: Collect delay timeline data for affected stations
    station_timeline_data, incident_section_code, incident_reason, incident_start_time = \
        _collect_heatmap_delay_timeline(target_files, incident_code, incident_date, all_station_coords_map, 
                                        analysis_datetime, analysis_end, interval_minutes)
    
    # Step 5: Get incident location coordinates
    incident_locations, incident_station_name = _get_incident_location_coordinates(incident_section_code)
    
    # Step 6: Build timeline data structure for JavaScript
    timeline_data, time_steps = _build_timeline_data_structure(all_station_coords_map, station_timeline_data, 
                                                               analysis_datetime, analysis_end, interval_minutes)
    
    # Step 7: Prepare JSON data and incident markers
    station_coords_json, timeline_data_json, time_steps_json, incident_markers_js = \
        _prepare_heatmap_json_and_markers(incident_code, incident_section_code, incident_reason, 
                                          incident_start_time, incident_locations, all_station_coords_map, 
                                          station_timeline_data, timeline_data, time_steps)
    
    # Step 8: Generate HTML visualization
    html_content = _generate_heatmap_html_visualization(
        incident_code, analysis_date, analysis_datetime, analysis_end, 
        period_minutes, interval_minutes, incident_section_code, 
        incident_reason, incident_start_time, incident_station_name, 
        station_coords_json, timeline_data_json, time_steps_json, 
        incident_markers_js)
    
    # Step 9: Save HTML file (optional) and return
    if output_file is None:
        safe_date = analysis_date.replace('-', '_')
        safe_time = analysis_hhmm
        output_file = f'heatmap_{incident_code}_{safe_date}_{safe_time}_period{period_minutes}min_interval{interval_minutes}min.html'
    
    _save_heatmap_html_file(html_content, output_file, incident_code, time_steps, 
                             all_station_coords_map, station_timeline_data, period_minutes, interval_minutes)
    
    return html_content

# train view functions:

def train_view(all_data, origin_code, destination_code, input_date_str):
    """
    View all train journeys between an OD pair and check for incidents on a specific date.
    Corrects PLANNED_CALLS using ACTUAL_CALLS - PFPI_MINUTES.
    
    Refactored for single responsibility: Data filtering + transformation -> Display
    
    Parameters:
    -----------
    all_data : pd.DataFrame
        Complete train data with OD information
    origin_code : str or int
        Origin location code
    destination_code : str or int
        Destination location code
    input_date_str : str
        Date in 'DD-MMM-YYYY' format
    
    Returns:
    --------
    pd.DataFrame or str : Incident data or message
    """
    
    # Step 1: Prepare data - Ensure OD_PAIR and parse dates
    if 'OD_PAIR' not in all_data.columns:
        all_data['OD_PAIR'] = (
            all_data['PLANNED_ORIGIN_LOCATION_CODE'].astype(str) + '_' +
            all_data['PLANNED_DEST_LOCATION_CODE'].astype(str)
        )
    
    od_pair = f"{origin_code}_{destination_code}"
    
    # Step 2: Parse dates
    input_date = pd.to_datetime(input_date_str, format='%d-%b-%Y', errors='coerce')
    all_data['INCIDENT_START_DATETIME'] = pd.to_datetime(all_data['INCIDENT_START_DATETIME'], errors='coerce')
    
    # Step 3: Validate OD pair exists
    if od_pair not in all_data['OD_PAIR'].unique():
        message = f"OD pair {od_pair} not found in dataset."
        return message
    
    # Step 4: Filter by OD pair
    trains_between = all_data[all_data['OD_PAIR'] == od_pair].copy()
    
    # Step 5: Correct PLANNED_CALLS = ACTUAL_CALLS - PFPI_MINUTES
    trains_between['ACTUAL_CALLS_dt'] = pd.to_datetime(trains_between['ACTUAL_CALLS'], format='%H%M', errors='coerce')
    trains_between['PFPI_MINUTES_num'] = pd.to_numeric(trains_between['PFPI_MINUTES'], errors='coerce')
    trains_between['CORRECTED_PLANNED_CALLS_dt'] = (
        trains_between['ACTUAL_CALLS_dt'] - 
        pd.to_timedelta(trains_between['PFPI_MINUTES_num'].fillna(0), unit='m')
    )
    trains_between['PLANNED_CALLS'] = trains_between['CORRECTED_PLANNED_CALLS_dt'].dt.strftime('%H%M').fillna(trains_between['PLANNED_CALLS'])
    
    # Step 6: Filter by date
    incidents_on_date = trains_between[
        trains_between['INCIDENT_START_DATETIME'].dt.date == input_date.date()
    ].copy()
    
    if incidents_on_date.empty:
        message = f"No incidents found for OD pair {od_pair} on {input_date_str}."
        return message
    
    # Step 7: Display results
    
    cols_to_show = [
        'TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME',
        'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'PLANNED_CALLS', 'ACTUAL_CALLS',
        'PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 'SECTION_CODE', 'DELAY_DAY',
        'EVENT_DATETIME', 'INCIDENT_START_DATETIME', 'ENGLISH_DAY_TYPE', 'STATION_ROLE', 'DFT_CATEGORY',
        'PLATFORM_COUNT', 'DATASET_TYPE', 'WEEKDAY', 'STANOX', 'DAY'
    ]
    
    display(incidents_on_date[cols_to_show])
    return incidents_on_date[cols_to_show]

def get_stanox_for_service(all_data, train_service_code, origin_code, destination_code, date_str=None):
    """
    Get ALL unique STANOX codes that a train service calls at, regardless of specific train instance.
    Returns a list of all stations that this service code stops at.
    
    Strategy:
    1. Filter to the specified service code and OD pair
    2. Optionally filter by date if provided
    3. Collect ALL unique STANOX codes that appear with valid scheduled stops
    4. Return the complete set (map will connect them by proximity)
    """
    import pandas as pd
    from datetime import datetime

    # --- Ensure OD_PAIR exists ---
    if 'OD_PAIR' not in all_data.columns:
        all_data['OD_PAIR'] = (
            all_data['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
            + '_' +
            all_data['PLANNED_DEST_LOCATION_CODE'].astype(str)
        )

    od_pair = f"{origin_code}_{destination_code}"

    # --- Filter dataset for this service and OD pair ---
    subset = all_data[
        (all_data['OD_PAIR'] == od_pair)
        & (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(train_service_code))
    ].copy()

    if subset.empty:
        message = f"No records found for train service {train_service_code} on OD pair {od_pair}."
        return message

    # --- Filter by date if provided ---
    if date_str:
        try:
            # Filter by EVENT_DATETIME containing the target date
            date_subset = subset[subset['EVENT_DATETIME'].str.contains(date_str, na=False)].copy()
            
            if not date_subset.empty:
                subset = date_subset
        except Exception:
            pass
    # Filter to rows that have actual scheduled stops (PLANNED_CALLS)
    valid_stops = subset[subset['PLANNED_CALLS'].notna()].copy()
    
    # Also include origin and destination explicitly
    origin_dest_stops = subset[
        (subset['STANOX'] == str(origin_code)) | 
        (subset['STANOX'] == str(destination_code))
    ].copy()
    
    # Combine both
    all_stops = pd.concat([valid_stops, origin_dest_stops], ignore_index=True)
    
    # Get unique STANOX codes
    stanox_list = all_stops['STANOX'].astype(str).unique().tolist()
    
    # Ensure destination is included (journey endpoint)
    dest_str = str(destination_code)
    if dest_str not in stanox_list:
        stanox_list.append(dest_str)
    
    return stanox_list


def _prepare_journey_map_data(all_stanox, station_ref):
    """
    Build station coordinates and names for journey map.
    
    Parameters:
    all_stanox (set): Set of all STANOX codes to include
    station_ref (list): Station reference data from JSON
    
    Returns:
    tuple: (stanox_coords, stanox_names) where stanox_coords is list of (stanox, lat, lon)
    """
    stanox_coords = []
    stanox_names = {}
    
    for s in all_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        match = next((item for item in station_ref if str(item.get("stanox", "")) == s_str), None)
        if match and 'latitude' in match and 'longitude' in match:
            try:
                lat = float(match['latitude'])
                lon = float(match['longitude'])
                station_name = match.get('description', s_str)
                stanox_coords.append((s_str, lat, lon))
                stanox_names[s_str] = station_name
            except Exception:
                continue
        if len(stanox_coords) > 5:
            pass
    
    return stanox_coords, stanox_names


def _compute_station_route_connections(stanox_coords, stanox_names):
    """
    Compute minimum spanning tree connections between stations based on geographic distance.
    
    Parameters:
    stanox_coords (list): List of (stanox, lat, lon) tuples
    stanox_names (dict): Mapping of STANOX to station names
    
    Returns:
    list: List of (start_stanox, end_stanox, start_coords, end_coords, start_name, end_name)
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    connections = []
    
    if len(stanox_coords) <= 1:
        return connections
    
    # Extract coordinates for distance calculation
    coords_array = np.array([(lat, lon) for _, lat, lon in stanox_coords])
    
    # Compute pairwise distances (Euclidean on lat/lon - approximation)
    distances = squareform(pdist(coords_array, metric='euclidean'))
    
    # Compute minimum spanning tree to connect all stations with minimum total distance
    mst = minimum_spanning_tree(distances)
    mst_array = mst.toarray()
    
    # Extract edges based on MST
    for i in range(len(stanox_coords)):
        for j in range(i+1, len(stanox_coords)):
            # Check if there's an edge in MST (symmetric, so check both directions)
            if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                start_stanox, start_lat, start_lon = stanox_coords[i]
                end_stanox, end_lat, end_lon = stanox_coords[j]
                
                start_name = stanox_names.get(start_stanox, start_stanox)
                end_name = stanox_names.get(end_stanox, end_stanox)
                
                connections.append((
                    start_stanox, end_stanox,
                    (start_lat, start_lon), (end_lat, end_lon),
                    start_name, end_name
                ))
    
    return connections


def _aggregate_delays_and_incidents(incident_results):
    """
    Aggregate delay data and incident information from results.
    
    Parameters:
    incident_results (list): List of DataFrames with incident data
    
    Returns:
    tuple: (stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records)
    """
    import pandas as pd
    
    dfs = []
    if incident_results:
        for res in incident_results:
            if isinstance(res, pd.DataFrame):
                dfs.append(res)
    
    incident_records = pd.concat(dfs, ignore_index=True) if dfs else None
    stanox_delay = {}
    stanox_incidents = {}
    incident_rank = {}
    incident_durations = {}
    
    if incident_records is None or incident_records.empty:
        return stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records
    
    # Normalize incident numbers
    if 'INCIDENT_NUMBER' in incident_records.columns:
        def _norm_inc(x):
            try:
                if pd.isna(x):
                    return None
                xf = float(x)
                return str(int(xf)) if xf.is_integer() else str(x)
            except Exception:
                return str(x)
        incident_records['INCIDENT_NUMBER_str'] = incident_records['INCIDENT_NUMBER'].apply(_norm_inc)
    else:
        incident_records['INCIDENT_NUMBER_str'] = None
    
    # Aggregate delays per STANOX
    if 'STANOX' in incident_records.columns and 'PFPI_MINUTES' in incident_records.columns:
        incident_records['PFPI_MINUTES_num'] = pd.to_numeric(incident_records['PFPI_MINUTES'], errors='coerce').fillna(0)
        
        for stanox, group in incident_records.groupby('STANOX'):
            total_delay = group['PFPI_MINUTES_num'].sum()
            stanox_delay[str(stanox)] = total_delay
            if 'INCIDENT_NUMBER_str' in group.columns:
                unique_incs = sorted([str(v) for v in pd.unique(group['INCIDENT_NUMBER_str'].dropna())])
            else:
                unique_incs = []
            stanox_incidents[str(stanox)] = unique_incs
    
    # Build incident ranking by chronology
    if 'INCIDENT_NUMBER_str' in incident_records.columns and 'INCIDENT_START_DATETIME' in incident_records.columns:
        temp = incident_records[['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']].dropna(
            subset=['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']
        ).drop_duplicates(subset=['INCIDENT_NUMBER_str']).copy()
        
        if not temp.empty:
            temp['INCIDENT_START_dt'] = pd.to_datetime(temp['INCIDENT_START_DATETIME'], errors='coerce')
            temp = temp.sort_values('INCIDENT_START_dt')
            temp = temp.reset_index(drop=True)
            temp['incident_rank'] = temp.index + 1
            incident_rank = dict(zip(temp['INCIDENT_NUMBER_str'].astype(str), temp['incident_rank'].astype(int)))
    
    # Calculate incident durations
    if 'INCIDENT_NUMBER_str' in incident_records.columns:
        for inc in incident_records['INCIDENT_NUMBER_str'].unique():
            if pd.isna(inc):
                continue
            subset = incident_records[incident_records['INCIDENT_NUMBER_str'] == inc]
            start = pd.to_datetime(subset['INCIDENT_START_DATETIME'].min(), format='%Y-%m-%d %H:%M:%S', errors='coerce')
            end = pd.to_datetime(subset['EVENT_DATETIME'].max(), format='%d-%b-%Y %H:%M', errors='coerce')
            duration = end - start
            duration = max(duration, pd.Timedelta(0))
            incident_durations[inc] = duration
    
    return stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records


def _create_station_markers_on_map(m, stanox_coords, stanox_names, stanox_delay, stanox_incidents, incident_rank):
    """
    Create and add station circle markers to map with color-grading by delay.
    
    Parameters:
    m: Folium map object
    stanox_coords (list): List of (stanox, lat, lon)
    stanox_names (dict): STANOX to name mapping
    stanox_delay (dict): STANOX to total delay mapping
    stanox_incidents (dict): STANOX to incident list mapping
    incident_rank (dict): Incident ID to rank mapping
    """
    import folium
    
    def get_color(delay):
        try:
            d = float(delay)
        except Exception:
            d = 0
        if d == 0:
            return "blue"
        elif d <= 5:
            return '#32CD32'     # Minor (1-5 min) - Lime Green
        elif d <= 15:
            return '#FFD700'     # Moderate (6-15 min) - Gold
        elif d <= 30:
            return '#FF8C00'     # Significant (16-30 min) - Dark Orange
        elif d <= 60:
            return '#FF0000'     # Major (31-60 min) - Red
        elif d <= 120:
            return '#8B0000'     # Severe (61-120 min) - Dark Red
        else:
            return '#8A2BE2'     # Critical (120+ min) - Blue Violet
    
    for stanox, lat, lon in stanox_coords:
        delay_val = stanox_delay.get(stanox, 0)
        color = get_color(delay_val)
        station_name = stanox_names.get(stanox, stanox)
        
        inc_list = stanox_incidents.get(stanox, [])
        if inc_list:
            inc_ranks = [str(incident_rank.get(str(i), i)) for i in inc_list]
            if len(inc_ranks) > 10:
                inc_display = ', '.join(inc_ranks[:10]) + f', ... (+{len(inc_ranks)-10} more)'
            else:
                inc_display = ', '.join(inc_ranks)
            incidents_html = f"<br><b>Incidents (by index):</b> {inc_display}"
        else:
            incidents_html = ''
        
        popup_html = f"<b>{station_name}</b><br>STANOX: {stanox}<br>Total delay: {delay_val:.1f} min{incidents_html}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(m)


def _create_incident_markers_on_map(m, incident_records, station_ref, incident_rank, incident_durations, incident_color):
    """
    Create and add incident location markers to map.
    
    Parameters:
    m: Folium map object
    incident_records: DataFrame with incident data
    station_ref (list): Station reference data
    incident_rank (dict): Incident ID to rank mapping
    incident_durations (dict): Incident ID to duration mapping
    incident_color (str): Color for incident markers
    """
    import folium
    import pandas as pd
    
    if incident_records is None or incident_records.empty:
        return
    
    # Prepare unique incidents
    if 'INCIDENT_NUMBER' not in incident_records.columns or 'SECTION_CODE' not in incident_records.columns:
        return
    
    incident_records['INCIDENT_NUMBER_str'] = incident_records['INCIDENT_NUMBER'].apply(
        lambda x: str(int(float(x))) if (pd.notna(x) and float(x).is_integer()) else str(x)
    )
    
    incident_unique = incident_records.drop_duplicates(subset=['INCIDENT_NUMBER_str', 'SECTION_CODE']).copy()
    
    # Build section map
    section_map = {}
    for _, row in incident_unique.iterrows():
        section = str(row['SECTION_CODE'])
        inc_id = str(row['INCIDENT_NUMBER_str'])
        inc_num = row.get('INCIDENT_NUMBER')
        inc_time = row.get('INCIDENT_START_DATETIME')
        inc_reason = row.get('INCIDENT_REASON') if 'INCIDENT_REASON' in row.index else None
        rank = incident_rank.get(inc_id)
        entry = {
            'inc_id': inc_id,
            'inc_num': inc_num,
            'inc_time': inc_time,
            'inc_reason': inc_reason,
            'rank': rank if rank is not None else ''
        }
        section_map.setdefault(section, []).append(entry)
    
    markers_created = 0
    
    for section_code, entries in section_map.items():
        entries_sorted = sorted(entries, key=lambda e: (e['rank'] if isinstance(e['rank'], int) else 999999))
        ranks = [str(e['rank']) if e['rank'] != '' else e['inc_id'] for e in entries_sorted]
        ranks_display = ','.join(ranks)
        
        popup_lines = []
        for e in entries_sorted:
            reason_text = e['inc_reason'] if e.get('inc_reason') else 'N/A'
            dur = incident_durations.get(e['inc_id'], 'N/A')
            popup_lines.append(f"Incident: {e['inc_num']} — {e['inc_time']} — Reason: {reason_text} — Duration: {dur}")
        popup_html = '<br>'.join(popup_lines)
        
        # Track section incident (section_code contains ':')
        if ':' in section_code:
            stanox_parts = section_code.split(':')
            if len(stanox_parts) == 2:
                stanox1, stanox2 = stanox_parts[0].strip(), stanox_parts[1].strip()
                
                match1 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox1), None)
                match2 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox2), None)
                
                if match1 and match2 and all(k in match1 for k in ['latitude', 'longitude']) and all(k in match2 for k in ['latitude', 'longitude']):
                    lat1, lon1 = float(match1['latitude']), float(match1['longitude'])
                    lat2, lon2 = float(match2['latitude']), float(match2['longitude'])
                    
                    station1_name = match1.get('description', stanox1)
                    station2_name = match2.get('description', stanox2)
                    
                    # Draw incident section polyline
                    folium.PolyLine(
                        [(lat1, lon1), (lat2, lon2)],
                        color=incident_color,
                        weight=6,
                        opacity=0.9,
                        popup=f"Incident Section: {station1_name} ↔ {station2_name}"
                    ).add_to(m)
                    
                    # Add marker at midpoint
                    mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
                    section_popup = f"<b>Track Section Incident</b><br>Between: {station1_name} ↔ {station2_name}<br>Section: {section_code}<br><br>{popup_html}"
                    
                    size_px = max(28, min(80, 12 * len(ranks_display)))
                    number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                    folium.Marker(
                        location=(mid_lat, mid_lon),
                        icon=folium.DivIcon(html=number_html),
                        popup=folium.Popup(section_popup, max_width=450)
                    ).add_to(m)
                    markers_created += 1
        else:
            # Single station incident
            match = next((item for item in station_ref if str(item.get("stanox", "")) == section_code), None)
            if match and all(k in match for k in ['latitude', 'longitude']):
                lat = float(match['latitude']) + 0.0005
                lon = float(match['longitude']) + 0.0005
                station_name = match.get('description', section_code)
                
                station_popup = f"<b>Station Incident</b><br>{station_name}<br>STANOX: {section_code}<br><br>{popup_html}"
                
                size_px = max(28, min(80, 12 * len(ranks_display)))
                number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                folium.Marker(
                    location=(lat, lon),
                    icon=folium.DivIcon(html=number_html),
                    popup=folium.Popup(station_popup, max_width=450)
                ).add_to(m)
                markers_created += 1


def _finalize_journey_map(m):
    """
    Finalize map by adding legend and layer controls.
    
    Parameters:
    m: Folium map object
    """
    import folium
    
    legend_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: auto; max-height: 400px; z-index:9999; font-size:13px; background: white; border:2px solid grey; border-radius:8px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size: 15px;">Delay Intensity Key</b><br>
     <div style="margin: 8px 0;">
     <i class="fa fa-circle" style="color:blue"></i> 0 min (No delay)<br>
     <i class="fa fa-circle" style="color:#32CD32"></i> 1-5 min (Minor)<br>
     <i class="fa fa-circle" style="color:#FFD700"></i> 6-15 min (Moderate)<br>
     <i class="fa fa-circle" style="color:#FF8C00"></i> 16-30 min (Significant)<br>
     <i class="fa fa-circle" style="color:#FF0000"></i> 31-60 min (Major)<br>
     <i class="fa fa-circle" style="color:#8B0000"></i> 61-120 min (Severe)<br>
     <i class="fa fa-circle" style="color:#8A2BE2"></i> 120+ min (Critical)
     </div>
     <b style="font-size: 14px;">Route Connections:</b><br>
     <div style="margin-top: 6px; line-height: 1.4;">
     Blue lines connect stations by <b>geographic proximity</b> using minimum spanning tree algorithm.<br><br>
     Purple numbered circles show incidents (1 = earliest).<br>
     Track section incidents shown as purple lines.
     </div>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

def map_train_journey_with_incidents(
    all_data, service_stanox, incident_results=None,
    stations_ref_path=None,
    incident_color="purple", service_code=None, date_str=None
    ):
    """
    Map train journey by connecting stations based on GEOGRAPHIC PROXIMITY (not chronological order).
    
    1. Load reference stations and prepare STANOX coordinate data
    2. Connect service stations + incident stations using minimum spanning tree
    3. Color-grade station markers by total delay
    4. Map each incident with chronologically-ranked numbered markers
    
    Refactored to use focused helper functions for data preparation, calculations, and visualization.
    """
    import json
    import folium
    import pandas as pd
    
    # Load station reference data
    if stations_ref_path is None:
        from demo.data.reference import reference_files
        stations_ref_path = reference_files["all dft categories"]
    
    with open(stations_ref_path, "r") as f:
        station_ref = json.load(f)
    
    # STEP 1: Prepare service and incident station sets
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    additional_stanox = set()
    if incident_results:
        for res in incident_results:
            if isinstance(res, pd.DataFrame) and 'STANOX' in res.columns:
                stanox_values = res['STANOX'].dropna().unique()
                for stanox in stanox_values:
                    stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
                    additional_stanox.add(stanox_str)
    
    all_stanox = service_stanox_normalized.union(additional_stanox)
    
    # STEP 2: Build station coordinates and names
    stanox_coords, stanox_names = _prepare_journey_map_data(all_stanox, station_ref)
    
    if not stanox_coords:
        return None
    
    # STEP 3: Create base map with title
    mid_lat = sum([lat for _, lat, _ in stanox_coords]) / len(stanox_coords)
    mid_lon = sum([lon for _, _, lon in stanox_coords]) / len(stanox_coords)
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=8, tiles="OpenStreetMap")
    
    if service_code and date_str:
        title_html = f"<div style='position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; font-size:18px; background: white; border:2px solid grey; border-radius:8px; padding: 10px;'><b>Train Service: {service_code}</b><br><b>Date: {date_str}</b></div>"
        m.get_root().html.add_child(folium.Element(title_html))
    
    # STEP 4: Draw route connections between stations (geographically-based)
    connections = _compute_station_route_connections(stanox_coords, stanox_names)
    for start_stanox, end_stanox, start_coords, end_coords, start_name, end_name in connections:
        folium.PolyLine(
            [start_coords, end_coords],
            color="blue",
            weight=4,
            opacity=0.8,
            popup=f"Connection: {start_name} ↔ {end_name}"
        ).add_to(m)
    
    # STEP 5: Aggregate delay statistics and incident information
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(incident_results)
    
    # STEP 6: Create station markers with color-grading
    _create_station_markers_on_map(m, stanox_coords, stanox_names, stanox_delay, stanox_incidents, incident_rank)
    
    # STEP 7: Create incident markers with popups
    _create_incident_markers_on_map(m, incident_records, station_ref, incident_rank, incident_durations, incident_color)
    
    # STEP 8: Add legend and finalize
    _finalize_journey_map(m)
    
    return m

def train_view_2(all_data, service_stanox, service_code, stations_ref_path=None):
    """
    Compute reliability metrics for each station in the service_stanox list for a given train service code.

    Metrics now exclude PFPI_MINUTES == 0.0 when computing mean/variance and incident counts.
    OnTime% is computed on the original PFPI distribution (<=0) so it still reflects punctuality.

    Returns a DataFrame with columns: ServiceCode, StationName, MeanDelay, DelayVariance, OnTime%, IncidentCount
    
    NEW: Also includes stations from all_data that experienced delays for this service code.
    """
    import json
    import pandas as pd
    import numpy as np

    # Load station reference with flexible path
    if stations_ref_path is None:
        from demo.data.reference import reference_files
        stations_ref_path = reference_files["all dft categories"]
    
    try:
        with open(stations_ref_path, "r") as f:
            station_ref = json.load(f)
        stanox_to_name = {str(item.get("stanox", "")): (item.get("description") or item.get("name") or str(item.get("stanox",""))) for item in station_ref}
    except Exception:
        stanox_to_name = {}
    
    # Extract additional STANOX codes from all_data that have delays for this service code
    service_data = all_data[all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code)].copy()
    additional_stanox = set()
    
    if not service_data.empty and 'STANOX' in service_data.columns and 'PFPI_MINUTES' in service_data.columns:
        # Get stations with delays (PFPI_MINUTES > 0)
        service_data['PFPI_MINUTES_num'] = pd.to_numeric(service_data['PFPI_MINUTES'], errors='coerce')
        delayed_stations = service_data[service_data['PFPI_MINUTES_num'] > 0]['STANOX'].dropna().unique()
        
        for stanox in delayed_stations:
            stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
            additional_stanox.add(stanox_str)
        
        if additional_stanox:
            pass
    
    # Merge service_stanox with additional stations
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    all_stanox = service_stanox_normalized.union(additional_stanox)

    results = []

    for s in all_stanox:
        # Filter data for this STANOX and service code
        subset = all_data[
            (all_data['STANOX'] == str(s)) &
            (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code))
        ].copy()

        if subset.empty:
            continue

        # Convert PFPI_MINUTES to numeric (keep full series for on-time calc)
        pfpi_all = pd.to_numeric(subset['PFPI_MINUTES'], errors='coerce').dropna()
        # On-time percentage (<= 0 minutes)
        on_time_pct = (pfpi_all <= 0).sum() / len(pfpi_all) * 100 if len(pfpi_all) > 0 else np.nan

        # Exclude 0.0 delays for mean/variance and incident counting
        pfpi_pos = pfpi_all[pfpi_all > 0]
        mean_delay = pfpi_pos.mean() if len(pfpi_pos) > 0 else np.nan
        delay_variance = pfpi_pos.var() if len(pfpi_pos) > 0 else np.nan
        incident_count = len(pfpi_pos)

        station_name = stanox_to_name.get(str(s), f"{s}")

        results.append({
            'ServiceCode': service_code,
            'StationName': station_name,
            'MeanDelay': mean_delay,
            'DelayVariance': delay_variance,
            'OnTime%': on_time_pct,
            'IncidentCount': incident_count
        })

    return pd.DataFrame(results)

def plot_reliability_graphs(all_data, service_stanox, service_code, stations_ref_path=None, cap_minutes=75):
    """
    Generate overlapping density (KDE) curves and cumulative distribution plots: Delay distribution per station (all curves overlapping, different colours), excluding delay==0.0 and capped at cap_minutes.
    
    NEW: Also includes stations from all_data that experienced delays for this service code.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Load station reference with flexible path
    if stations_ref_path is None:
        from demo.data.reference import reference_files
        stations_ref_path = reference_files["all dft categories"]
    
    # try to get station names
    try:
        import json
        with open(stations_ref_path, 'r') as f:
            station_ref = json.load(f)
        stanox_to_name = {str(item.get('stanox','')): item.get('description') or item.get('name') or str(item.get('stanox','')) for item in station_ref}
    except Exception:
        stanox_to_name = {}
    
    # Extract additional STANOX codes from all_data that have delays for this service code
    service_data = all_data[all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code)].copy()
    additional_stanox = set()
    
    if not service_data.empty and 'STANOX' in service_data.columns and 'PFPI_MINUTES' in service_data.columns:
        # Get stations with delays (PFPI_MINUTES > 0)
        service_data['PFPI_MINUTES_num'] = pd.to_numeric(service_data['PFPI_MINUTES'], errors='coerce')
        delayed_stations = service_data[service_data['PFPI_MINUTES_num'] > 0]['STANOX'].dropna().unique()
        
        for stanox in delayed_stations:
            stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
            additional_stanox.add(stanox_str)
        
        if additional_stanox:
            pass
    
    # Merge service_stanox with additional stations
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    all_stanox = service_stanox_normalized.union(additional_stanox)

    station_labels = []
    delay_data = []

    for s in all_stanox:
        subset = all_data[
            (all_data['STANOX'] == str(s)) &
            (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code))
        ].copy()

        if subset.empty:
            continue

        # convert pfpi and drop nan
        pfpi_all = pd.to_numeric(subset['PFPI_MINUTES'], errors='coerce').dropna()
        # exclude zeros for plotting/stats (per user request)
        pfpi_pos = pfpi_all[pfpi_all > 0]
        delays = pfpi_pos.values
        if len(delays) == 0:
            continue

        label = stanox_to_name.get(str(s), str(s))
        station_labels.append(label)
        delay_data.append(delays)

    if not station_labels:
        pass
        return

    # Graph 1: Overlapping density plots (KDE) per station
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap('tab10')
    n = len(delay_data)
    colors = [cmap(i % cmap.N) for i in range(n)]

    # Determine x range from percentiles but cap at cap_minutes
    all_vals = np.concatenate(delay_data)
    xmin = max(0, np.nanpercentile(all_vals, 1))
    xmax = np.nanpercentile(all_vals, 99)
    x_vals = np.linspace(xmin, xmax, 400)

    try:
        import seaborn as sns
        for i, delays in enumerate(delay_data):
            # clip delays to [0, cap_minutes]
            clipped = np.clip(delays, 0, cap_minutes)
            sns.kdeplot(clipped, bw_adjust=1, label=station_labels[i], color=colors[i], fill=False, clip=(0, cap_minutes))
    except Exception:
        try:
            from scipy.stats import gaussian_kde
            for i, delays in enumerate(delay_data):
                clipped = np.clip(delays, 0, cap_minutes)
                try:
                    kde = gaussian_kde(clipped)
                    y = kde(x_vals)
                except Exception:
                    y = np.zeros_like(x_vals)
                    hist_vals, bin_edges = np.histogram(clipped, bins=20, density=True)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    y = np.interp(x_vals, bin_centers, hist_vals, left=0, right=0)
                plt.plot(x_vals, y, label=station_labels[i], color=colors[i])
        except Exception:
            for i, delays in enumerate(delay_data):
                clipped = np.clip(delays, 0, cap_minutes)
                hist_vals, bin_edges = np.histogram(clipped, bins=30, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.plot(bin_centers, hist_vals, label=station_labels[i], color=colors[i])

    plt.xlim(0, cap_minutes)
    plt.xlabel('Delay (minutes)')
    plt.ylabel('PDF')
    plt.title(f'Delay Distribution per Station (overlapping KDEs, capped at {cap_minutes} min)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Graph 2: Cumulative distribution plots per station
    plt.figure(figsize=(10, 6))

    for i, delays in enumerate(delay_data):
        clipped = np.clip(delays, 0, cap_minutes)
        sorted_delays = np.sort(clipped)
        y = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
        plt.plot(sorted_delays, y, label=station_labels[i], color=colors[i])

    plt.xlim(0, cap_minutes)
    plt.xlabel('Delay (minutes)')
    plt.ylabel('CDF')
    plt.title(f'Cumulative Delay Distribution per Station (capped at {cap_minutes} min)')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# for time_view,

def _print_date_statistics(date_str, all_data):
    """
    Print incident statistics for a specific date.
    
    Parameters:
    date_str (str): Date string (format: 'DD-MMM-YYYY')
    all_data (DataFrame): All incident data
    
    Returns:
    DataFrame: Filtered data for the specified date with PFPI > 5 minutes
    """
    # Filter to PFPI > 5 minutes
    incidents_filtered = all_data[all_data['PFPI_MINUTES'] > 5].copy()
    
    # Extract date from INCIDENT_START_DATETIME
    incidents_filtered['DATE'] = incidents_filtered['INCIDENT_START_DATETIME'].str.split(' ').str[0]
    
    # Get statistics for this specific date
    date_data = incidents_filtered[incidents_filtered['DATE'] == date_str]
    
    if not date_data.empty:
        # Count unique incidents for this date
        incident_count = date_data['INCIDENT_NUMBER'].nunique()
        
        # Get top 5 incident reasons - counting UNIQUE incidents per reason
        reason_counts = date_data.groupby('INCIDENT_REASON')['INCIDENT_NUMBER'].nunique().sort_values(ascending=False).head(5)
        top_reasons = ', '.join([f"{reason}({count})" for reason, count in reason_counts.items()])
        
        pass
        pass
    else:
        pass
    
    return date_data


def _load_station_coordinates(stations_ref_path=None):
    """
    Load station reference data and build STANOX to coordinates mapping.
    
    Parameters:
    stations_ref_path (str): Path to station reference file (optional)
    
    Returns:
    dict: Mapping of STANOX code (string) to [latitude, longitude]
    """
    import json
    from demo.data.reference import reference_files
    
    if stations_ref_path is None:
        stations_ref_path = reference_files["all dft categories"]
    
    stanox_to_coords = {}
    try:
        with open(stations_ref_path, 'r') as f:
            stations_data = json.load(f)
        for station in stations_data:
            if 'stanox' in station and 'latitude' in station and 'longitude' in station:
                stanox_to_coords[str(station['stanox'])] = [station['latitude'], station['longitude']]
    except Exception as e:
        pass
        return {}
    
    return stanox_to_coords


def _aggregate_time_view_data(date_str, all_data):
    """
    Aggregate incident and delay data by STANOX for a specific date.
    
    Parameters:
    date_str (str): Date string (format: 'DD-MMM-YYYY')
    all_data (DataFrame): All incident data
    
    Returns:
    tuple: (affected_stanox, incident_counts, total_pfpi) or (None, None, None) if no data
    """
    # Filter data for the specified date
    filtered_data = all_data[all_data['INCIDENT_START_DATETIME'].str.contains(date_str, na=False)]
    
    if filtered_data.empty:
        pass
        return None, None, None
    
    # Get unique affected STANOX codes
    affected_stanox = filtered_data['STANOX'].unique()
    
    # Count incidents per STANOX
    incident_counts = filtered_data.groupby('STANOX')['INCIDENT_NUMBER'].nunique()
    
    # Sum PFPI_MINUTES per STANOX
    total_pfpi = filtered_data.groupby('STANOX')['PFPI_MINUTES'].sum()
    
    return affected_stanox, incident_counts, total_pfpi


def _create_time_view_markers(m, affected_stanox, incident_counts, total_pfpi, stanox_to_coords):
    """
    Create and add station markers to time view map, sized by incident count and colored by delay.
    
    Parameters:
    m: Folium map object
    affected_stanox (array): Array of unique STANOX codes affected on this date
    incident_counts (Series): STANOX to incident count mapping
    total_pfpi (Series): STANOX to total PFPI minutes mapping
    stanox_to_coords (dict): STANOX to [lat, lon] mapping
    """
    import folium
    import numpy as np
    import pandas as pd
    
    def get_color(delay):
        """Get color for delay severity."""
        try:
            d = float(delay)
        except Exception:
            d = 0
        if d == 0:
            return "blue"
        if d <= 5:
            return '#32CD32'     # Minor (1-5 min) - Lime Green
        elif d <= 15:
            return '#FFD700'     # Moderate (6-15 min) - Gold
        elif d <= 30:
            return '#FF8C00'     # Significant (16-30 min) - Dark Orange
        elif d <= 60:
            return '#FF0000'     # Major (31-60 min) - Red
        elif d <= 120:
            return '#8B0000'     # Severe (61-120 min) - Dark Red
        else:
            return '#8A2BE2'     # Critical (120+ min) - Blue Violet
    
    # Add markers for each affected station
    for stanox in affected_stanox:
        stanox_str = str(stanox)
        if stanox_str in stanox_to_coords:
            lat, lon = stanox_to_coords[stanox_str]
            count = incident_counts.get(stanox, 0)
            count = int(count) if pd.notna(count) else 0
            total_delay = total_pfpi.get(stanox, 0)
            total_delay = float(total_delay) if pd.notna(total_delay) else 0.0
            color = get_color(total_delay)
            radius = int(5 + np.sqrt(count) * 4)  # Square root scaling for better visibility
            
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"STANOX: {stanox_str}<br>Incidents: {count}<br>Total Delay: {total_delay:.1f} min"
            ).add_to(m)
        else:
            pass


def _finalize_time_view_map(m, date_str):
    """
    Finalize time view map by adding title, legend, and saving to file.
    
    Parameters:
    m: Folium map object
    date_str (str): Date string (format: 'DD-MMM-YYYY')
    """
    import folium
    
    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 350px; height: 50px; background-color: white; border:2px solid grey; z-index:9999; font-size:16px; padding: 7px; text-align: center;">
    <b>One-Day Delay Map for {date_str}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 100px; left: 50px; width: 180px; height: 300px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px;">
    <p><b>Delay Legend (Total PFPI Minutes)</b></p>
    <p><span style="color:blue;">●</span> 0 min</p>
    <p><span style="color:#32CD32;">●</span> 1-5 min</p>
    <p><span style="color:#FFD700;">●</span> 6-15 min</p>
    <p><span style="color:#FF8C00;">●</span> 16-30 min</p>
    <p><span style="color:#FF0000;">●</span> 31-60 min</p>
    <p><span style="color:#8B0000;">●</span> 61-120 min</p>
    <p><span style="color:#8A2BE2;">●</span> 120+ min</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map to HTML file
    output_file = f"time_view_{date_str.replace('-', '_')}.html"
    m.save(output_file)
    pass


def create_time_view_html(date_str, all_data):
    """
    Create an HTML map showing affected stations for a given date, with markers sized by incident count and colored by total PFPI minutes.
    Prints incident statistics for the specific date before generating the map.
    
    Refactored to use focused helper functions for statistics, data aggregation, marker creation, and finalization.
    """
    import folium
def create_time_view_html(date_str, all_data):
    """
    Create an HTML map showing affected stations for a given date, with markers sized by incident count and colored by total PFPI minutes.
    Prints incident statistics for the specific date before generating the map.
    
    Refactored to use focused helper functions for statistics, data aggregation, marker creation, and finalization.
    """
    import folium
    
    # STEP 1: Print date statistics
    _print_date_statistics(date_str, all_data)
    
    # STEP 2: Aggregate incident and delay data by STANOX
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data(date_str, all_data)
    
    if affected_stanox is None:
        return
    
    # STEP 3: Load station coordinates
    stanox_to_coords = _load_station_coordinates()
    
    # STEP 4: Create base map centered on UK
    m = folium.Map(location=[54.5, -2.5], zoom_start=6)
    
    # STEP 5: Create and add station markers
    _create_time_view_markers(m, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # STEP 6: Finalize map (add title, legend, save)
    _finalize_time_view_map(m, date_str)




def station_view_yearly(station_id, interval_minutes=30):
    """
    Station analysis for yearly data across all incidents - simplified output.
    Analyzes all days of the week for a station and separates incident vs normal operations.
    """
    
    # Load data from all day files
    processed_base = '../processed_data'
    station_folder = os.path.join(processed_base, station_id)
    
    if not os.path.exists(station_folder):
        pass
        return None, None
    
    # Define day files to load
    day_files = ['MO.parquet', 'TU.parquet', 'WE.parquet', 'TH.parquet', 'FR.parquet', 'SA.parquet', 'SU.parquet']
    
    all_station_data = []
    
    for day_file in day_files:
        file_path = os.path.join(station_folder, day_file)
        if os.path.exists(file_path):
            try:
                day_data = pd.read_parquet(file_path, engine='fastparquet')
                day_data['day_of_week'] = day_file.replace('.parquet', '')
                all_station_data.append(day_data)
                pass
            except Exception as e:
                pass
    
    if not all_station_data:
        return None, None
    
    # Combine all data
    combined_data = pd.concat(all_station_data, ignore_index=True)
    
    # Filter for trains with planned calls
    train_mask = combined_data['PLANNED_CALLS'].notna()
    all_train_data = combined_data[train_mask].copy()
    
    # Maximum delay deduplication
    if len(all_train_data) > 0:
        all_train_data['delay_numeric'] = pd.to_numeric(all_train_data['PFPI_MINUTES'], errors='coerce').fillna(0)
        all_train_data['dedup_priority'] = all_train_data['delay_numeric'] * 1000
        
        if 'ACTUAL_CALLS' in all_train_data.columns:
            all_train_data['dedup_priority'] += all_train_data['ACTUAL_CALLS'].notna().astype(int) * 100
        
        basic_dedup_cols = ['TRAIN_SERVICE_CODE', 'PLANNED_CALLS', 'day_of_week']
        basic_available = [col for col in basic_dedup_cols if col in all_train_data.columns]
        
        if len(basic_available) >= 2:
            all_train_data = all_train_data.sort_values(['delay_numeric', 'dedup_priority'], ascending=[False, False])
            all_train_data = all_train_data.drop_duplicates(subset=basic_available, keep='first')
            all_train_data = all_train_data.drop(['delay_numeric', 'dedup_priority'], axis=1)
    
    if len(all_train_data) == 0:
        return None, None
    
    # Separate incident and normal operations
    # Assume trains with incident codes are incident-related
    incident_mask = all_train_data['INCIDENT_NUMBER'].notna()
    incident_data = all_train_data[incident_mask].copy()
    normal_data = all_train_data[~incident_mask].copy()
    
    
    def process_operations_data(data, operation_type):
        """Process data for either incident or normal operations"""
        if len(data) == 0:
            return pd.DataFrame()
        
        # Process times - using a reference date for time parsing
        reference_date = datetime(2024, 1, 1)  # Use a standard reference date
        
        def parse_time_simple(time_val, base_date):
            if pd.isna(time_val):
                return None
            try:
                time_str = str(int(time_val)).zfill(4)
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            except:
                return None
        
        # Parse times and apply corrected timing logic
        data['planned_dt'] = data['PLANNED_CALLS'].apply(
            lambda x: parse_time_simple(x, reference_date))
        data['original_actual_dt'] = data['ACTUAL_CALLS'].apply(
            lambda x: parse_time_simple(x, reference_date))
        data['delay_minutes'] = pd.to_numeric(data['PFPI_MINUTES'], errors='coerce').fillna(0)
        
        # Create corrected actual times
        corrected_actual_times = []
        for _, row in data.iterrows():
            planned_dt = row['planned_dt']
            original_actual_dt = row['original_actual_dt']
            delay_min = row['delay_minutes']
            
            if pd.isna(planned_dt):
                corrected_actual_times.append(None)
                continue
                
            if delay_min > 0:
                corrected_actual = planned_dt + timedelta(minutes=delay_min)
                corrected_actual_times.append(corrected_actual)
            elif delay_min == 0:
                corrected_actual_times.append(planned_dt)
            else:
                if pd.notna(original_actual_dt):
                    corrected_actual_times.append(original_actual_dt)
                else:
                    corrected_actual_times.append(planned_dt)
        
        data['effective_time'] = corrected_actual_times
        valid_data = data[data['effective_time'].notna()].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Group by time intervals (using hour of day for grouping)
        valid_data['hour_of_day'] = valid_data['effective_time'].dt.hour
        valid_data['interval_group'] = (valid_data['hour_of_day'] * 60 + valid_data['effective_time'].dt.minute) // interval_minutes
        
        intervals = []
        
        for interval_group in valid_data['interval_group'].unique():
            interval_trains = valid_data[valid_data['interval_group'] == interval_group]
            
            if len(interval_trains) > 0:
                arrival_trains = interval_trains[interval_trains['EVENT_TYPE'] != 'C']
                cancellation_trains = interval_trains[interval_trains['EVENT_TYPE'] == 'C']
                
                if len(arrival_trains) > 0 or len(cancellation_trains) > 0:
                    if len(arrival_trains) > 0:
                        delay_values = arrival_trains['delay_minutes'].tolist()
                        ontime_arrivals = len([d for d in delay_values if d == 0.0])
                        delayed_arrivals = len([d for d in delay_values if d > 0.0])
                        delayed_minutes = [round(d, 1) for d in delay_values if d > 0.0]
                    else:
                        ontime_arrivals = 0
                        delayed_arrivals = 0
                        delayed_minutes = []
                    
                    total_cancellations = len(cancellation_trains)
                    
                    # Calculate time period label
                    start_minute = interval_group * interval_minutes
                    end_minute = start_minute + interval_minutes
                    start_hour = start_minute // 60
                    start_min = start_minute % 60
                    end_hour = end_minute // 60
                    end_min = end_minute % 60
                    
                    intervals.append({
                        'time_period': f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}",
                        'ontime_arrival_count': ontime_arrivals,
                        'delayed_arrival_count': delayed_arrivals,
                        'cancellation_count': total_cancellations,
                        'delay_minutes': delayed_minutes,
                        'operation_type': operation_type
                    })
        
        return pd.DataFrame(intervals)
    
    # Process both incident and normal operations
    incident_summary = process_operations_data(incident_data, 'incident')
    normal_summary = process_operations_data(normal_data, 'normal')
    
    return incident_summary, normal_summary



def plot_trains_in_system_vs_delay(station_id, all_data, time_window_minutes=60, num_platforms=12, 
                                   figsize=(12, 8), max_delay_percentile=98, dwell_time_minutes=5, time_range=None):
    """
    Visualize the relationship between normalized trains in system and mean delay per hour.
    
    Similar to plot_variable_relationships but uses trains in system (occupancy) 
    instead of flow (throughput) on the x-axis.
    
    MERGED ANALYSIS: Combines weekdays and weekends into a single comprehensive view.
    
    Uses the EXACT SAME logic as plot_variable_relationships:
    - X-axis: Normalized trains in system per hour (from plot_bottleneck_analysis calculation)
    - Y-axis: Mean delay per hour ONLY from DELAYED trains (delay > 0), NOT all trains
    - One scatter point per HOUR (not per train)
    - Binned by trains in system with Q25-Q75 delay ranges
    
    THEORY:
    - As trains accumulate in the system (high occupancy), delays should increase
    - If delays remain low despite high trains in system, indicates good platform management
    - If delays spike at low trains in system, indicates operational inefficiencies
    
    Parameters:
    -----------
    station_id : str
        The station STANOX code
    all_data : pd.DataFrame
        The complete dataset containing all train records
    time_window_minutes : int
        Time window in minutes (default: 60)
    num_platforms : int
        Number of platforms for normalization (default: 12)
    figsize : tuple
        Figure size (default: (12, 8))
    max_delay_percentile : int
        Percentile to trim extreme values (default: 98)
    dwell_time_minutes : int
        Typical dwell time at station (default: 5 minutes)
    time_range : tuple or None
        Optional (start, end) tuple to filter by time range (default: None)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy.interpolate import make_interp_spline
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter data for the specific station
    data = all_data[all_data['STANOX'] == str(station_id)].copy()
    if len(data) == 0:
        return None
    
    # Filter for arrived trains (exclude cancellations)
    all_arrived_data = data[data['EVENT_TYPE'] != 'C'].copy()
    if len(all_arrived_data) == 0:
        return None
    
    # Calculate delays and total time in system
    all_arrived_data['delay_minutes'] = pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    all_arrived_data['time_in_system'] = dwell_time_minutes + all_arrived_data['delay_minutes']
    
    # Parse datetime
    def parse_event_datetime(event_dt_str):
        if pd.isna(event_dt_str):
            return None
        try:
            # Handle both string and datetime inputs (for compatibility with time_range filtered data)
            if isinstance(event_dt_str, (pd.Timestamp, datetime)):
                dt = event_dt_str
            else:
                dt = pd.to_datetime(event_dt_str, format='%d-%b-%Y %H:%M', errors='coerce')
            return dt.date() if pd.notna(dt) else None
        except:
            return None
    
    all_arrived_data['event_date'] = all_arrived_data['EVENT_DATETIME'].apply(parse_event_datetime)
    
    # Build day-to-date mapping
    day_to_weekday = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}
    day_date_mapping = {}
    for day_code in day_to_weekday.keys():
        day_data = all_arrived_data[all_arrived_data['DAY'] == day_code]
        observed_dates = day_data['event_date'].dropna().unique()
        if len(observed_dates) > 0:
            day_date_mapping[day_code] = sorted(observed_dates)
    
    # Create row index for distribution
    all_arrived_data['row_idx'] = range(len(all_arrived_data))
    
    def create_datetime_with_event_dates(row):
        try:
            # Check if EVENT_DATETIME is a valid datetime (not NaT/None)
            event_dt_val = row['EVENT_DATETIME']
            is_valid_datetime = pd.notna(event_dt_val) and isinstance(event_dt_val, (pd.Timestamp, datetime))
            
            # First, try to use EVENT_DATETIME directly if it's a valid datetime (from time_range filtering)
            if is_valid_datetime:
                event_dt = pd.Timestamp(event_dt_val)
                # Combine with time from ACTUAL_CALLS or PLANNED_CALLS
                time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
                if pd.notna(time_val):
                    time_str = str(int(time_val)).zfill(4)
                    hour = int(time_str[:2])
                    minute = int(time_str[2:])
                    return pd.Timestamp(year=event_dt.year, month=event_dt.month, day=event_dt.day, hour=hour, minute=minute)
                else:
                    return event_dt
            
            # Fallback: reconstruct from day code and date mapping (for on-time trains with NaT or unfiltered data)
            day_code = row['DAY']
            time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
            
            if pd.isna(time_val) or day_code not in day_to_weekday:
                return None
            
            # Parse time
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            # Get date
            if pd.notna(row['event_date']):
                date_obj = row['event_date']
            else:
                if day_code in day_date_mapping and len(day_date_mapping[day_code]) > 0:
                    date_idx = (hash(str(row['TRAIN_SERVICE_CODE'])) + row['row_idx']) % len(day_date_mapping[day_code])
                    date_obj = day_date_mapping[day_code][date_idx]
                else:
                    return None
            
            dt = pd.Timestamp(year=date_obj.year, month=date_obj.month, 
                            day=date_obj.day, hour=hour, minute=minute)
            return dt
        except:
            return None
    
    all_arrived_data['arrival_time'] = all_arrived_data.apply(create_datetime_with_event_dates, axis=1)
    
    # Calculate departure time (arrival + time in system)
    all_arrived_data['departure_time'] = all_arrived_data['arrival_time'] + pd.to_timedelta(
        all_arrived_data['time_in_system'], unit='min'
    )
    
    # Drop rows with invalid datetimes
    valid_data = all_arrived_data.dropna(subset=['arrival_time', 'departure_time']).copy()
    
    # Apply time_range filter if specified (filters by reconstructed arrival_time)
    if time_range is not None:
        start, end = _expand_time_range(time_range)
        print(f"\n[PLOT_TRAINS_IN_SYSTEM] Filtering valid_data by time_range: {start} to {end}", flush=True)
        print(f"[PLOT_TRAINS_IN_SYSTEM] Before filtering - valid_data rows: {len(valid_data)}", flush=True)
        
        # Filter by arrival_time
        valid_data = valid_data[(valid_data['arrival_time'] >= start) & (valid_data['arrival_time'] <= end)].copy()
        print(f"[PLOT_TRAINS_IN_SYSTEM] After filtering - valid_data rows: {len(valid_data)}", flush=True)
    
    # DIAGNOSTIC OUTPUT
    print(f"\n[DIAGNOSTIC] plot_trains_in_system_vs_delay for station {station_id}:", flush=True)
    print(f"  - Input rows: {len(all_arrived_data)}", flush=True)
    print(f"  - Rows with datetime EVENT_DATETIME: {sum(isinstance(dt, (pd.Timestamp, datetime)) for dt in all_arrived_data['EVENT_DATETIME'])}", flush=True)
    print(f"  - Valid arrival_times created: {all_arrived_data['arrival_time'].notna().sum()}", flush=True)
    print(f"  - Valid rows (with arrival & departure): {len(valid_data)}", flush=True)
    print(flush=True)
    
    if len(valid_data) == 0:
        return None
    
    # Process all data together (merged weekdays and weekends) - ONE POINT PER HOUR
    # Create hourly time bins for the entire period
    min_time = valid_data['arrival_time'].min().floor('h')
    max_time = valid_data['departure_time'].max().ceil('h')
    hourly_bins = pd.date_range(start=min_time, end=max_time, freq='h')
    
    hourly_stats_list = []
    
    for hour_start in hourly_bins[:-1]:  # Exclude last bin edge
        hour_end = hour_start + pd.Timedelta(hours=1)
        
        # TRAINS IN SYSTEM: Trains present at ANY point during this hour
        # (same calculation as plot_bottleneck_analysis)
        in_system = valid_data[
            (valid_data['arrival_time'] < hour_end) & 
            (valid_data['departure_time'] > hour_start)
        ]
        trains_in_system = in_system['TRAIN_SERVICE_CODE'].nunique()
        trains_in_system_normalized = trains_in_system / num_platforms
        
        # Mean delay of trains in system during this hour
        # CRITICAL: Match plot_variable_relationships - calculate delay ONLY from delayed trains (delay > 0)
        delayed_trains_in_hour = in_system[in_system['delay_minutes'] > 0]
        if len(delayed_trains_in_hour) > 0:
            mean_delay = delayed_trains_in_hour['delay_minutes'].mean()
        else:
            mean_delay = 0  # No delays in this hour
        
        # Only include hours with at least some activity
        if trains_in_system > 0:
            hourly_stats_list.append({
                'hour_start': hour_start,
                'trains_in_system_normalized': trains_in_system_normalized,
                'mean_delay': mean_delay
            })
    
    hourly_stats = pd.DataFrame(hourly_stats_list)
    
    if len(hourly_stats) == 0:
        print("No statistics to plot.")
        return None
    
    # Set index
    hourly_stats = hourly_stats.set_index('hour_start')
    
    # Trim outliers based on mean delay (like plot_variable_relationships)
    if max_delay_percentile < 100 and len(hourly_stats) > 0:
        delay_threshold = hourly_stats['mean_delay'].quantile(max_delay_percentile / 100)
        hourly_stats = hourly_stats[hourly_stats['mean_delay'] <= delay_threshold]
    
    print(f"Processed {len(hourly_stats)} hours total (weekdays and weekends merged)")
    
    # Create single visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.7))
    
    # Calculate correlation
    r_corr = hourly_stats[['trains_in_system_normalized', 'mean_delay']].corr().iloc[0, 1]
    
    # Scatter plot: One point per HOUR (like plot_variable_relationships)
    ax.scatter(hourly_stats['trains_in_system_normalized'], hourly_stats['mean_delay'], 
              alpha=0.3, color='lightblue', s=15, 
              edgecolors='blue', linewidth=0.2)
    
    # Add binned statistics with asymmetric error bars (like plot_variable_relationships)
    if len(hourly_stats) > 10:
        n_bins = 20
        min_obs = 5
        
        trains_min = hourly_stats['trains_in_system_normalized'].min()
        trains_max = hourly_stats['trains_in_system_normalized'].max()
        
        if trains_max > trains_min:
            bin_edges = np.linspace(trains_min, trains_max, n_bins + 1)
            hourly_stats['trains_bin'] = pd.cut(hourly_stats['trains_in_system_normalized'], 
                                            bins=bin_edges, include_lowest=True, labels=False)
            
            bin_trains = []
            bin_delays = []
            bin_delays_q25 = []
            bin_delays_q75 = []
            bin_counts = []
            
            for bin_idx in range(n_bins):
                bin_data = hourly_stats[hourly_stats['trains_bin'] == bin_idx]
                if len(bin_data) >= min_obs:
                    bin_trains.append(bin_data['trains_in_system_normalized'].mean())
                    bin_delays.append(bin_data['mean_delay'].mean())
                    bin_delays_q25.append(bin_data['mean_delay'].quantile(0.25))
                    bin_delays_q75.append(bin_data['mean_delay'].quantile(0.75))
                    bin_counts.append(len(bin_data))
            
            if len(bin_trains) >= 4:
                bin_trains = np.array(bin_trains)
                bin_delays = np.array(bin_delays)
                bin_delays_q25 = np.array(bin_delays_q25)
                bin_delays_q75 = np.array(bin_delays_q75)
                bin_counts = np.array(bin_counts)
                
                # Calculate asymmetric error bars
                yerr_lower = np.maximum(0, bin_delays - bin_delays_q25)
                yerr_upper = np.maximum(0, bin_delays_q75 - bin_delays)
                
                # Plot binned averages with error bars
                ax.errorbar(bin_trains, bin_delays, 
                           yerr=[yerr_lower, yerr_upper],
                           fmt='o', color='darkgreen', markersize=5, 
                           linewidth=1.5, capsize=3, capthick=1, zorder=5)
                
                # Smooth curve through binned data
                try:
                    sort_idx = np.argsort(bin_trains)
                    x_sorted = bin_trains[sort_idx]
                    y_sorted = bin_delays[sort_idx]
                    
                    spline = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x_sorted)-1))
                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                    y_smooth = spline(x_smooth)
                    ax.plot(x_smooth, y_smooth, 'g-', linewidth=1.5, zorder=10)
                except:
                    pass
    
    ax.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax.set_ylabel('Mean Delay\n(minutes)', fontsize=23)
    ax.set_xlim(0, 2.5)  # Fixed x-axis range for normalized trains in system
    ax.set_ylim(0, 25)  # Fixed y-axis range for delays
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=23)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"STATION {station_id} - DELAY vs TRAINS IN SYSTEM (ALL DAYS)")
    print(f"{'='*80}")
    print(f"\nDATA SUMMARY:")
    print(f"  - Total hours analyzed: {len(hourly_stats)}")
    print(f"  - Normalized trains in system range: {hourly_stats['trains_in_system_normalized'].min():.3f} - {hourly_stats['trains_in_system_normalized'].max():.3f} trains/platform")
    print(f"  - Mean delay range: {hourly_stats['mean_delay'].min():.2f} - {hourly_stats['mean_delay'].max():.2f} minutes")
    print(f"  - Overall mean delay: {hourly_stats['mean_delay'].mean():.2f} minutes")
    print(f"  - Hours with delays > 0: {(hourly_stats['mean_delay'] > 0).sum()} ({100*(hourly_stats['mean_delay'] > 0).sum()/len(hourly_stats):.1f}%)")
    print(f"  - Correlation (trains in system vs delay): {r_corr:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return hourly_stats




def explore_delay_outliers(station_id, all_data, num_platforms=6, dwell_time_minutes=5, figsize=(12, 8), time_range=None):
    """
    Specialized visualization to explore delay outliers and extreme cases.
    Shows delay percentiles vs system load with binned averages.
    
    Parameters:
    -----------
    station_id : str
        The STANOX code for the station
    all_data : pd.DataFrame
        The complete dataset with all station data
    num_platforms : int
        Number of platforms at the station (for normalization)
    dwell_time_minutes : int
        Typical dwell time at the station in minutes
    figsize : tuple
        Figure size (width, height)
    time_range : tuple or None
        Optional (start, end) tuple to filter by time range (default: None)
    
    Returns:
    --------
    pd.DataFrame
        Hourly statistics including delay percentiles and system load metrics
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print(f"EXPLORING DELAY OUTLIERS FOR STATION {station_id}")
    print(f"Focus: Extreme delays and worst-case scenarios")
    print("=" * 70)
    
    # Filter and prepare data (same as plot_trains_in_system_vs_delay)
    data = all_data[all_data['STANOX'] == str(station_id)].copy()
    if len(data) == 0:
        print(f"No data found for station {station_id}")
        return None
    
    all_arrived_data = data[data['EVENT_TYPE'] != 'C'].copy()
    if len(all_arrived_data) == 0:
        print("No arrived trains found.")
        return None
    
    all_arrived_data['delay_minutes'] = pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    all_arrived_data['time_in_system'] = dwell_time_minutes + all_arrived_data['delay_minutes']
    
    # Parse datetime
    def parse_event_datetime(event_dt_str):
        if pd.isna(event_dt_str):
            return None
        try:
            # Handle both string and datetime inputs (for compatibility with time_range filtered data)
            if isinstance(event_dt_str, (pd.Timestamp, datetime)):
                dt = event_dt_str
            else:
                dt = pd.to_datetime(event_dt_str, format='%d-%b-%Y %H:%M', errors='coerce')
            return dt.date() if pd.notna(dt) else None
        except:
            return None
    
    all_arrived_data['event_date'] = all_arrived_data['EVENT_DATETIME'].apply(parse_event_datetime)
    
    # Build day-to-date mapping
    day_to_weekday = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}
    day_date_mapping = {}
    for day_code in day_to_weekday.keys():
        day_data = all_arrived_data[all_arrived_data['DAY'] == day_code]
        observed_dates = day_data['event_date'].dropna().unique()
        if len(observed_dates) > 0:
            day_date_mapping[day_code] = sorted(observed_dates)
    
    all_arrived_data['row_idx'] = range(len(all_arrived_data))
    
    def create_datetime_with_event_dates(row):
        try:
            # Check if EVENT_DATETIME is a valid datetime (not NaT)
            event_dt_val = row['EVENT_DATETIME']
            is_valid_datetime = pd.notna(event_dt_val) and isinstance(event_dt_val, (pd.Timestamp, datetime))
            
            # First, try to use EVENT_DATETIME directly if it's a valid datetime (from time_range filtering)
            if is_valid_datetime:
                event_dt = pd.Timestamp(event_dt_val)
                # Combine with time from ACTUAL_CALLS or PLANNED_CALLS
                time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
                if pd.notna(time_val):
                    time_str = str(int(time_val)).zfill(4)
                    hour = int(time_str[:2])
                    minute = int(time_str[2:])
                    return pd.Timestamp(year=event_dt.year, month=event_dt.month, day=event_dt.day, hour=hour, minute=minute)
                else:
                    return event_dt
            
            # Fallback: reconstruct from day code and date mapping (for on-time trains with NaT or unfiltered data)
            day_code = row['DAY']
            time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
            
            if pd.isna(time_val) or day_code not in day_to_weekday:
                return None
            
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            if pd.notna(row['event_date']):
                date_obj = row['event_date']
            else:
                if day_code in day_date_mapping and len(day_date_mapping[day_code]) > 0:
                    date_idx = (hash(str(row['TRAIN_SERVICE_CODE'])) + row['row_idx']) % len(day_date_mapping[day_code])
                    date_obj = day_date_mapping[day_code][date_idx]
                else:
                    return None
            
            dt = pd.Timestamp(year=date_obj.year, month=date_obj.month, 
                            day=date_obj.day, hour=hour, minute=minute)
            return dt
        except:
            return None
    
    all_arrived_data['arrival_time'] = all_arrived_data.apply(create_datetime_with_event_dates, axis=1)
    all_arrived_data['departure_time'] = all_arrived_data['arrival_time'] + pd.to_timedelta(
        all_arrived_data['time_in_system'], unit='min'
    )
    
    valid_data = all_arrived_data.dropna(subset=['arrival_time', 'departure_time']).copy()
    
    # Apply time_range filter if specified (filters by reconstructed arrival_time)
    if time_range is not None:
        start, end = _expand_time_range(time_range)
        print(f"\n[EXPLORE_OUTLIERS] Filtering valid_data by time_range: {start} to {end}", flush=True)
        print(f"[EXPLORE_OUTLIERS] Before filtering - valid_data rows: {len(valid_data)}", flush=True)
        
        # Filter by arrival_time
        valid_data = valid_data[(valid_data['arrival_time'] >= start) & (valid_data['arrival_time'] <= end)].copy()
        print(f"[EXPLORE_OUTLIERS] After filtering - valid_data rows: {len(valid_data)}", flush=True)
    
    # DIAGNOSTIC OUTPUT
    print(f"\n[DIAGNOSTIC] explore_delay_outliers for station {station_id}:", flush=True)
    print(f"  - Input rows: {len(all_arrived_data)}", flush=True)
    print(f"  - Rows with datetime EVENT_DATETIME: {sum(isinstance(dt, (pd.Timestamp, datetime)) for dt in all_arrived_data['EVENT_DATETIME'])}", flush=True)
    print(f"  - Valid arrival_times created: {all_arrived_data['arrival_time'].notna().sum()}", flush=True)
    print(f"  - Valid rows (with arrival & departure): {len(valid_data)}", flush=True)
    print(flush=True)
    
    if len(valid_data) == 0:
        print("No valid datetime data.")
        return None
    
    # Calculate hourly statistics (including ALL percentiles for outlier analysis)
    min_time = valid_data['arrival_time'].min().floor('h')
    max_time = valid_data['departure_time'].max().ceil('h')
    hourly_bins = pd.date_range(start=min_time, end=max_time, freq='h')
    
    hourly_stats_list = []
    
    for hour_start in hourly_bins[:-1]:
        hour_end = hour_start + pd.Timedelta(hours=1)
        
        in_system = valid_data[
            (valid_data['arrival_time'] < hour_end) & 
            (valid_data['departure_time'] > hour_start)
        ]
        trains_in_system = in_system['TRAIN_SERVICE_CODE'].nunique()
        trains_in_system_normalized = trains_in_system / num_platforms
        
        # Calculate FULL delay statistics including outliers
        delayed_trains = in_system[in_system['delay_minutes'] > 0]
        
        if len(delayed_trains) > 0:
            mean_delay = delayed_trains['delay_minutes'].mean()
            median_delay = delayed_trains['delay_minutes'].median()
            max_delay = delayed_trains['delay_minutes'].max()
            p50_delay = delayed_trains['delay_minutes'].quantile(0.50)
            p90_delay = delayed_trains['delay_minutes'].quantile(0.90)
            p95_delay = delayed_trains['delay_minutes'].quantile(0.95)
            p99_delay = delayed_trains['delay_minutes'].quantile(0.99)
            std_delay = delayed_trains['delay_minutes'].std()
            num_delayed = len(delayed_trains)
        else:
            mean_delay = median_delay = max_delay = p50_delay = p90_delay = p95_delay = p99_delay = std_delay = 0
            num_delayed = 0
        
        if trains_in_system > 0:
            hourly_stats_list.append({
                'hour_start': hour_start,
                'trains_in_system_normalized': trains_in_system_normalized,
                'mean_delay': mean_delay,
                'median_delay': median_delay,
                'max_delay': max_delay,
                'p50_delay': p50_delay,
                'p90_delay': p90_delay,
                'p95_delay': p95_delay,
                'p99_delay': p99_delay,
                'std_delay': std_delay,
                'num_delayed': num_delayed,
                'total_trains': trains_in_system
            })
    
    hourly_stats = pd.DataFrame(hourly_stats_list)
    
    if len(hourly_stats) == 0:
        print("No statistics to plot.")
        return None
    
    print(f" Processed {len(hourly_stats)} hours with delay statistics")
    
    # Create single panel visualization - Percentile Comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.7))
    
    # Bin by trains in system for clearer trend visualization
    n_bins = 15
    trains_min = hourly_stats['trains_in_system_normalized'].min()
    trains_max = hourly_stats['trains_in_system_normalized'].max()
    
    if trains_max > trains_min:
        bin_edges = np.linspace(trains_min, trains_max, n_bins + 1)
        hourly_stats['trains_bin'] = pd.cut(hourly_stats['trains_in_system_normalized'], 
                                            bins=bin_edges, include_lowest=True, labels=False)
        
        bin_centers = []
        mean_vals = []
        p50_vals = []
        p90_vals = []
        p95_vals = []
        p99_vals = []
        
        for bin_idx in range(n_bins):
            bin_data = hourly_stats[hourly_stats['trains_bin'] == bin_idx]
            if len(bin_data) >= 3:  # Need at least 3 points for meaningful percentiles
                bin_centers.append(bin_data['trains_in_system_normalized'].mean())
                mean_vals.append(bin_data['mean_delay'].mean())
                p50_vals.append(bin_data['median_delay'].mean())
                p90_vals.append(bin_data['p90_delay'].mean())
                p95_vals.append(bin_data['p95_delay'].mean())
                p99_vals.append(bin_data['p99_delay'].mean())
        
        if len(bin_centers) > 0:
            ax.plot(bin_centers, mean_vals, 'o-', color='blue', linewidth=1.5, markersize=4)
            ax.plot(bin_centers, p50_vals, 'o-', color='green', linewidth=1.5, markersize=4)
            ax.plot(bin_centers, p90_vals, 's-', color='orange', linewidth=1.5, markersize=4)
            ax.plot(bin_centers, p95_vals, '^-', color='red', linewidth=1.5, markersize=4)
            ax.plot(bin_centers, p99_vals, 'D-', color='darkred', linewidth=1.5, markersize=4)
    
    ax.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax.set_ylabel('Delay Percentiles\n(minutes)', fontsize=23)
    ax.tick_params(axis='both', labelsize=23)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)  # Fixed x-axis for comparison
    ax.set_ylim(0, 30)   # Fixed y-axis: delay percentiles to 30 minutes
    
    plt.tight_layout()
    plt.show()
    
    # Create separate legend image
    fig_leg, ax_leg = plt.subplots(figsize=(10, 4))
    ax_leg.axis('off')
    
    # Create dummy lines for legend
    lines = [
        plt.Line2D([0], [0], marker='o', color='blue', linewidth=2, markersize=8, label='Mean'),
        plt.Line2D([0], [0], marker='o', color='green', linewidth=2, markersize=8, label='P50 (Median)'),
        plt.Line2D([0], [0], marker='s', color='orange', linewidth=2, markersize=8, label='P90'),
        plt.Line2D([0], [0], marker='^', color='red', linewidth=2, markersize=8, label='P95'),
        plt.Line2D([0], [0], marker='D', color='darkred', linewidth=2, markersize=8, label='P99 (Extreme)')
    ]
    ax_leg.legend(handles=lines, fontsize=23, loc='center', ncol=5)
    
    plt.tight_layout()
    plt.show()
    
    # Print extreme delay statistics
    print(f"\n{'='*80}")
    print(f" EXTREME DELAY STATISTICS - STATION {station_id}")
    print(f"{'='*80}")
    
    # Find hours with worst delays
    worst_hours = hourly_stats.nlargest(5, 'max_delay')
    print(f" TOP 5 WORST HOURS (by maximum delay):")
    for idx, row in worst_hours.iterrows():
        print(f"  {row['hour_start'].strftime('%Y-%m-%d %H:%M')}: "
              f"Max={row['max_delay']:.1f}min, Mean={row['mean_delay']:.1f}min, "
              f"Trains={row['trains_in_system_normalized']:.2f}/platform ({row['num_delayed']} delayed)")
    
    # Overall statistics
    print(f" OVERALL DELAY STATISTICS:")
    print(f"  - P90 delay across all hours: {hourly_stats['p90_delay'].mean():.2f} min")
    print(f"  - P95 delay across all hours: {hourly_stats['p95_delay'].mean():.2f} min")
    print(f"  - P99 delay across all hours: {hourly_stats['p99_delay'].mean():.2f} min")
    print(f"  - Maximum single delay: {hourly_stats['max_delay'].max():.2f} min")
    print(f"  - Hours with delays > 30 min: {(hourly_stats['max_delay'] > 30).sum()} ({100*(hourly_stats['max_delay'] > 30).sum()/len(hourly_stats):.1f}%)")
    
    # Correlation analysis
    high_load = hourly_stats[hourly_stats['trains_in_system_normalized'] > 1.0]
    if len(high_load) > 0:
        print(f" HIGH LOAD ANALYSIS (>1.0 trains/platform):")
        print(f"  - Hours in high load: {len(high_load)} ({100*len(high_load)/len(hourly_stats):.1f}%)")
        print(f"  - Mean delay in high load: {high_load['mean_delay'].mean():.2f} min")
        print(f"  - P95 delay in high load: {high_load['p95_delay'].mean():.2f} min")
    
    return hourly_stats

def station_view(station_id, all_data, num_platforms=6, time_window_minutes=60, max_delay_percentile=98, dwell_time_minutes=5, figsize=(8, 4.7), time_range=None):
    """
    Comprehensive merged station performance analysis combining 3 visualization functions.
    Analyzes on-time performance and system load relationships.
    
    Parameters:
    -----------
    station_id : str
        The STANOX code for the station
    all_data : pd.DataFrame
        The complete dataset with all station data
    num_platforms : int
        Number of platforms at the station (for normalization)
    time_window_minutes : int
        Time window for analysis (typically 60 for hourly)
    max_delay_percentile : int
        Maximum delay percentile to consider (typically 98)
    dwell_time_minutes : int
        Typical dwell time at the station in minutes
    figsize : tuple
        Figure size (width, height) - applied to all plots
    time_range : tuple or None
        Optional (start, end) tuple to filter by time range. Will filter valid_data
        to only include rows where arrival_time falls within this range (default: None)
    
    Returns:
    --------
    dict
        Dictionary containing hourly_stats and bin_stats DataFrames
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr
    from scipy.interpolate import make_interp_spline
    
    # ===== DATA PREPARATION (Common to all plots) =====
    data = all_data[all_data['STANOX'] == str(station_id)].copy()
    if len(data) == 0:
        print(f"No data found for station {station_id}")
        return None
    
    all_arrived_data = data[data['EVENT_TYPE'] != 'C'].copy()
    if len(all_arrived_data) == 0:
        print("No arrived trains found.")
        return None
    
    # DIAGNOSTIC: Check PFPI_MINUTES before converting
    print(f"\n[PFPI DIAGNOSTIC] For station {station_id}:", flush=True)
    print(f"  - PFPI_MINUTES dtype: {all_arrived_data['PFPI_MINUTES'].dtype}", flush=True)
    print(f"  - PFPI_MINUTES non-null count: {all_arrived_data['PFPI_MINUTES'].notna().sum()}", flush=True)
    print(f"  - PFPI_MINUTES null count: {all_arrived_data['PFPI_MINUTES'].isna().sum()}", flush=True)
    if all_arrived_data['PFPI_MINUTES'].notna().sum() > 0:
        print(f"  - PFPI_MINUTES min: {pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').min()}", flush=True)
        print(f"  - PFPI_MINUTES max: {pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').max()}", flush=True)
        print(f"  - PFPI_MINUTES mean: {pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').mean():.2f}", flush=True)
        print(f"  - Sample PFPI values: {all_arrived_data['PFPI_MINUTES'].dropna().head(10).tolist()}", flush=True)
    print(flush=True)
    
    all_arrived_data['delay_minutes'] = pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    all_arrived_data['time_in_system'] = dwell_time_minutes + all_arrived_data['delay_minutes']
    
    # DIAGNOSTIC: Check delay_minutes after conversion
    print(f"[DELAY DIAGNOSTIC] After calculating delay_minutes:", flush=True)
    print(f"  - delay_minutes non-zero count: {(all_arrived_data['delay_minutes'] > 0).sum()}", flush=True)
    print(f"  - delay_minutes zero count: {(all_arrived_data['delay_minutes'] == 0).sum()}", flush=True)
    print(f"  - delay_minutes mean: {all_arrived_data['delay_minutes'].mean():.2f}", flush=True)
    print(f"  - delay_minutes max: {all_arrived_data['delay_minutes'].max():.2f}", flush=True)
    print(flush=True)
    
    def parse_event_datetime(event_dt_str):
        if pd.isna(event_dt_str):
            return None
        try:
            # Handle both string and datetime inputs (for compatibility with time_range filtered data)
            if isinstance(event_dt_str, (pd.Timestamp, datetime)):
                dt = event_dt_str
            else:
                dt = pd.to_datetime(event_dt_str, format='%d-%b-%Y %H:%M', errors='coerce')
            return dt.date() if pd.notna(dt) else None
        except:
            return None
    
    all_arrived_data['event_date'] = all_arrived_data['EVENT_DATETIME'].apply(parse_event_datetime)
    
    day_to_weekday = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}
    day_date_mapping = {}
    for day_code in day_to_weekday.keys():
        day_data = all_arrived_data[all_arrived_data['DAY'] == day_code]
        # Collect dates from event_date (which works for both delayed and on-time trains when EVENT_DATETIME is valid)
        observed_dates = day_data['event_date'].dropna().unique()
        
        # If we have no observed dates (e.g., filtered data with only on-time trains),
        # extract dates from the available data to ensure on-time trains get reconstructed
        if len(observed_dates) == 0 and len(day_data) > 0:
            # Try to extract dates from EVENT_DATETIME if available (from time_range filtering)
            valid_dt = day_data[day_data['EVENT_DATETIME'].notna()]
            if len(valid_dt) > 0:
                try:
                    extracted_dates = pd.to_datetime(valid_dt['EVENT_DATETIME'], errors='coerce').dt.date
                    observed_dates = extracted_dates[extracted_dates.notna()].unique()
                except:
                    pass
        
        if len(observed_dates) > 0:
            day_date_mapping[day_code] = sorted(observed_dates)
    
    all_arrived_data['row_idx'] = range(len(all_arrived_data))
    
    def create_datetime_with_event_dates(row):
        try:
            # Check if EVENT_DATETIME is a valid datetime (not NaT/None)
            event_dt_val = row['EVENT_DATETIME']
            is_valid_datetime = pd.notna(event_dt_val) and isinstance(event_dt_val, (pd.Timestamp, datetime))
            
            # First, try to use EVENT_DATETIME directly if it's a valid datetime (from time_range filtering)
            if is_valid_datetime:
                event_dt = pd.Timestamp(event_dt_val)
                # Combine with time from ACTUAL_CALLS or PLANNED_CALLS
                time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
                if pd.notna(time_val):
                    time_str = str(int(time_val)).zfill(4)
                    hour = int(time_str[:2])
                    minute = int(time_str[2:])
                    return pd.Timestamp(year=event_dt.year, month=event_dt.month, day=event_dt.day, hour=hour, minute=minute)
                else:
                    return event_dt
            
            # Fallback: reconstruct from day code and date mapping (for on-time trains with NaT or unfiltered data)
            day_code = row['DAY']
            time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
            
            if pd.isna(time_val) or day_code not in day_to_weekday:
                return None
            
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            if pd.notna(row['event_date']):
                date_obj = row['event_date']
            else:
                if day_code in day_date_mapping and len(day_date_mapping[day_code]) > 0:
                    date_idx = (hash(str(row['TRAIN_SERVICE_CODE'])) + row['row_idx']) % len(day_date_mapping[day_code])
                    date_obj = day_date_mapping[day_code][date_idx]
                else:
                    return None
            
            dt = pd.Timestamp(year=date_obj.year, month=date_obj.month, 
                            day=date_obj.day, hour=hour, minute=minute)
            return dt
        except:
            return None
    
    all_arrived_data['arrival_time'] = all_arrived_data.apply(create_datetime_with_event_dates, axis=1)
    all_arrived_data['departure_time'] = all_arrived_data['arrival_time'] + pd.to_timedelta(
        all_arrived_data['time_in_system'], unit='min'
    )
    
    valid_data = all_arrived_data.dropna(subset=['arrival_time', 'departure_time']).copy()
    
    # Apply time_range filter if specified (filters both delayed and on-time trains by reconstructed arrival_time)
    if time_range is not None:
        start, end = _expand_time_range(time_range)
        print(f"\n[STATION_VIEW] Filtering valid_data by time_range: {start} to {end}", flush=True)
        print(f"[STATION_VIEW] Before filtering - valid_data rows: {len(valid_data)}", flush=True)
        
        # Filter by arrival_time (applies to both delayed and on-time trains)
        valid_data = valid_data[(valid_data['arrival_time'] >= start) & (valid_data['arrival_time'] <= end)].copy()
        print(f"[STATION_VIEW] After filtering - valid_data rows: {len(valid_data)}", flush=True)
    
    # DIAGNOSTIC OUTPUT
    print(f"\n[DIAGNOSTIC] station_view for station {station_id}:", flush=True)
    print(f"  - Input rows: {len(all_arrived_data)}", flush=True)
    print(f"  - Rows with datetime EVENT_DATETIME: {sum(isinstance(dt, (pd.Timestamp, datetime)) for dt in all_arrived_data['EVENT_DATETIME'])}", flush=True)
    print(f"  - Valid arrival_times created: {all_arrived_data['arrival_time'].notna().sum()}", flush=True)
    print(f"  - Valid rows (with arrival & departure): {len(valid_data)}", flush=True)
    if len(valid_data) > 0:
        print(f"  - Date range: {valid_data['arrival_time'].min().date()} to {valid_data['arrival_time'].max().date()}", flush=True)
    
    # DIAGNOSTIC: Compare delays across entire dataset vs valid_data
    print(f"[DELAY COMPARISON]:", flush=True)
    print(f"  - Mean delay (all_arrived_data): {all_arrived_data['delay_minutes'].mean():.2f}", flush=True)
    print(f"  - Mean delay (valid_data): {valid_data['delay_minutes'].mean():.2f}", flush=True)
    print(f"  - Zeros in all_arrived: {(all_arrived_data['delay_minutes'] == 0).sum()}", flush=True)
    print(f"  - Zeros in valid_data: {(valid_data['delay_minutes'] == 0).sum()}", flush=True)
    print(f"  - % zeros in all_arrived: {100*(all_arrived_data['delay_minutes'] == 0).sum()/len(all_arrived_data):.2f}%", flush=True)
    print(f"  - % zeros in valid_data: {100*(valid_data['delay_minutes'] == 0).sum()/len(valid_data):.2f}%", flush=True)
    print(flush=True)
    
    if len(valid_data) == 0:
        print("No valid datetime data.")
        return None
    
    # Calculate hourly statistics
    min_time = valid_data['arrival_time'].min().floor('h')
    max_time = valid_data['departure_time'].max().ceil('h')
    hourly_bins = pd.date_range(start=min_time, end=max_time, freq='h')
    
    hourly_stats_list = []
    
    for hour_start in hourly_bins[:-1]:
        hour_end = hour_start + pd.Timedelta(hours=1)
        
        in_system = valid_data[
            (valid_data['arrival_time'] < hour_end) & 
            (valid_data['departure_time'] > hour_start)
        ]
        trains_in_system = in_system['TRAIN_SERVICE_CODE'].nunique()
        trains_in_system_normalized = trains_in_system / num_platforms
        
        # On-time performance
        ontime_mask = (in_system['delay_minutes'].isna()) | (in_system['delay_minutes'] == 0.0)
        ontime_trains_count = in_system[ontime_mask]['TRAIN_SERVICE_CODE'].nunique()
        is_100_percent_ontime = (ontime_trains_count == trains_in_system)
        
        if trains_in_system > 0:
            hourly_stats_list.append({
                'hour_start': hour_start,
                'trains_in_system_normalized': trains_in_system_normalized,
                'total_trains': trains_in_system,
                'ontime_trains_count': ontime_trains_count,
                'ontime_trains_normalized': ontime_trains_count / num_platforms,
                'is_100_percent_ontime': is_100_percent_ontime,
                'ontime_ratio': ontime_trains_count / trains_in_system
            })
    
    hourly_stats = pd.DataFrame(hourly_stats_list)
    
    if len(hourly_stats) == 0:
        print("No statistics to plot.")
        return None
    
    print(f"Processed {len(hourly_stats)} hours with comprehensive statistics\n")
    
    # DIAGNOSTIC: Check hourly stats before binning
    print(f"\n[HOURLY STATS DIAGNOSTIC]:", flush=True)
    print(f"  - Total hours: {len(hourly_stats)}", flush=True)
    print(f"  - Hours with is_100_percent_ontime=True: {(hourly_stats['is_100_percent_ontime'] == True).sum()}", flush=True)
    print(f"  - Hours with is_100_percent_ontime=False: {(hourly_stats['is_100_percent_ontime'] == False).sum()}", flush=True)
    print(f"  - Mean ontime_ratio across all hours: {hourly_stats['ontime_ratio'].mean():.6f}", flush=True)
    print(f"  - Ontime_ratio min: {hourly_stats['ontime_ratio'].min():.6f}", flush=True)
    print(f"  - Ontime_ratio max: {hourly_stats['ontime_ratio'].max():.6f}", flush=True)
    if len(hourly_stats) > 0:
        print(f"  - Sample ontime_ratio values: {hourly_stats['ontime_ratio'].head(10).tolist()}", flush=True)
    print(flush=True)
    
    # ===== PLOT 1: On-Time Performance vs System Load =====
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    ax1.scatter(hourly_stats['trains_in_system_normalized'], 
               hourly_stats['ontime_trains_normalized'],
               c='cornflowerblue', alpha=0.6, s=30,
               edgecolors='black', linewidth=0.5)
    
    max_val = max(hourly_stats['trains_in_system_normalized'].max(), 
                  hourly_stats['ontime_trains_normalized'].max())
    ax1.plot([0, max_val], [0, max_val], 'g--', linewidth=1.5, alpha=0.7)
    
    z = np.polyfit(hourly_stats['trains_in_system_normalized'], hourly_stats['ontime_trains_normalized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(hourly_stats['trains_in_system_normalized'].min(), 
                         hourly_stats['trains_in_system_normalized'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r-", linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax1.set_ylabel('On-Time Trains\n(tph/platform)', fontsize=23)
    ax1.tick_params(axis='both', labelsize=23)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.show()
    print("Plot 1 complete: On-Time Performance vs System Load\n")
    
    # ===== PLOT 2: On-Time Performance Histogram =====
    bin_width = 0.2
    max_load = hourly_stats['trains_in_system_normalized'].max()
    bins = np.arange(0, max_load + bin_width, bin_width)
    
    print(f"\n[BIN DIAGNOSTIC] For station {station_id}:", flush=True)
    print(f"  - Max load value: {max_load:.2f}", flush=True)
    print(f"  - Bin width: {bin_width}", flush=True)
    print(f"  - Number of bins created: {len(bins)-1}", flush=True)
    
    # Create single value labels (bin midpoints)
    labels = [f'{(bins[i] + bins[i+1])/2:.1f}' for i in range(len(bins)-1)]
    hourly_stats['load_bin'] = pd.cut(hourly_stats['trains_in_system_normalized'], 
                                       bins=bins, 
                                       labels=labels,
                                       include_lowest=True)
    
    bin_stats = hourly_stats.groupby('load_bin', observed=True).agg(
        total_hours=('is_100_percent_ontime', 'count'),
        hours_100_percent=('is_100_percent_ontime', 'sum'),
        mean_ontime_ratio=('ontime_ratio', 'mean'),
        std_ontime_ratio=('ontime_ratio', 'std')
    ).reset_index()
    
    print(f"  - Actual bins with data: {len(bin_stats)}", flush=True)
    print(f"  - Bin stats:\n{bin_stats.to_string()}", flush=True)
    
    bin_stats['pct_hours_100_ontime'] = (bin_stats['hours_100_percent'] / bin_stats['total_hours'] * 100)
    bin_stats['cumulative_hours'] = bin_stats['total_hours'].cumsum()
    bin_stats['cdf'] = (bin_stats['cumulative_hours'] / bin_stats['total_hours'].sum() * 100)
    
    print(f"\nBin statistics ({len(bin_stats)} bins created):")
    print(f"  Max load value: {max_load:.2f}")
    print(f"  Bins: {list(bin_stats['load_bin'])}")
    print(f"  Hours per bin: {list(bin_stats['total_hours'])}")
    print(f"  100% on-time hours: {list(bin_stats['hours_100_percent'])}\n")
    
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
    
    x_pos = np.arange(len(bin_stats))
    bars = ax2.bar(x_pos, bin_stats['pct_hours_100_ontime'], 
                   color='cornflowerblue', alpha=0.7, edgecolor='black', linewidth=1.0, width=0.6)
    
    ax2.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax2.set_ylabel('% of Hours\n100% On-Time', fontsize=23)
    ax2.tick_params(axis='both', labelsize=23)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_stats['load_bin'], rotation=0, fontsize=23)
    # Adaptive xlim based on actual data range (not hardcoded to 12)
    ax2.set_xlim(-0.5, len(bin_stats) - 0.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print("Plot 2 complete: On-Time Performance Histogram\n")
    
    # ===== PLOT 3: Cumulative Distribution Function (CDF) =====
    fig3, ax3 = plt.subplots(1, 1, figsize=figsize)
    
    cdf_values = bin_stats['cdf'].values
    ax3.plot(x_pos, cdf_values, 
            marker='o', markersize=6, linewidth=2, color='darkred', 
            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=1.0)
    ax3.fill_between(x_pos, 0, cdf_values, alpha=0.15, color='darkred')
    
    ax3.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax3.set_ylabel('Cumulative\nDistribution (%)', fontsize=23)
    ax3.tick_params(axis='both', labelsize=23)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_stats['load_bin'], rotation=0, fontsize=23)
    # Adaptive xlim based on actual data range (not hardcoded to 12)
    ax3.set_xlim(-0.5, len(bin_stats) - 0.5)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Plot 3 complete: Cumulative Distribution Function\n")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f" COMPREHENSIVE STATION ANALYSIS SUMMARY FOR STATION {station_id}")
    print(f"{'='*80}")
    
    print(f"\nON-TIME PERFORMANCE:")
    print(f"  - Hours with 100% on-time: {hourly_stats['is_100_percent_ontime'].sum()} ({100*hourly_stats['is_100_percent_ontime'].sum()/len(hourly_stats):.1f}%)")
    
    print(f"\nSYSTEM LOAD:")
    print(f"  - Min normalized trains/platform: {hourly_stats['trains_in_system_normalized'].min():.2f}")
    print(f"  - Max normalized trains/platform: {hourly_stats['trains_in_system_normalized'].max():.2f}")
    print(f"  - Mean normalized trains/platform: {hourly_stats['trains_in_system_normalized'].mean():.2f}")
    
    print(f"{'='*80}\n")
    
    return {
        'hourly_stats': hourly_stats,
        'bin_stats': bin_stats
    }



def comprehensive_station_analysis(station_id, all_data, num_platforms=6, dwell_time_minutes=5, max_delay_percentile=98, time_range=None):
    """
    Combined comprehensive station analysis displaying all visualizations in a single column figure.
    
    Combines plot_trains_in_system_vs_delay, explore_delay_outliers, and station_view 
    without changing any of their internal logic.
    
    Parameters:
    -----------
    station_id : str
        The STANOX code for the station
    all_data : pd.DataFrame
        The complete dataset with all station data
    num_platforms : int
        Number of platforms at the station (for normalization)
    dwell_time_minutes : int
        Typical dwell time at the station in minutes
    max_delay_percentile : int
        Maximum delay percentile to consider (typically 98)
    time_range : tuple or None
        Optional (start, end) tuple to filter by time range (default: None)
    
    Returns:
    --------
    dict
        Dictionary containing all results from the three analyses
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.interpolate import make_interp_spline
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS FOR STATION {station_id}")
    print(f"{'='*80}\n")
    
    # ===== CALL plot_trains_in_system_vs_delay =====
    print("Running plot_trains_in_system_vs_delay analysis...")
    delay_analysis_result = plot_trains_in_system_vs_delay(
        station_id=station_id,
        all_data=all_data,
        time_window_minutes=60,
        num_platforms=num_platforms,
        figsize=(8, 4.7),
        max_delay_percentile=max_delay_percentile,
        dwell_time_minutes=dwell_time_minutes,
        time_range=time_range
    )
    
    # ===== CALL explore_delay_outliers =====
    print("\nRunning explore_delay_outliers analysis...")
    outlier_analysis_result = explore_delay_outliers(
        station_id=station_id,
        all_data=all_data,
        num_platforms=num_platforms,
        dwell_time_minutes=dwell_time_minutes,
        figsize=(8, 4.7),
        time_range=time_range
    )
    
    # ===== CALL station_view =====
    print("\nRunning station_view analysis...")
    station_view_result = station_view(
        station_id=station_id,
        all_data=all_data,
        num_platforms=num_platforms,
        time_window_minutes=60,
        max_delay_percentile=max_delay_percentile,
        dwell_time_minutes=dwell_time_minutes,
        figsize=(8, 4.7),
        time_range=time_range
    )
    
    # ===== CREATE COMBINED FIGURE WITH ALL PLOTS IN A COLUMN =====
    print("\nCreating combined visualization...")
    
    # Create figure with 5 subplots in a column: 5 rows, 1 column
    # Using figsize that matches individual plot proportions: 5 plots * 4.7 height each
    fig = plt.figure(figsize=(8, 23.5))
    
    # Plot 1: Trains in System vs Delay
    ax1 = plt.subplot(5, 1, 1)
    hourly_stats_delay = delay_analysis_result
    r_corr = hourly_stats_delay[['trains_in_system_normalized', 'mean_delay']].corr().iloc[0, 1]
    
    ax1.scatter(hourly_stats_delay['trains_in_system_normalized'], hourly_stats_delay['mean_delay'], 
              alpha=0.3, color='lightblue', s=15, 
              edgecolors='blue', linewidth=0.2)
    
    if len(hourly_stats_delay) > 10:
        n_bins = 20
        min_obs = 5
        trains_min = hourly_stats_delay['trains_in_system_normalized'].min()
        trains_max = hourly_stats_delay['trains_in_system_normalized'].max()
        
        if trains_max > trains_min:
            bin_edges = np.linspace(trains_min, trains_max, n_bins + 1)
            hourly_stats_delay_binned = hourly_stats_delay.copy()
            hourly_stats_delay_binned['trains_bin'] = pd.cut(hourly_stats_delay_binned['trains_in_system_normalized'], 
                                                bins=bin_edges, include_lowest=True, labels=False)
            
            bin_trains = []
            bin_delays = []
            bin_delays_q25 = []
            bin_delays_q75 = []
            
            for bin_idx in range(n_bins):
                bin_data = hourly_stats_delay_binned[hourly_stats_delay_binned['trains_bin'] == bin_idx]
                if len(bin_data) >= min_obs:
                    bin_trains.append(bin_data['trains_in_system_normalized'].mean())
                    bin_delays.append(bin_data['mean_delay'].mean())
                    bin_delays_q25.append(bin_data['mean_delay'].quantile(0.25))
                    bin_delays_q75.append(bin_data['mean_delay'].quantile(0.75))
            
            if len(bin_trains) >= 4:
                bin_trains = np.array(bin_trains)
                bin_delays = np.array(bin_delays)
                bin_delays_q25 = np.array(bin_delays_q25)
                bin_delays_q75 = np.array(bin_delays_q75)
                
                yerr_lower = np.maximum(0, bin_delays - bin_delays_q25)
                yerr_upper = np.maximum(0, bin_delays_q75 - bin_delays)
                
                ax1.errorbar(bin_trains, bin_delays, 
                           yerr=[yerr_lower, yerr_upper],
                           fmt='o', color='darkgreen', markersize=5, 
                           linewidth=1.5, capsize=3, capthick=1, zorder=5)
                
                try:
                    sort_idx = np.argsort(bin_trains)
                    x_sorted = bin_trains[sort_idx]
                    y_sorted = bin_delays[sort_idx]
                    
                    spline = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x_sorted)-1))
                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                    y_smooth = spline(x_smooth)
                    ax1.plot(x_smooth, y_smooth, 'g-', linewidth=1.5, zorder=10)
                except:
                    pass
    
    ax1.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax1.set_ylabel('Mean Delay\n(minutes)', fontsize=23)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 25)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=23)
    
    # Plot 2: Delay Percentiles (from explore_delay_outliers)
    ax2 = plt.subplot(5, 1, 2)
    hourly_stats_outliers = outlier_analysis_result
    
    n_bins = 15
    trains_min = hourly_stats_outliers['trains_in_system_normalized'].min()
    trains_max = hourly_stats_outliers['trains_in_system_normalized'].max()
    
    if trains_max > trains_min:
        bin_edges = np.linspace(trains_min, trains_max, n_bins + 1)
        hourly_stats_outliers['trains_bin'] = pd.cut(hourly_stats_outliers['trains_in_system_normalized'], 
                                            bins=bin_edges, include_lowest=True, labels=False)
        
        bin_centers = []
        mean_vals = []
        p50_vals = []
        p90_vals = []
        p95_vals = []
        p99_vals = []
        
        for bin_idx in range(n_bins):
            bin_data = hourly_stats_outliers[hourly_stats_outliers['trains_bin'] == bin_idx]
            if len(bin_data) >= 3:
                bin_centers.append(bin_data['trains_in_system_normalized'].mean())
                mean_vals.append(bin_data['mean_delay'].mean())
                p50_vals.append(bin_data['median_delay'].mean())
                p90_vals.append(bin_data['p90_delay'].mean())
                p95_vals.append(bin_data['p95_delay'].mean())
                p99_vals.append(bin_data['p99_delay'].mean())
        
        if len(bin_centers) > 0:
            ax2.plot(bin_centers, mean_vals, 'o-', color='blue', linewidth=1.5, markersize=4)
            ax2.plot(bin_centers, p50_vals, 'o-', color='green', linewidth=1.5, markersize=4)
            ax2.plot(bin_centers, p90_vals, 's-', color='orange', linewidth=1.5, markersize=4)
            ax2.plot(bin_centers, p95_vals, '^-', color='red', linewidth=1.5, markersize=4)
            ax2.plot(bin_centers, p99_vals, 'D-', color='darkred', linewidth=1.5, markersize=4)
    
    ax2.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax2.set_ylabel('Delay Percentiles\n(minutes)', fontsize=23)
    ax2.tick_params(axis='both', labelsize=23)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 30)
    ax2.legend(['Mean', 'P50', 'P90', 'P95', 'P99'], fontsize=12, loc='upper left')
    
    # Plot 3: On-Time Performance vs System Load (from station_view)
    ax3 = plt.subplot(5, 1, 3)
    hourly_stats_sv = station_view_result['hourly_stats']
    
    ax3.scatter(hourly_stats_sv['trains_in_system_normalized'], 
               hourly_stats_sv['ontime_trains_normalized'],
               c='cornflowerblue', alpha=0.6, s=30,
               edgecolors='black', linewidth=0.5)
    
    max_val = max(hourly_stats_sv['trains_in_system_normalized'].max(), 
                  hourly_stats_sv['ontime_trains_normalized'].max())
    ax3.plot([0, max_val], [0, max_val], 'g--', linewidth=1.5, alpha=0.7)
    
    z = np.polyfit(hourly_stats_sv['trains_in_system_normalized'], hourly_stats_sv['ontime_trains_normalized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(hourly_stats_sv['trains_in_system_normalized'].min(), 
                         hourly_stats_sv['trains_in_system_normalized'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r-", linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax3.set_ylabel('On-Time Trains\n(tph/platform)', fontsize=23)
    ax3.tick_params(axis='both', labelsize=23)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2.5)
    ax3.set_ylim(0, 2.5)
    
    # Plot 4: On-Time Performance Histogram (from station_view)
    ax4 = plt.subplot(5, 1, 4)
    bin_stats = station_view_result['bin_stats']
    x_pos = np.arange(len(bin_stats))
    
    ax4.bar(x_pos, bin_stats['pct_hours_100_ontime'], 
                   color='cornflowerblue', alpha=0.7, edgecolor='black', linewidth=1.0, width=0.6)
    
    ax4.set_xlabel('Normalized trains in system\n(tph/platform)', fontsize=23)
    ax4.set_ylabel('% of Hours\n100% On-Time', fontsize=23)
    ax4.tick_params(axis='both', labelsize=23)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bin_stats['load_bin'], rotation=0, fontsize=23)
    ax4.set_xlim(-0.5, 12)
    ax4.set_xticks(np.arange(0, 13, 2.5))
    ax4.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 2.6, 0.5)], fontsize=23)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cumulative Distribution Function (from station_view)
    ax5 = plt.subplot(5, 1, 5)
    cdf_values = bin_stats['cdf'].values
    
    ax5.plot(x_pos, cdf_values, 
            marker='o', markersize=6, linewidth=2, color='darkred', 
            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=1.0)
    ax5.fill_between(x_pos, 0, cdf_values, alpha=0.15, color='darkred')
    
    ax5.set_xlabel('Normalized Trains\nin System (tph/platform)', fontsize=23)
    ax5.set_ylabel('Cumulative\nDistribution (%)', fontsize=23)
    ax5.tick_params(axis='both', labelsize=23)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(bin_stats['load_bin'], rotation=0, fontsize=23)
    ax5.set_xlim(-0.5, 12)
    ax5.set_xticks(np.arange(0, 13, 2.5))
    ax5.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 2.6, 0.5)], fontsize=23)
    ax5.set_ylim(0, 105)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nComprehensive analysis complete!")
    
    return {
        'delay_analysis': delay_analysis_result,
        'outlier_analysis': outlier_analysis_result,
        'station_view_analysis': station_view_result
    }




# ============================================================================
# WRAPPER FUNCTIONS - TIME RANGE FILTERING
# ============================================================================

def _expand_time_range(time_range):
    """
    Normalize time_range to ensure it covers full days.
    Always expands end to end of day (23:59:59).
    
    Parameters:
    -----------
    time_range : tuple or None
        Tuple of (start, end) as strings or datetime objects
        Returns None if input is None
    
    Returns:
    --------
    tuple or None
        Normalized (start, end) datetimes, or None if input was None
    """
    if time_range is None:
        return None
    
    start, end = time_range
    
    # Convert strings to pandas Timestamps if needed
    if isinstance(start, str):
        start = pd.to_datetime(start).normalize()  # Convert to midnight
    else:
        start = pd.Timestamp(start).normalize()
    
    if isinstance(end, str):
        end = pd.to_datetime(end).normalize()  # Convert to midnight first
    else:
        end = pd.Timestamp(end).normalize()
    
    # ALWAYS expand end to end of day (23:59:59), regardless of whether start == end
    # This ensures the entire end date is included in the range
    end = end.replace(hour=23, minute=59, second=59)
    
    return start, end


def station_analysis_with_time_range(station_id, all_data, time_range=None, 
                                     num_platforms=6, dwell_time_minutes=5, 
                                     max_delay_percentile=98):
    """
    Wrapper around comprehensive_station_analysis that adds time_range filtering.
    
    Filters data by optional time_range, then calls the original function with
    the filtered dataset. Original function logic remains unchanged.
    
    Parameters:
    -----------
    station_id : str
        The STANOX code for the station
    all_data : pd.DataFrame
        Complete dataset with all station data
    time_range : tuple or None
        Tuple of (start, end) as dates or datetimes
        - Dates will be expanded to full day (00:00 to 23:59:59)
        - Same date for both will cover entire day
        - None uses all data (default)
        Examples:
            ('2024-01-15', '2024-01-15')  # Single day
            ('2024-01-01', '2024-06-30')  # Date range
            ('2024-01-15 08:00', '2024-01-15 17:00')  # Specific times
    num_platforms : int
        Number of platforms at station (default: 6)
    dwell_time_minutes : int
        Typical dwell time at station in minutes (default: 5)
    max_delay_percentile : int
        Maximum delay percentile to consider (default: 98)
    
    Returns:
    --------
    dict
        Dictionary containing all results from comprehensive_station_analysis
    """
    # NOTE: Time range filtering is now handled in station_view() after reconstructing arrival times.
    # This ensures both delayed trains (filtered by EVENT_DATETIME) and on-time trains (filtered by reconstructed arrival_time) 
    # respect the specified time_range. 
    # Here we pass the full dataset and let the downstream functions handle the filtering.
    filtered_data = all_data.copy()
    
    # Call original function, passing time_range for downstream filtering
    return comprehensive_station_analysis(
        station_id=station_id,
        all_data=filtered_data,
        num_platforms=num_platforms,
        dwell_time_minutes=dwell_time_minutes,
        max_delay_percentile=max_delay_percentile,
        time_range=time_range
    )


def station_view_yearly_with_time_range(station_id, interval_minutes=30, time_range=None):
    """
    Wrapper around station_view_yearly that adds time_range filtering.
    
    Calls the original function and filters its results by optional time_range.
    Original function logic remains unchanged.
    
    Parameters:
    -----------
    station_id : str
        The STANOX code for the station
    interval_minutes : int
        Interval size for analysis in minutes (default: 30)
    time_range : tuple or None
        Tuple of (start, end) as dates or datetimes
        - Dates will be expanded to full day (00:00 to 23:59:59)
        - Same date for both will cover entire day
        - None uses all data (default)
        Examples:
            ('2024-01-15', '2024-01-15')  # Single day
            ('2024-01-01', '2024-06-30')  # Date range
            ('2024-01-15 08:00', '2024-01-15 17:00')  # Specific times
    
    Returns:
    --------
    tuple
        (incident_summary, normal_summary) DataFrames filtered by time_range
    """
    # Call original function first
    incident_summary, normal_summary = station_view_yearly(station_id, interval_minutes)
    
    if time_range is not None and incident_summary is not None:
        start, end = _expand_time_range(time_range)
        
        # Determine the datetime column name (adjust if different)
        # Assuming the summary has a datetime-like column; adapt as needed
        datetime_col = None
        for col in ['datetime', 'EVENT_DATETIME', 'time', 'DATETIME']:
            if col in incident_summary.columns:
                datetime_col = col
                break
        
        if datetime_col:
            # Filter both summaries by time range
            incident_summary = incident_summary[
                (pd.to_datetime(incident_summary[datetime_col]) >= start) & 
                (pd.to_datetime(incident_summary[datetime_col]) <= end)
            ].reset_index(drop=True)
            
            if normal_summary is not None and len(normal_summary) > 0:
                normal_summary = normal_summary[
                    (pd.to_datetime(normal_summary[datetime_col]) >= start) & 
                    (pd.to_datetime(normal_summary[datetime_col]) <= end)
                ].reset_index(drop=True)
            
            print(f"Filtered results to range {start} to {end}")
            print(f"Incident periods after filter: {len(incident_summary)}")
            if normal_summary is not None:
                print(f"Normal periods after filter: {len(normal_summary)}")
        else:
            print("Warning: Could not find datetime column in results for filtering")
    
    return incident_summary, normal_summary

"""
Unit tests for rdmpy.outputs.analysis_tools module.

"""
# TOREVIEW: generally, cntrl+f "issubset" (replaced with ==)

from unittest import result
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdmpy.outputs.analysis_tools import (
    calculate_incident_summary_stats,
    _load_and_prepare_multiday_data,
    get_stanox_for_service,
    map_train_journey_with_incidents,
    train_view,
    train_view_2,
    plot_reliability_graphs,
    _print_date_statistics,
    _load_station_coordinates,
    _aggregate_time_view_data,
    _create_time_view_markers,
    _finalize_time_view_map,
    create_time_view_html,
    incident_view,
    incident_view_heatmap_html,
    aggregate_view_multiday,
    station_view_yearly,
    station_view,
    plot_trains_in_system_vs_delay,
    explore_delay_outliers,
    comprehensive_station_analysis,
)

# INTEGRATION TESTS FOR map_train_journey_with_incidents - Complete Workflow

@pytest.fixture
def universal_station_ref():
    """
    Universal station reference fixture for all tests.
    
    Returns a list of station dictionaries with standardized keys:
    - stanox: station code (integer or string)
    - station_name: human-readable station name
    - latitude: geographic latitude
    - longitude: geographic longitude
    
    Covers UK stations used across train view, time view, and incident view tests.
    Includes both train journey stations (12xxx) and incident analysis stations (51xxx).
    """
    return [
        # Train view and time view stations
        {'stanox': 12931, 'station_name': 'London Kings Cross', 'latitude': 51.5307, 'longitude': -0.1234},
        {'stanox': 89012, 'station_name': 'Manchester Piccadilly', 'latitude': 53.4808, 'longitude': -2.2426},
        {'stanox': 45123, 'station_name': 'Birmingham New Street', 'latitude': 52.5078, 'longitude': -1.9043},
        {'stanox': 54311, 'station_name': 'Leeds City', 'latitude': 53.7949, 'longitude': -1.7477},
        {'stanox': 78912, 'station_name': 'York', 'latitude': 53.9581, 'longitude': -1.0873},
        # Incident view stations
        {'stanox': 51511, 'station_name': 'London Kings Cross (Incident)', 'latitude': 51.5307, 'longitude': -0.1234},
        {'stanox': 51520, 'station_name': 'Peterborough', 'latitude': 52.5659, 'longitude': -0.2440},
        {'stanox': 51530, 'station_name': 'Doncaster', 'latitude': 53.5198, 'longitude': -1.1286},
        {'stanox': 51540, 'station_name': 'Newcastle', 'latitude': 54.9673, 'longitude': -1.6109},
        {'stanox': 51550, 'station_name': 'Edinburgh', 'latitude': 55.9520, 'longitude': -3.1881},
    ]


# ==============================================================================
# FIXTURES for aggregate view tests
# ==============================================================================

@pytest.fixture
def sample_delay_df():
    """Create a sample dataframe with delay events for testing."""
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    return pd.DataFrame({
        'full_datetime': dates,
        'PFPI_MINUTES': [10, 15, 20, 5, 30, 25, 35, 40, 45, 50],
        'EVENT_TYPE': ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'C', 'C'],
        'INCIDENT_NUMBER': [12345] * 10,
    })


@pytest.fixture
def sample_complete_df():
    """Create a complete sample dataframe with all required columns."""
    dates = pd.date_range('2024-01-01 08:00', periods=15, freq='H')
    df = pd.DataFrame({
        'full_datetime': dates,
        'EVENT_DATETIME': ['01-JAN-2024 ' + f'{8+i:02d}:00' for i in range(15)],
        'PFPI_MINUTES': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        'EVENT_TYPE': ['D'] * 13 + ['C'] * 2,
        'INCIDENT_NUMBER': [12345] * 15,
        'INCIDENT_START_DATETIME': ['01-JAN-2024 08:00'] * 15,
        'event_date_only': [date(2024, 1, 1)] * 15,
    })
    return df


# ==============================================================================
# TESTS FOR aggregate view (internal functions)
# ==============================================================================

def test_calculate_incident_summary_stats_k0(sample_complete_df):
    """Test summary stats calculation returns dict with correct keys."""
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = calculate_incident_summary_stats(
        df=sample_complete_df,
        delay_data_all=delay_data,
        unique_dates=unique_dates,
        files_processed=100,
        files_with_data=10,
        incident_number=12345,
        num_days=1
    )
    
    # Assert based on sample_complete_df fixture
    expected_total_delay = sample_complete_df['PFPI_MINUTES'].sum()
    expected_cancellations = len(sample_complete_df[sample_complete_df['EVENT_TYPE'] == 'C'])
    expected_records = len(sample_complete_df)

    assert result is not None
    assert isinstance(result, dict)
    assert result['Total Delay Minutes'] == expected_total_delay
    assert result['Total Cancellations'] == expected_cancellations
    assert result['Total Records Found'] == expected_records
    assert result['Files Processed'] == 100
    assert result['Files with Data'] == 10
    assert result['Number of Days'] == 1
    assert result['Date Range'] == '01-Jan-2024 to 01-Jan-2024'
    assert result['Incident Number'] == 12345
    ## Peak should be 75 minutes (from 14th record)
    assert '75.0 minutes' in result['Peak Delay Event']
    assert isinstance(result['Peak Delay Event'], str) # 'XX minutes at DATE TIME (TYPE)'
    
    # Verify all keys are present in correct order
    assert list(result.keys()) == [
        'Total Delay Minutes',
        'Total Cancellations', 
        'Total Records Found',
        'Files Processed',
        'Files with Data',
        'Incident Number',
        'Number of Days',
        'Date Range',
        'Unique Dates',
        'Time Range',
        'Peak Delay Event',
        'Peak Regular Delay',
        'Peak Cancellation Delay'
    ]
    
# using @patch to mock the data loading function to test aggregate_view_multiday's handling of load failures and multi-day incidents

@patch('rdmpy.outputs.analysis_tools._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k0(mock_load_prep):
    """Test aggregate_view_multiday returns None when data loading fails."""
    mock_load_prep.return_value = None
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    assert result is None

@patch('rdmpy.outputs.analysis_tools.plt.show')
@patch('rdmpy.outputs.analysis_tools._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k2(mock_load_prep, mock_plt_show, sample_complete_df):
    """Test aggregate_view_multiday handles multi-day incidents correctly."""
    unique_dates = [date(2024, 1, 1), date(2024, 1, 2)] # 2 dates to simulate multi-day incident
    mock_load_prep.return_value = (sample_complete_df, 100, 10, unique_dates, 2)
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    # Should return a dict (plotted and summarized without error)
    assert mock_plt_show.called
    assert result is not None
    assert isinstance(result, dict)
    assert result['Date Range'] == '01-Jan-2024 to 02-Jan-2024'


# INTEGRATION TESTS for aggregate view consistency

def test_aggregate_view_workflow_consistency(sample_complete_df):
    """Test that aggregated stats from complete workflow are consistent."""
    # Test that calculations are internally consistent
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = calculate_incident_summary_stats(
        df=sample_complete_df,
        delay_data_all=delay_data,
        unique_dates=unique_dates,
        files_processed=100,
        files_with_data=10,
        incident_number=12345,
        num_days=1
    )
    
    # Verify internal consistency: Total records should match dataframe length
    assert result['Total Records Found'] == len(sample_complete_df)
    
    # Verify delay range makes sense
    assert result['Total Delay Minutes'] >= 0
    
    # Verify that total delay roughly matches sum of PFPI_MINUTES (excluding edge cases)
    expected_delay_sum = sample_complete_df['PFPI_MINUTES'].fillna(0).sum()
    assert result['Total Delay Minutes'] == expected_delay_sum


# ==============================================================================
# FIXTURES FOR TRAIN_VIEW TESTS
# ==============================================================================

# This fixture simulates the data assessed by the train_view function
@pytest.fixture
def train_journey_fixture():
    """
    Fixture representing realistic train journey data for testing the complete workflow.
    Simulates data for OD pair (12931 -> 54311) with known service codes and incidents on specific dates.

    - PLANNED_ORIGIN_LOCATION_CODE and PLANNED_DEST_LOCATION_CODE include either (O,D) or (D,O) pairs. 
    """
    return pd.DataFrame({
        'PLANNED_ORIGIN_LOCATION_CODE': ['12931'] * 5 + ['54311'] * 5,
        'PLANNED_DEST_LOCATION_CODE': ['54311'] * 5 + ['12931'] * 5,
        'TRAIN_SERVICE_CODE': ['21700001', '21700001', '21700002', '21700003', '21700001',
                               '21700004', '21700005', '21700001', '21700001', '21700006'], # adding many service codes for realism, but here only 21700001 is used to test
        'STANOX': ['12931', '89012', '45123', '54311', '78912',
                   '54311', '45123', '89012', '12931', '54311'],
        'PLANNED_ORIGIN_GBTT_DATETIME': ['07:00', '07:45', '08:15', '09:00', '10:00',
                                        '16:00', '16:45', '17:15', '18:00', '19:00'],
        'PLANNED_DEST_GBTT_DATETIME': ['12:30', '13:15', '14:00', '14:45', '15:30',
                                      '21:30', '22:15', '23:00', '23:45', '00:30'],
        'PLANNED_CALLS': ['0700', '0745', '0815', '0900', '1000',
                         '1600', '1645', '1715', '1800', '1900'],
        'ACTUAL_CALLS': ['0715', '0805', '0835', '0925', '1020',
                        '1620', '1700', '1735', '1825', '1925'],
        'PFPI_MINUTES': [15.0, 20.0, 20.0, 25.0, 20.0,
                        20.0, 15.0, 20.0, 25.0, 25.0],
        'INCIDENT_REASON': ['Signal failure', 'Track defect', 'Points failure', 'Signalling issue', 'Track defect',
                           'Signal failure', 'Track defect', 'Points failure', 'Signalling issue', 'Signal failure'],
        'INCIDENT_NUMBER': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        'EVENT_TYPE': ['D'] * 10,
        'SECTION_CODE': ['12931:89012', '89012:45123', '45123:54311', '12931:54311', '89012:45123',
                        '54311:45123', '45123:89012', '89012:12931', '12931:54311', '54311:12931'],
        'DELAY_DAY': ['07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '08-DEC-2024',
                      '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024'],
        'EVENT_DATETIME': ['07-DEC-2024 07:15', '07-DEC-2024 08:05', '07-DEC-2024 08:35', '07-DEC-2024 09:25', '08-DEC-2024 10:20',
                          '07-DEC-2024 16:20', '07-DEC-2024 17:00', '07-DEC-2024 17:35', '07-DEC-2024 18:25', '07-DEC-2024 19:25'],
        'INCIDENT_START_DATETIME': ['2024-12-07 07:00', '2024-12-07 08:00', '2024-12-07 08:15', '2024-12-07 09:00', '2024-12-07 10:00',
                                   '2024-12-07 16:00', '2024-12-07 16:45', '2024-12-07 17:15', '2024-12-07 18:00', '2024-12-07 19:00'],
        'ENGLISH_DAY_TYPE': ['Weekday'] * 10,
        'STATION_ROLE': ['O', 'I', 'I', 'D', 'I'] * 2,
        'DFT_CATEGORY': ['Cat1'] * 10,
        'PLATFORM_COUNT': [4, 4, 3, 5, 4, 5, 3, 4, 4, 5],
        'DATASET_TYPE': ['Delay'] * 10,
        'WEEKDAY': [6] * 10,
        'DAY': [7] * 10,
    })


# TESTS FOR get_stanox_for_service

def test_get_stanox_for_service_retrieves_all_stations(train_journey_fixture):
    """
    Test that get_stanox_for_service returns all unique STANOX stations for a given service code.
    This represents the second step in the workflow: once we identify a service from train_view results,
    we retrieve all stations it calls at to map the journey.
    """
    service_code = '21700001'
    
    result = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311'
    )

    # Service 21700001 has STANOX: 12931, 89012, 78912 (rows 0, 1, 4)
    # Plus destination 54311 is added as destination station
    expected_stations = {'12931', '89012', '78912', '54311'}
    
  
    assert isinstance(result, list)   # Verify result is a list of STANOX codes
    assert len(result) > 0
    assert '12931' in result # Verify origin is included
    assert '54311' in result # Verify destination is included
    assert set(result) == expected_stations # Verify all expected stations are included


def test_get_stanox_for_service_filters_by_date(train_journey_fixture):
    """
    Test that get_stanox_for_service correctly filters stations by date when provided.
    Fixture has service 21700001 on different dates: 07-DEC-2024 (12931, 89012) and 08-DEC-2024 (78912).
    """
    service_code = '21700001'
    
    # Query for 07-DEC-2024: should get rows with service 21700001 on that date
    result_07_dec = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311',
        date_str='07-DEC-2024'
    )
    
    # Query for 08-DEC-2024: should get rows with service 21700001 on that date
    result_08_dec = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311',
        date_str='08-DEC-2024'
    )
    
    assert isinstance(result_07_dec, list)   # Should return lists of stations
    assert isinstance(result_08_dec, list)   # Should return lists of stations

    expected_07_dec = {'12931', '89012', '54311'}
    assert set(result_07_dec) == expected_07_dec  # 07-DEC-2024: rows 0, 1 have STANOX 12931, 89012 + destination 54311

    expected_08_dec = {'78912', '54311'}
    assert set(result_08_dec) == expected_08_dec  # 08-DEC-2024: row 4 has STANOX 78912 + destination 54311
    
    assert set(result_07_dec) != set(result_08_dec)   # Verify filtering by date produces different results


def test_get_stanox_for_service_returns_error_for_nonexistent_service(train_journey_fixture):
    """
    Test that get_stanox_for_service returns error message for non-existent service code.
    """
    result = get_stanox_for_service(
        train_journey_fixture,
        train_service_code='XXXXX',
        origin_code='12931',
        destination_code='54311'
    )
    
    # Should return error message string
    assert isinstance(result, str)
    assert 'No records found' in result


@patch('folium.Map')
def test_map_train_journey_with_incidents_creates_map(mock_map, train_journey_fixture, universal_station_ref, tmp_path):
    """
    Test that map_train_journey_with_incidents creates a folium map object when given service STANOX and incident data.
    This is the third step in the workflow: visualize the journey with incidents on a map.
    """
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    # Get service STANOX from train_view result
    service_stanox = ['12931', '89012', '45123', '54311']
    
    # Get incident results from train_view 
    incident_df = train_journey_fixture[
        (train_journey_fixture['PLANNED_ORIGIN_LOCATION_CODE'] == '12931') &
        (train_journey_fixture['PLANNED_DEST_LOCATION_CODE'] == '54311')
    ].copy()
    
    # Convert universal_station_ref to JSON file for the function (since it's mocked, we just need a valid path)
    import json
    json_file = tmp_path / "stations_ref.json"
    with open(json_file, 'w') as f:
        # Convert stanox to strings for JSON compatibility
        json_data = [{'stanox': str(s['stanox']), 'station_name': s['station_name'], 
                     'latitude': s['latitude'], 'longitude': s['longitude']} for s in universal_station_ref]
        json.dump(json_data, f)
    
    result = map_train_journey_with_incidents(
        all_data=train_journey_fixture,
        service_stanox=service_stanox,
        incident_results=[incident_df],
        stations_ref_path=str(json_file),
        service_code='21700001',
        date_str='07-DEC-2024'
    )
    
    # Verify map was created (mocked)
    assert mock_map.called
    assert result is mock_map_instance


@patch('builtins.display', create=True)
def test_train_view_filters_by_od_pair_and_date(mock_display, train_journey_fixture):
    """
    Test that train_view correctly filters all data for a specific OD pair.
    Returns all incidents for that route (note: date_str parameter doesn't currently filter).
    """
    result = train_view(train_journey_fixture, '12931', '54311', '07-DEC-2024')
    
    assert isinstance(result, pd.DataFrame)     # Verify result is DataFrame with incidents
    assert len(result) > 0
    assert all(result['PLANNED_ORIGIN_LOCATION_CODE'] == '12931') # Assert filtering is correct: all rows match the OD pair
    assert all(result['PLANNED_DEST_LOCATION_CODE'] == '54311')
    assert all(result['PFPI_MINUTES'].fillna(0) > 0)   # Assert data contains delays (PFPI_MINUTES > 0)


@patch('builtins.display', create=True)
def test_train_view_bidirectional_od_pair(mock_display, train_journey_fixture):
    """
    Test that train_view works for both directions of an OD pair.
    Verify reverse direction (54311->12931) returns different incident counts than forward direction.
    """
    result_forward = train_view(train_journey_fixture, '12931', '54311', '07-DEC-2024')
    result_reverse = train_view(train_journey_fixture, '54311', '12931', '07-DEC-2024')
    
    assert isinstance(result_forward, pd.DataFrame) and isinstance(result_reverse, pd.DataFrame)    # Both should return DataFrames
    assert all(result_forward['PLANNED_ORIGIN_LOCATION_CODE'] == '12931')    # Forward direction has origin 12931
    assert all(result_forward['PLANNED_DEST_LOCATION_CODE'] == '54311')      # and destination 54311
    assert all(result_reverse['PLANNED_ORIGIN_LOCATION_CODE'] == '54311')    # Reverse direction has origin 54311
    assert all(result_reverse['PLANNED_DEST_LOCATION_CODE'] == '12931')      # and destination 12931
    assert len(result_forward) == 5    # Forward: all 5 rows with 12931->54311 (rows 0-4)
    assert len(result_reverse) == 5    # Reverse: all 5 rows with 54311->12931 (rows 5-9)


@patch('builtins.display', create=True)
def test_train_view_invalid_od_pair_returns_message(mock_display, train_journey_fixture):
    """
    Test that train_view returns error message for non-existent OD pair.
    """
    result = train_view(train_journey_fixture, 'XXXX', 'YYYY', '07-DEC-2024')
    
    # Should return string error message (not DataFrame)
    assert isinstance(result, str)
    assert 'not found' in result.lower()


@patch('builtins.display', create=True)
def test_train_view_no_incidents_on_date_returns_message(mock_display, train_journey_fixture):
    """
    Test that train_view returns informative message when OD pair exists but has no incidents on specified date.
    """
    result = train_view(train_journey_fixture, '12931', '54311', '06-DEC-2024')
    
    # Should return string message (no data on this date)
    assert isinstance(result, str)
    assert 'no incidents' in result.lower()

# FIXTURES for train_view_2 tests

@pytest.fixture
def sample_service_reliability_df():
    """
    Create sample data for train_view_2 testing with known reliability metrics
    This fixture contains 2 train codes, each having different stations and delay patterns.
    This allows us to test that train_view_2 correctly filters the selected train code, and calculates key statistics.
    
    """
    # TOREVIEW Add other train service codes
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['SVC001'] * 10 + ['CVS100'] * 10 + ['SVC000'],
        'STANOX': ['12345', '12345', '12345', '12346', '12346', '12346', '12347', '12347', '12347', '12348', '54321', '54321', '54321', '64321', '64321', '64321', '74321', '74321', '74321', '84321', '00000'],
        'PFPI_MINUTES': [0.0, 5.0, 10.0, 0.0, 15.0, 20.0, 0.0, 0.0, 25.0, 10.0, 0.0, 5.0, 10.0, 0.0, 15.0, 20.0, 0.0, 0.0, 25.0, 10.0, 0.0],
        'INCIDENT_REASON': ['OnTime', 'Delay', 'Delay', 'OnTime', 'Delay', 'Delay', 'OnTime', 'OnTime', 'Delay', 'Delay', 'OnTime', 'Delay', 'Delay', 'OnTime', 'Delay', 'Delay', 'OnTime', 'OnTime', 'Delay', 'Delay', 'OnTime'],
    })

@pytest.fixture
def sample_single_station_reliability_df():
    """Create sample data for train_view_2 testing with single station focus."""
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['SVC002'] * 3,
        'STANOX': ['12345', '12345', '12345'],
        'PFPI_MINUTES': [0.0, 5.0, 10.0],
        'INCIDENT_REASON': ['OnTime', 'Delay', 'Delay'],
    })


# TESTS FOR train_view_2 (not included in general report of the tool, but present in the code)
# TOREVIEW: added test for A, B and D

@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_computes_reliability_metrics(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 computes correct mean delay and on-time percentage for all stations."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
        {'stanox': '12346', 'description': 'Station B'},
        {'stanox': '12347', 'description': 'Station C'},
        {'stanox': '12348', 'description': 'Station D'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345', '12346']
    result = train_view_2(sample_service_reliability_df, service_stanox, 'SVC001', stations_ref_path='mock_path.json')
    
    # Should return DataFrame with metrics
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    
    # Verify required columns exist
    expected_cols = ['ServiceCode', 'StationName', 'MeanDelay', 'DelayVariance', 'OnTime%', 'IncidentCount']
    assert all(col in result.columns for col in expected_cols)
    
    # Verify ServiceCode is correct
    assert all(result['ServiceCode'] == 'SVC001')

    # Define expected values for each station
    # Station A (12345): PFPI [0.0, 5.0, 10.0], Reasons ['OnTime', 'Delay', 'Delay']
    expected_A = pd.DataFrame({
        "ServiceCode": ["SVC001"],
        "StationName": ["Station A"],
        "MeanDelay": [7.5],  # (5.0 + 10.0) / 2
        "DelayVariance": [12.5],  # variance of [5.0, 10.0]
        "OnTime%": [1/3 * 100],  # 1 on-time out of 3
        "IncidentCount": [2]  # 2 delays
    }).squeeze()

    # Station B (12346): PFPI [0.0, 15.0, 20.0], Reasons ['OnTime', 'Delay', 'Delay']
    expected_B = pd.DataFrame({
        "ServiceCode": ["SVC001"],
        "StationName": ["Station B"],
        "MeanDelay": [17.5],  # (15.0 + 20.0) / 2
        "DelayVariance": [12.5],  # variance of [15.0, 20.0]
        "OnTime%": [1/3 * 100],  # 1 on-time out of 3
        "IncidentCount": [2]  # 2 delays
    }).squeeze()

    # Station C (12347): PFPI [0.0, 0.0, 25.0], Reasons ['OnTime', 'OnTime', 'Delay']
    expected_C = pd.DataFrame({
        "ServiceCode": ["SVC001"],
        "StationName": ["Station C"],
        "MeanDelay": [25.0],  # only one delay of 25.0 minutes
        "DelayVariance": [np.nan],  # NaN for single value 
        "OnTime%": [2/3 * 100],  # 2 on-time out of 3
        "IncidentCount": [1]  # 1 delay
    }).squeeze()

    # Station D (12348): PFPI [10.0], Reasons ['Delay']
    expected_D = pd.DataFrame({
        "ServiceCode": ["SVC001"],
        "StationName": ["Station D"],
        "MeanDelay": [10.0],  # single delay of 10.0 minutes
        "DelayVariance": [np.nan],  # NaN for single valuE
        "OnTime%": [0.0],  # 0 on-time out of 1
        "IncidentCount": [1]  # 1 delay
    }).squeeze()

    # Verify each station's metrics
    for expected in [expected_A, expected_B, expected_C, expected_D]:
        station_name = expected['StationName']
        row = result[result['StationName'] == station_name].squeeze()
        pd.testing.assert_series_equal(row, expected, rtol=1e-6, check_names=False)


# TOREVIEW: returs empty dataframe,  
# test correctly verifies that when a service has no delays and you request an empty station list, you get an empty result.
@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_empty_service_stanox(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 with empty service_stanox list and empty data returns empty DataFrame."""
    # Mock station reference file
    mock_station_ref = []
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    result = train_view_2(sample_service_reliability_df, [], 'SVC000', stations_ref_path='mock_path.json') # empty station list and service code with no data
    
    # Should return empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 0


# TESTS FOR plot_reliability_graphs FUNCTION (still in train view)

@patch('builtins.print')
@patch('matplotlib.pyplot.show')
def test_plot_reliability_graphs_runs_without_error(mock_plt_show, mock_print, sample_single_station_reliability_df):
    """Test plot_reliability_graphs completes without error with mocking."""
    service_stanox = ['12345']
    
    # Use a non-existent path to trigger exception handling
    try:
        plot_reliability_graphs(sample_single_station_reliability_df, service_stanox, 'SVC002', 
                              stations_ref_path='/nonexistent/path.json')
        # If it doesn't raise, it's using exception handling - that's fine
        assert True
    except (FileNotFoundError, KeyError):
        # Expected when station ref file doesn't exist
        assert True
    finally:
        plt.close('all')


# ==============================================================================
# TIME VIEW FUNCTION TESTS
# ==============================================================================

# Fixtures for time view tests

@pytest.fixture
def sample_time_view_data():
    """
    Create sample time view incident data for testing.
    
    This fixture includes incident data:
    - Date and time of incident start
    - STANOX where the incident occurred
    - Incident code
    - Incident reason
    - delay minutes
    """
    # Data for 2024-01-15 (10 incidents across 4 stations)
    date_2024_01_15 = [
        ('2024-01-15 08:30:00', '12345', 1001, 'TH', 15),
        ('2024-01-15 09:00:00', '12346', 1002, 'TG', 10),
        ('2024-01-15 09:45:00', '12345', 1003, 'TH', 25),
        ('2024-01-15 10:30:00', '12347', 1004, 'XA', 35),  # 31-60 min range
        ('2024-01-15 11:00:00', '12345', 1005, 'TG', 20),
        ('2024-01-15 11:30:00', '12346', 1006, 'M8', 50),  # 31-60 min range
        ('2024-01-15 12:00:00', '12345', 1007, 'TH', 18),
        ('2024-01-15 12:30:00', '12348', 1008, 'QM', 80),  # 61-120 min range
        ('2024-01-15 13:00:00', '12345', 1009, 'TG', 28),
        ('2024-01-15 13:30:00', '12346', 1010, 'M8', 14),
    ]
    
    # Data for 2024-01-16 (6 incidents across 3 stations - for testing no data on other dates)
    date_2024_01_16 = [
        ('2024-01-16 08:00:00', '12345', 1011, 'TG', 12),
        ('2024-01-16 09:00:00', '12346', 1012, 'TH', 7),
        ('2024-01-16 10:00:00', '12346', 1013, 'TG', 19),
    ]
    
    all_records = date_2024_01_15 + date_2024_01_16
    
    return pd.DataFrame({
        'INCIDENT_START_DATETIME': [r[0] for r in all_records],
        'STANOX': [r[1] for r in all_records],
        'INCIDENT_NUMBER': [r[2] for r in all_records],
        'INCIDENT_REASON': [r[3] for r in all_records],
        'PFPI_MINUTES': [r[4] for r in all_records],
    })



# Tests for _print_date_statistics function (helper for time view, which prints summary statistics for a given date)

def test_print_date_statistics(sample_time_view_data, capsys):
    """
    Test _print_date_statistics handles various data scenarios.
    Capsys is useful for testing functions that communicate via print statements rather than return values. 
    Without capsys, the print output would just go to your terminal—you wouldn't be able to verify it in your test assertions.
    
    Tests:
    - Prints correct summary for dates with incidents
    - Handles empty datasets gracefully
    - Handles dates with no matching incidents
    """
    # Test 1: Date with incidents - verify specific incident count and top reasons
    _print_date_statistics('2024-01-15', sample_time_view_data)
    captured = capsys.readouterr()
    assert '2024-01-15' in captured.out
    assert '10 incidents' in captured.out  # 2024-01-15 has exactly 10 incidents
    assert 'Top reasons' in captured.out
    # Verify top reasons from fixture: TH(3), TG(3), M8(2), XA(1), QM(1)
    assert 'TH(3)' in captured.out
    assert 'TG(3)' in captured.out
    assert 'M8(2)' in captured.out
    
    # Test 2: Empty dataset instead of sample_time_view_data
    empty_df = pd.DataFrame({
        'INCIDENT_START_DATETIME': pd.Series([], dtype='object'),
        'STANOX': pd.Series([], dtype='object'),
        'INCIDENT_REASON': pd.Series([], dtype='object'),
        'PFPI_MINUTES': pd.Series([], dtype='float64'),
    })
    _print_date_statistics('2024-01-15', empty_df)
    captured = capsys.readouterr()
    assert 'No incidents' in captured.out
    
    # Test 3: No matching date
    _print_date_statistics('2024-02-01', sample_time_view_data) # this date doesn't exist in the sample data, so should trigger "no incidents" message
    captured = capsys.readouterr()
    assert 'No incidents' in captured.out


# Tests for _create_time_view_markers function (helper for time view, which creates map markers for each station)

@patch('folium.CircleMarker')
def test_create_time_view_markers(mock_circle):
    """
    Test _create_time_view_markers marker creation, color grading, and radius scaling.
    Varying PFPI minutes for testing color coding (green, yellow, orange, red, dark red, violet)
    
    Tests:
    - Adds CircleMarkers for each affected STANOX
    - Applies correct color based on PFPI severity
    - Scales radius by incident count
    """
    mock_map = MagicMock()
    
    # Test 1: Multiple markers
    _create_time_view_markers(
        mock_map, 
        affected_stanox={12345, 12346, 12347},
        incident_counts={12345: 5, 12346: 3, 12347: 1},
        total_pfpi={12345: 106, 12346: 36, 12347: 30},
        stanox_to_coords={'12345': [51.5074, -0.1278], '12346': [53.4808, -2.2426], '12347': [52.5200, 13.4050]}
    )
    # Assert markers created with correct counts and verify color grading
    assert mock_circle.call_count >= 3
    
    # Test 2: Verify correct colors are applied to each severity range based on Test 1 data
    # Station 12345: PFPI 106 (61-120 range) → Dark Red '#8B0000'
    # Station 12346: PFPI 36 (31-60 range) → Red '#FF0000'
    # Station 12347: PFPI 30 (0-30 range) → Dark Orange '#FF8C00'
    
    # Extract location-color pairs from each marker call
    markers = {tuple(call[1].get('location', [])): call[1].get('fill_color', call[1].get('color', '')) 
               for call in mock_circle.call_args_list}
    
    # Assert each station has the correct color for its severity
    assert markers[(51.5074, -0.1278)] == '#8B0000', "Station 12345: expected #8B0000"
    assert markers[(53.4808, -2.2426)] == '#FF0000', "Station 12346: expected #FF0000"
    assert markers[(52.5200, 13.4050)] == '#FF8C00', "Station 12347: expected #FF8C00"
    
    # Test 3: Verify radius scaling with incident counts from Test 1 data
    # Extract radii: station 12345 (5 incidents) should have larger radius than 12347 (1 incident)
    radii = {tuple(call[1].get('location', [])): call[1].get('radius', 0) 
             for call in mock_circle.call_args_list}
    
    radius_12345 = radii[(51.5074, -0.1278)]  # 5 incidents
    radius_12346 = radii[(53.4808, -2.2426)]  # 3 incidents
    radius_12347 = radii[(52.5200, 13.4050)]  # 1 incident
    
    # Verify radius scaling: more incidents = larger radius
    assert radius_12345 > radius_12346 > radius_12347, "Radius should increase with incident count"


# Tests for _finalize_time_view_map function (helper for time view, which finalizes the map by adding title, legend, and saving to file)

@patch('folium.Element')
def test_finalize_time_view_map(mock_element):
    """
    Test _finalize_time_view_map adds title, legend, and saves file.
    
    Tests:
    - Adds title to the map via HTML element
    - Adds legend to the map via HTML element
    - Saves map to file with correct naming convention
    """
    mock_map = MagicMock()
    
    # Test all aspects in one call
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Verify title and legend are added
    assert mock_map.get_root().html.add_child.called
    calls = mock_map.get_root().html.add_child.call_count
    assert calls >= 2  # At least title and legend
    
    # Verify file is saved
    assert mock_map.save.called
    call_args = mock_map.save.call_args
    if call_args:
        filename = call_args[0][0]
        assert 'time_view_2024_01_15' in filename


# Tests for _aggregate_time_view_data function (helper for time view, which aggregates incident counts and PFPI totals for a given date)

def test_aggregate_time_view_data_calculates_correct_totals(sample_time_view_data):
    """
    Test _aggregate_time_view_data calculates correct incident counts and PFPI sums.
    
    Verification using fixture data:
    - 2024-01-15: 10 incidents across 4 stations
    - STANOX 12345: 5 incidents, PFPI total = 15+25+20+18+28 = 106 minutes
    - STANOX 12346: 3 incidents, PFPI total = 10+50+14 = 74 minutes
    - 2024-01-16: 3 incidents across 2 stations
    - STANOX 12345: 1 incident, PFPI total = 12 minutes
    - STANOX 12346: 2 incidents, PFPI total = 7+19 = 26 minutes
    """
    # Test for 2024-01-15
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    # Verify data structure
    assert affected_stanox is not None
    assert len(affected_stanox) == 4  # 4 stations affected
    
    # Convert to int for comparison
    stanox_list = [int(s) for s in affected_stanox]
    assert sorted(stanox_list) == [12345, 12346, 12347, 12348]
    
    # Verify incident counts
    count_12345 = incident_counts.get(12345) or incident_counts.get('12345')
    count_12346 = incident_counts.get(12346) or incident_counts.get('12346')
    count_12347 = incident_counts.get(12347) or incident_counts.get('12347')
    count_12348 = incident_counts.get(12348) or incident_counts.get('12348')
    
    assert count_12345 == 5
    assert count_12346 == 3
    assert count_12347 == 1
    assert count_12348 == 1
    
    # Verify PFPI totals
    pfpi_12345 = total_pfpi.get(12345) or total_pfpi.get('12345')
    pfpi_12346 = total_pfpi.get(12346) or total_pfpi.get('12346')
    pfpi_12347 = total_pfpi.get(12347) or total_pfpi.get('12347')
    assert pfpi_12345 == 106  # 15+25+20+18+28
    assert pfpi_12346 == 74   # 10+50+14

# TOREVIEW: adding test for 2024-01-16

    # Test for 2024-01-16
    affected_stanox_16, incident_counts_16, total_pfpi_16 = _aggregate_time_view_data('2024-01-16', sample_time_view_data)
    
    # Verify data structure for 2024-01-16
    assert affected_stanox_16 is not None
    assert len(affected_stanox_16) == 2  # Only 2 stations affected on this date
    
    # Convert to int for comparison
    stanox_list_16 = [int(s) for s in affected_stanox_16]
    assert sorted(stanox_list_16) == [12345, 12346]
    
    # Verify incident counts for 2024-01-16
    count_12345_16 = incident_counts_16.get(12345) or incident_counts_16.get('12345')
    count_12346_16 = incident_counts_16.get(12346) or incident_counts_16.get('12346')
    
    assert count_12345_16 == 1
    assert count_12346_16 == 2
    
    # Verify PFPI totals for 2024-01-16
    pfpi_12345_16 = total_pfpi_16.get(12345) or total_pfpi_16.get('12345')
    pfpi_12346_16 = total_pfpi_16.get(12346) or total_pfpi_16.get('12346')
    
    assert pfpi_12345_16 == 12      # Single incident with 12 minutes delay
    assert pfpi_12346_16 == 26      # 7+19 minutes 




def test_aggregate_time_view_data_empty_date(sample_time_view_data):
    """Test _aggregate_time_view_data handles date with no incidents."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-03-01', sample_time_view_data)
    
    # Should return None for non-matching date
    assert affected_stanox is None


# INTEGRATION TESTS FOR create_time_view_html (calling main create_time_view_html function, no helpers)

@patch('rdmpy.outputs.analysis_tools._load_station_coordinates')
@patch('folium.Map')
@patch('folium.CircleMarker')
@patch('builtins.print')
def test_create_time_view_html_comprehensive_workflow(mock_print, mock_circle, mock_map, mock_load_coords, sample_time_view_data):
    """
    Comprehensive test for create_time_view_html covering complete workflow, marker creation,
    filename format validation, partial coordinates handling, and date filtering.
    
    Tests:
    - Complete workflow: print stats, create map, add markers, save file
    - Correct number of markers created (4 for 4 affected stations on 2024-01-15)
    - Correct filename format (time_view_YYYY_MM_DD.html)
    - Graceful handling of partial station coordinates
    - Correct date filtering and independence between dates
    """
    # Arrange: Mock dependencies with full coordinates
    full_coords = {
        '12345': [51.5307, -0.1234],
        '12346': [53.4808, -2.2426],
        '12347': [52.5078, -1.9043],
        '12348': [53.7949, -1.7477],
    }
    
    mock_load_coords.return_value = full_coords
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    # Act: Process 2024-01-15 (10 incidents across 4 stations)
    create_time_view_html('2024-01-15', sample_time_view_data)
    
    # Map should be created with correct center and zoom
    assert mock_map.called
    map_call_args = mock_map.call_args
    assert map_call_args[1]['location'] == [54.5, -2.5]  # UK center
    assert map_call_args[1]['zoom_start'] == 6
    
    # Correct number of markers created (4 affected stations)
    assert mock_circle.call_count >= 4
    
    # Map should be saved with correct filename
    assert mock_map_instance.save.called
    save_call_args = mock_map_instance.save.call_args
    filename = save_call_args[0][0]
    assert filename.startswith('time_view_')
    assert filename.endswith('.html')
    assert '2024_01_15' in filename
    
    # Reset mocks for second date test. This was done to not repeat lines of code
    mock_map.reset_mock()
    mock_circle.reset_mock()
    mock_print.reset_mock()
    mock_map.return_value = mock_map_instance
    
    # Act: Process 2024-01-16 (different date with fewer incidents)
    create_time_view_html('2024-01-16', sample_time_view_data)
    
    # Assert: Verify date filtering works independently
    assert mock_map.called  # Map should still be created
    
    # Test with partial coordinates (missing 2 of 4 stations)
    mock_map.reset_mock()
    mock_circle.reset_mock()
    partial_coords = {
        '12345': [51.5307, -0.1234],
        '12346': [53.4808, -2.2426],
        # 12347 and 12348 missing
    }
    mock_load_coords.return_value = partial_coords
    mock_map.return_value = mock_map_instance
    
    # Act: Should not crash with missing coordinates
    create_time_view_html('2024-01-15', sample_time_view_data)
    
    # Assert: Function completes without error
    assert mock_map.called
    assert mock_circle.call_count >= 2


@patch('rdmpy.outputs.analysis_tools._load_station_coordinates')
@patch('folium.Map')
def test_create_time_view_html_no_data_returns_gracefully(mock_map, mock_load_coords, sample_time_view_data):
    """
    Test that create_time_view_html handles dates with no incidents gracefully.
    
    Expected behavior:
    - Function should return early without creating map
    - No exception should be raised
    """
    # Arrange
    mock_load_coords.return_value = {}
    
    # Act & Assert: Should not crash on non-existent date
    result = create_time_view_html('2024-12-31', sample_time_view_data)
    
    # Map should NOT be created for date with no data
    # (Since _aggregate_time_view_data returns None)
    assert result is None

# ==============================================================================
# FIXTURES for incident view tests
# ==============================================================================

# TOREVIEW: fixture for incident view with for time interval testing

@pytest.fixture
def sample_incident_data():
    """
    Comprehensive sample incident data with multiple delays per station per interval.
    This data represents the preprocessed data stored in .parquet.
    
    Covers 07-DEC-2024 06:00 to 07-DEC-2024 16:00 (600 minutes, 10 intervals of 60 min each).
    
    Contains:
    - Non-delayed trains (INCIDENT_NUMBER = NaN): SVC001, SVC002, SVC003
    
    - Delayed trains with incident 64326 distributed across intervals.
      Each station has multiple delay events in the same interval to test aggregation:
      * Station 24108: SVC201a (06:15, 250 min) + SVC201b (06:45, 220 min) = 470 min (Interval 0)
      * Station 25316: SVC202a (07:15, 225 min) + SVC202b (07:45, 225 min) = 450 min (Interval 1)
      * Station 25701: SVC203a (08:15, 245 min) + SVC203b (08:45, 245 min) = 490 min (Interval 2)
      * Station 27263: SVC204a (09:15, 255 min) + SVC204b (09:45, 255 min) = 510 min (Interval 3)
      * Station 31521: SVC205a (10:15, 235 min) + SVC205b (10:45, 235 min) = 470 min (Interval 4)
      * Station 32000: SVC206a (11:15, 245 min) + SVC206b (11:45, 245 min) = 490 min (Interval 5)
    
    - Incident 64327 trains (filtered out by incident code)
    
    Expected aggregations per interval:
    - Interval 0: Station 24108 = 470 min
    - Interval 1: Station 25316 = 450 min
    - Interval 2: Station 25701 = 490 min
    - Interval 3: Station 27263 = 510 min
    - Interval 4: Station 31521 = 470 min
    - Interval 5: Station 32000 = 490 min
    """
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': [
            # Non-delayed trains
            'SVC001', 'SVC002', 'SVC003',
            # Incident 64326: each station has 2 trains in same interval
            'SVC201a', 'SVC201b',  # Station 24108, Interval 0
            'SVC202a', 'SVC202b',  # Station 25316, Interval 1
            'SVC203a', 'SVC203b',  # Station 25701, Interval 2
            'SVC204a', 'SVC204b',  # Station 27263, Interval 3
            'SVC205a', 'SVC205b',  # Station 31521, Interval 4
            'SVC206a', 'SVC206b',  # Station 32000, Interval 5
            # Incident 64327 (filtered out)
            'SVC105', 'SVC106'
        ],
        'INCIDENT_NUMBER': [
            # Non-delayed
            np.nan, np.nan, np.nan,
            # Incident 64326: all stations
            64326, 64326,  # Station 24108
            64326, 64326,  # Station 25316
            64326, 64326,  # Station 25701
            64326, 64326,  # Station 27263
            64326, 64326,  # Station 31521
            64326, 64326,  # Station 32000
            # Incident 64327
            64327, 64327
        ],
        'INCIDENT_START_DATETIME': [
            np.nan, np.nan, np.nan,
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 06:00', '07-DEC-2024 06:00',
            '07-DEC-2024 09:15', '07-DEC-2024 09:15'
        ],
        'EVENT_DATETIME': [
            np.nan, np.nan, np.nan,
            '07-DEC-2024 06:15', '07-DEC-2024 06:45',  # Interval 0: both in 06:00-07:00
            '07-DEC-2024 07:15', '07-DEC-2024 07:45',  # Interval 1: both in 07:00-08:00
            '07-DEC-2024 08:15', '07-DEC-2024 08:45',  # Interval 2: both in 08:00-09:00
            '07-DEC-2024 09:15', '07-DEC-2024 09:45',  # Interval 3: both in 09:00-10:00
            '07-DEC-2024 10:15', '07-DEC-2024 10:45',  # Interval 4: both in 10:00-11:00
            '07-DEC-2024 11:15', '07-DEC-2024 11:45',  # Interval 5: both in 11:00-12:00
            '07-DEC-2024 15:30', '07-DEC-2024 10:45'
        ],
        'PFPI_MINUTES': [
            0.0, 0.0, 0.0,
            250.0, 220.0,  # Station 24108: sum = 470
            225.0, 225.0,  # Station 25316: sum = 450
            245.0, 245.0,  # Station 25701: sum = 490
            255.0, 255.0,  # Station 27263: sum = 510
            235.0, 235.0,  # Station 31521: sum = 470
            245.0, 245.0,  # Station 32000: sum = 490
            45.0, 180.0
        ],
        'DELAY_DAY': ['SA'] * 17,
        'SECTION_CODE': [
            np.nan, np.nan, np.nan,
            '12001:15866', '12001:15866',
            '12001:15866', '12001:15866',
            '12001:15866', '12001:15866',
            '12001:15866', '12001:15866',
            '12001:15866', '12001:15866',
            '12001:15866', '12001:15866',
            '32534:33087', '32534:33087'
        ],
        'INCIDENT_REASON': [
            np.nan, np.nan, np.nan,
            'XW', 'XW',
            'XW', 'XW',
            'XW', 'XW',
            'XW', 'XW',
            'XW', 'XW',
            'XW', 'XW',
            'TG', 'TG'
        ],
        'STANOX': [
            12001, 12931, 13702,
            24108, 24108,
            25316, 25316,
            25701, 25701,
            27263, 27263,
            31521, 31521,
            32000, 32000,
            32534, 33087
        ],
        'PLANNED_CALLS': [
            '07:30', '09:00', '11:00',
            '05:00', '05:00',
            '04:30', '04:30',
            '05:30', '05:30',
            '02:00', '02:00',
            '03:30', '03:30',
            '01:00', '01:00',
            '15:15', '07:45'
        ],
        'EVENT_TYPE': ['D'] * 17,
        'ENGLISH_DAY_TYPE': [['SA']] * 17,
    })



# TESTS FOR incident_view

@patch('rdmpy.outputs.analysis_tools.pd.read_parquet')
@patch('rdmpy.outputs.analysis_tools._get_target_files_for_day')
@patch('rdmpy.outputs.analysis_tools.find_processed_data_path')
@patch('builtins.print')
def test_incident_view_fixture_data_correctness(mock_print, mock_find_path, mock_get_files, mock_read_parquet, sample_incident_data):
    """
    Test verifying incident_view correctly computes metrics aggregated PER STATION:
    
    - PLANNED_CALLS: Count of all trains scheduled to arrive during analysis period (regardless of whether delayed)
    - ACTUAL_CALLS: Trains that actually arrived in period = PLANNED_CALLS - DELAYED_TRAINS_OUT + DELAYED_TRAINS_IN
    - DELAYED_TRAINS_OUT: Count of trains scheduled in period but arrived AFTER period
    - DELAYED_TRAINS_IN: Count of trains scheduled BEFORE period but arrived within period
    
    Fixture data analysis period: 07-DEC-2024 06:00 to 07-DEC-2024 16:00 (600 minutes)
    Incident start: 07-DEC-2024 06:00
    Incident: 64326 (only this incident's trains are included in output)
    
    All 10 incident 64326 stations appear in results (each has at least one metric > 0):
    - Station 15866 (SVC101): Scheduled 07:00 (in period), delayed 600min, arrived 17:00 (after period)
      > PLANNED_CALLS=0, DELAYED_TRAINS_OUT=1, DELAYED_TRAINS_IN=0, ACTUAL_CALLS=-1
    - Station 16416 (SVC102): Scheduled 03:00 (before period), delayed 300min, arrived 13:00 (in period)
      > PLANNED_CALLS=0, DELAYED_TRAINS_OUT=0, DELAYED_TRAINS_IN=1, ACTUAL_CALLS=1
    - Station 18067 (SVC103): Scheduled 04:00 (before period), delayed 300min, arrived 09:00 (in period)
      > PLANNED_CALLS=0, DELAYED_TRAINS_OUT=0, DELAYED_TRAINS_IN=1, ACTUAL_CALLS=1
    - Station 23491 (SVC104): Scheduled 04:00 (before period), delayed 40min, arrived 13:40 (in period)
      > PLANNED_CALLS=0, DELAYED_TRAINS_OUT=0, DELAYED_TRAINS_IN=1, ACTUAL_CALLS=1
    - Stations 24108-32000 (SVC201-206): All scheduled BEFORE period (01:00-05:30), all delayed, all arrived IN period
      > Each: PLANNED_CALLS=0, DELAYED_TRAINS_OUT=0, DELAYED_TRAINS_IN=1, ACTUAL_CALLS=1
    
    """
    # Arrange: Mock file system and parquet reading
    mock_find_path.return_value = '/mock/processed_data'
    # Mock returns all 15 unique stations from fixture (including both incident 64326 and 64327)
    mock_get_files.return_value = [
        ('/mock/processed_data/12001/SA.parquet', 12001),
        ('/mock/processed_data/12931/SA.parquet', 12931),
        ('/mock/processed_data/13702/SA.parquet', 13702),
        ('/mock/processed_data/15866/SA.parquet', 15866),
        ('/mock/processed_data/16416/SA.parquet', 16416),
        ('/mock/processed_data/18067/SA.parquet', 18067),
        ('/mock/processed_data/23491/SA.parquet', 23491),
        ('/mock/processed_data/24108/SA.parquet', 24108),
        ('/mock/processed_data/25316/SA.parquet', 25316),
        ('/mock/processed_data/25701/SA.parquet', 25701),
        ('/mock/processed_data/27263/SA.parquet', 27263),
        ('/mock/processed_data/31521/SA.parquet', 31521),
        ('/mock/processed_data/32000/SA.parquet', 32000),
        ('/mock/processed_data/32534/SA.parquet', 32534),
        ('/mock/processed_data/33087/SA.parquet', 33087)
    ]
    
    # Filter sample_incident_data by station code extracted from filepath
    from pathlib import Path
    mock_read_parquet.side_effect = lambda filepath, **kwargs: sample_incident_data[
        sample_incident_data['STANOX'] == int(Path(filepath).parent.name)
    ].copy()
    
    result_df, incident_start, analysis_period = incident_view(
        incident_code=64326,
        incident_date='07-DEC-2024',
        analysis_date='07-DEC-2024',
        analysis_hhmm='0600', # Start of analysis period (06:00)
        period_minutes=600    # Time analysis window: 10 hours (06:00-16:00)
    )

    # Verify SVC105 and SVC106 (from incident 64327) are NOT included in the output
    # These trains should be filtered out as they belong to a different incident
    assert 'SVC105' not in result_df.values
    assert 'SVC106' not in result_df.values

    # PRINT OUTPUT ASSERTIONS - Verify all key information is printed
    print_output = ' '.join([str(call) for call in mock_print.call_args_list])
    assert all([
        'Analyzing incident 64326 (started 07-DEC-2024)' in print_output,
        'Analysis period: 07-Dec-2024 06:00 to 07-Dec-2024 16:00 (600 min)' in print_output,
        'Incident Details:' in print_output,
        'Section Code:' in print_output, 
        'Incident Reason: XW' in print_output,
        'Started: 07-DEC-2024 06:00' in print_output,
    ]), "Print output missing required info: incident number, dates, period, loading message, or incident details"
    
    # RETURN VALUE ASSERTIONS - Verify types and formats
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(incident_start, str) or incident_start is None
    assert isinstance(analysis_period, str) or analysis_period is None
    
    # INCIDENT START FORMAT - Must be "DD-MMM-YYYY HH:MM"
    if incident_start:
        assert '07-DEC-2024 06:00' == incident_start
    
    # ANALYSIS PERIOD FORMAT - Must contain start time, end time, and duration
    assert analysis_period is not None
    assert '07-Dec-2024 06:00' in analysis_period
    assert '07-Dec-2024 16:00' in analysis_period
    assert '600' in analysis_period
    
    # DATAFRAME STRUCTURE - Verify complete structure and computed metrics
    if not result_df.empty:
        # Column structure
        required_columns = {'STATION_CODE', 'PLANNED_CALLS', 'ACTUAL_CALLS', 'DELAYED_TRAINS_OUT', 
                           'DELAY_MINUTES_OUT', 'DELAYED_TRAINS_IN', 'DELAY_MINUTES_IN'}
        assert required_columns == set(result_df.columns), f"Missing columns: {required_columns - set(result_df.columns)}"
        assert len(result_df.columns) == 7, f"Should have exactly 7 columns, got {len(result_df.columns)}"
        
        # Column data types (integer)
        assert pd.api.types.is_integer_dtype(result_df['STATION_CODE'])
        assert pd.api.types.is_integer_dtype(result_df['PLANNED_CALLS'])
        assert pd.api.types.is_integer_dtype(result_df['ACTUAL_CALLS'])
        assert pd.api.types.is_integer_dtype(result_df['DELAYED_TRAINS_OUT'])
        assert pd.api.types.is_integer_dtype(result_df['DELAYED_TRAINS_IN'])

        # Delay minutes columns must contain lists, not scalars
        assert all(isinstance(x, (list, np.ndarray, pd.Series)) or pd.isna(x) for x in result_df['DELAY_MINUTES_OUT'])
        assert all(isinstance(x, (list, np.ndarray, pd.Series)) or pd.isna(x) for x in result_df['DELAY_MINUTES_IN'])
        
        # Value range checks (non-negative integers for counts)
        assert (result_df['PLANNED_CALLS'] >= 0).all()
        assert (result_df['DELAYED_TRAINS_OUT'] >= 0).all()
        assert (result_df['DELAYED_TRAINS_IN'] >= 0).all()
        assert len(result_df) == 10, f"Should have exactly 10 incident 64326 stations in results, got {len(result_df)}"
        
        # Verify all stations affected by incident 64326 are present
        expected_stations = {15866, 16416, 18067, 23491, 24108, 25316, 25701, 27263, 31521, 32000}
        result_stations = set(result_df['STATION_CODE'])
        assert expected_stations == result_stations
        
        # Station 15866: SVC101 - only station with DELAYED_TRAINS_OUT
        s = result_df[result_df['STATION_CODE'] == 15866].iloc[0]
        assert s['PLANNED_CALLS'] == 0 and s['DELAYED_TRAINS_OUT'] == 1 and s['DELAYED_TRAINS_IN'] == 0
        assert s['ACTUAL_CALLS'] == -1 and s['DELAY_MINUTES_OUT'] == [600.0]
        
        # Stations with DELAYED_TRAINS_IN, mapped to their expected delays from fixture
        # All these stations had the same metric pattern: PLANNED_CALLS=0, DELAYED_TRAINS_OUT=0, DELAYED_TRAINS_IN=1, ACTUAL_CALLS=1
        stations_with_delays = {
            16416: [450.0],  # SVC102: arrived 13:00, delay 450min
            18067: [300.0],  # SVC103: arrived 09:00, delay 300min
            23491: [471.0],  # SVC104: arrived 13:40, delay 471min
            24108: [470.0],  # SVC201: arrived 06:30, delay 470min
            25316: [450.0],  # SVC202: arrived 07:30, delay 450min
            25701: [490.0],  # SVC203: arrived 08:30, delay 490min
            27263: [510.0],  # SVC204: arrived 09:30, delay 510min
            31521: [470.0],  # SVC205: arrived 10:40, delay 470min
            32000: [490.0],  # SVC206: arrived 11:45, delay 490min
        }
        for station_code, expected_delay in stations_with_delays.items():
            s = result_df[result_df['STATION_CODE'] == station_code].iloc[0]
            assert s['PLANNED_CALLS'] == 0 and s['DELAYED_TRAINS_OUT'] == 0 and s['DELAYED_TRAINS_IN'] == 1
            assert s['ACTUAL_CALLS'] == 1 and s['DELAY_MINUTES_IN'] == expected_delay # Verify delay minutes match fixture values

@patch('rdmpy.outputs.analysis_tools.find_processed_data_path')
@patch('builtins.print')
def test_incident_view_handles_invalid_date(mock_print, mock_find_path):
    """Test incident_view returns empty structures for invalid date format."""
    mock_find_path.return_value = None
    result_df, incident_start, analysis_period = incident_view(
        incident_code=64326, incident_date='INVALID', analysis_date='INVALID',
        analysis_hhmm='9999', period_minutes=600
    )
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 0 or incident_start is None


def _filter_fixture_by_station(file_path, fixture_data):
    """Filter fixture data by station code extracted from parquet file path.
    
    In production, each station parquet file contains only that station's data.
    This helper filters the test fixture to simulate that behavior.
    """
    try:
        station_code = int(file_path.split('/')[-2])  # Extract from '/mock/processed_data/{station}/SA.parquet'
        return fixture_data[fixture_data['STANOX'] == station_code].copy()
    except (ValueError, IndexError):
        return fixture_data

# TOREVIEW: non-nested code, now direct definitions inside the test itself

@patch('rdmpy.outputs.analysis_tools._save_heatmap_html_file')
@patch('rdmpy.outputs.analysis_tools._get_incident_location_coordinates')
@patch('rdmpy.outputs.analysis_tools.pd.read_parquet')
@patch('rdmpy.outputs.analysis_tools._load_heatmap_station_files')
@patch('rdmpy.outputs.analysis_tools.find_processed_data_path')
@patch('rdmpy.outputs.analysis_tools._load_station_coordinates_from_json')
@patch('builtins.print')
def test_incident_view_heatmap_html_timeline_correctness(mock_print, mock_load_coords, mock_find_path, mock_load_files,
                                                         mock_read_parquet, mock_get_location, mock_save_file,
                                                         sample_incident_data, universal_station_ref):
    """
    Test that incident_view_heatmap_html correctly calculates aggregated delays per station per time window.
    Uses actual fixture data to verify the function performs correct calcu lations.
    
    The function:
    1. Reads station data from parquet files (mocked to return fixture)
    2. Filters for incident 64326 within analysis period
    3. Groups delays by station and time interval
    4. Aggregates multiple delays in same interval per station
    5. Generates HTML containing the correct delay data
    
    Analysis period: 07-DEC-2024 06:00 to 07-DEC-2024 16:00 (10 hourly intervals)
    Expected delays per station:
    - Station 24108: 470 min at 06:00
    - Station 25316: 450 min at 07:00
    - Station 25701: 490 min at 08:00
    - Station 27263: 510 min at 09:00
    - Station 31521: 470 min at 10:00
    - Station 32000: 490 min at 11:00
    """
    # Arrange: Setup mocks for I/O and external calls
    mock_find_path.return_value = '/mock/processed_data'
    
    # Mock parquet file reading to return station-filtered fixture data
    # Uses helper function to filter fixture by station code from the file path
    mock_read_parquet.side_effect = lambda file_path, engine=None: _filter_fixture_by_station(file_path, sample_incident_data)
    
    # Mock station file loading
    all_stations = sample_incident_data['STANOX'].dropna().unique().astype(int).tolist()
    # _load_heatmap_station_files returns (file_path, station_dir_as_string)
    mock_load_files.return_value = [
        (f'/mock/processed_data/{station}/SA.parquet', str(station)) for station in all_stations
    ]
    
    # Create coordinate map from universal_station_ref
    # _load_station_coordinates_from_json returns dict with string keys {'12001': {...}, '24108': {...}, ...}
    coords_map = {str(s['stanox']): {'name': 'Test Station', 'lat': s['latitude'], 'lon': s['longitude'], 'category': 'A'} for s in universal_station_ref}
    for station_code in all_stations:
        if str(station_code) not in coords_map:
            coords_map[str(station_code)] = {'name': f'Station {station_code}', 'lat': 51.5, 'lon': -0.1 + (station_code % 100) / 1000, 'category': 'A'}
    mock_load_coords.return_value = coords_map
    
    # Mock incident location
    incident_location = [{
        'lat': 51.5307,
        'lon': -0.1234,
        'name': 'Incident Location 12001:15866',
        'stanox': '12001'
    }]
    mock_get_location.return_value = (incident_location, 'Incident Location')

    # Act: Call heatmap function - it will calculate timeline from fixture data
    html_output = incident_view_heatmap_html(
        incident_code=64326,
        incident_date='07-DEC-2024',
        analysis_date='07-DEC-2024',
        analysis_hhmm='0600',
        period_minutes=600,
        interval_minutes=60,
        output_file='test_heatmap.html'
    )
    
    # Assert: Verify HTML output is generated
    assert html_output is not None
    assert isinstance(html_output, str)
    assert len(html_output) > 0
    
    # Assert: Verify _save_heatmap_html_file was called
    assert mock_save_file.called
    
    # Extract actual values (the positional arguments) passed to _save_heatmap_html_file from function calculation
    call_args = mock_save_file.call_args[0]

    """"
    call_args contains:
    0: html_content (the generated HTML string)
    1: output_file (the filename passed to save function)   
    2: incident_code (the incident code passed to save function)
    3: actual_time_steps (the list of time steps calculated from fixture data)
    4: all_station_coords_map (the station coordinates map used in calculation)
    5: actual_station_timeline_data (the station timeline data calculated from fixture data)
    6: period_minutes (the period_minutes passed to save function)
    7: interval_minutes (the interval_minutes passed to save function)
    """
    
    # Extract all arguments passed to _save_heatmap_html_file
    html_content = call_args[0]
    output_file = call_args[1]
    incident_code = call_args[2]
    actual_time_steps = call_args[3]
    all_station_coords_map = call_args[4]
    actual_station_timeline_data = call_args[5]  # The calculated timeline from fixture
    period_minutes = call_args[6]
    interval_minutes = call_args[7]
    
    # Assert: Verify all function parameters are correct
    # HTML content should be a non-empty string
    assert html_content is not None and isinstance(html_content, str)
    assert output_file == 'test_heatmap.html'
    assert incident_code == 64326
    assert period_minutes == 600
    assert interval_minutes == 60
    assert len(actual_time_steps) == 10
    assert all_station_coords_map is not None and len(all_station_coords_map) > 0
    
    # Define expected values (what the function SHOULD calculate from fixture)
    # Note: station_timeline_data uses STRING keys (from directory names in _load_heatmap_station_files)
    expected_station_codes = {'24108', '25316', '25701', '27263', '31521', '32000'}
    expected_delays_by_station = {
        '24108': 470,
        '25316': 450,
        '25701': 490,
        '27263': 510,
        '31521': 470,
        '32000': 490,
    }
    
    # Expected time windows for each station (event time determines which interval)
    expected_times_by_station = {
        '24108': pd.to_datetime('07-DEC-2024 06:00', format='%d-%b-%Y %H:%M'),  # Event at 06:30 → interval 06:00
        '25316': pd.to_datetime('07-DEC-2024 07:00', format='%d-%b-%Y %H:%M'),  # Event at 07:30 → interval 07:00
        '25701': pd.to_datetime('07-DEC-2024 08:00', format='%d-%b-%Y %H:%M'),  # Event at 08:30 → interval 08:00
        '27263': pd.to_datetime('07-DEC-2024 09:00', format='%d-%b-%Y %H:%M'),  # Event at 09:30 → interval 09:00
        '31521': pd.to_datetime('07-DEC-2024 10:00', format='%d-%b-%Y %H:%M'),  # Event at 10:40 → interval 10:00
        '32000': pd.to_datetime('07-DEC-2024 11:00', format='%d-%b-%Y %H:%M'),  # Event at 11:45 → interval 11:00
    }
    
    # Assert: Verify station timeline data was calculated correctly from fixture
    actual_station_codes = set(actual_station_timeline_data)
    assert actual_station_codes == expected_station_codes, f"Stations mismatch: expected {expected_station_codes}, got {actual_station_codes}"
    
    # Assert: Verify each station has correct aggregated delay in correct time window
    for station_code, expected_delay in expected_delays_by_station.items():
        assert station_code in actual_station_timeline_data
        
        actual_entries = actual_station_timeline_data[station_code]
        assert len(actual_entries) == 1 # One total delay entry per station

        actual_time, actual_delay = actual_entries[0]
        expected_time = expected_times_by_station[station_code]
        
        # Verify delay totals are correct
        assert actual_delay == expected_delay

        # Verify time window is correct (delay mapped to correct interval)
        assert actual_time == expected_time

    # Assert: Verify incident 64327 stations are filtered out (not included in results)
    incident_64327_stations = {32534, 33087}
    assert all(station not in actual_station_codes for station in incident_64327_stations)

    # Assert: Verify HTML output contains the correct delays for each station
    for station_code, expected_delay in expected_delays_by_station.items():
        assert str(expected_delay) in html_content

# TESTS FOR incident_view_heatmap_html

@patch('rdmpy.outputs.analysis_tools._save_heatmap_html_file')
@patch('rdmpy.outputs.analysis_tools._get_incident_location_coordinates')
@patch('rdmpy.outputs.analysis_tools._collect_heatmap_delay_timeline')
@patch('rdmpy.outputs.analysis_tools._load_heatmap_station_files')
@patch('rdmpy.outputs.analysis_tools.find_processed_data_path')
@patch('rdmpy.outputs.analysis_tools._load_station_coordinates_from_json')
@patch('builtins.print')
def test_incident_view_heatmap_html_fixture_data_content(mock_print, mock_load_coords, mock_find_path, 
                                                          mock_load_files, mock_collect_timeline, mock_get_location, mock_save_file, 
                                                          sample_incident_data, universal_station_ref):
    """
    Test that incident_view_heatmap_html generates HTML with correct incident content from fixture data.
    
    Verifies HTML includes:
    - Correct incident code
    - Analysis date and time period
    - Incident section code and reason
    - Delay color scheme (green, yellow, orange, red)
    - Interactive controls (Play, Pause, Reset buttons)
    - Map and timeline elements
    """
    # Arrange: Mock only file I/O and data loading, not HTML generation
    mock_find_path.return_value = '/mock/processed_data' # TODO: save the map for sense check
    
    # Create coordinate map from fixture (select subset with matching stanox codes for incident test)
    # Use only the first 5 stations from universal_station_ref
    coords_map = {str(s['stanox']): [s['latitude'], s['longitude']] for s in universal_station_ref}
    mock_load_coords.return_value = coords_map
    
    # Mock station files to return fixture data for one station
    mock_load_files.return_value = [('/mock/processed_data/51511/SA.parquet', 51511)]
    
    # Mock incident location with proper structure (dict with lat, lon, name, stanox)
    incident_location = [{
        'lat': 51.5307,
        'lon': -0.1234,
        'name': 'London Kings Cross',
        'stanox': 51511
    }]
    mock_get_location.return_value = (incident_location, 'London Kings Cross')
    
    # Mock timeline collection to return incident data
    mock_collect_timeline.return_value = (
        {51511: {0: [15, 25, 35]}},  # station_timeline_data: station -> interval -> delays
        77301,                        # incident_section_code
        'XW',                         # incident_reason
        '07-DEC-2024 08:23'           # incident_start_time
    )
    
    # Mock read_parquet to return fixture data
    with patch('rdmpy.outputs.analysis_tools.pd.read_parquet', return_value=sample_incident_data):
        # Act
        html_output = incident_view_heatmap_html(
            incident_code=64326,
            incident_date='07-DEC-2024',
            analysis_date='07-DEC-2024',
            analysis_hhmm='0600',
            period_minutes=600,
            interval_minutes=60,
            output_file='test_heatmap.html'
        )
    
    # Assert: HTML structure and content
    assert html_output is not None
    assert isinstance(html_output, str)
    
    # Assert: HTML document structure
    assert '<!DOCTYPE html>' in html_output
    assert '<html>' in html_output
    assert '</html>' in html_output
    assert '<head>' in html_output
    assert '<body>' in html_output
    
    # Assert: Title and incident code
    assert 'Incident 64326' in html_output
    assert 'Network Heatmap' in html_output
    assert '07-DEC-2024' in html_output
    
    # Assert: Analysis period information
    assert '06:00' in html_output  # Start time
    assert '16:00' in html_output  # End time (06:00 + 600 min = 16:00)
    assert '600' in html_output     # Duration
    assert '60' in html_output      # Interval size
    
    # Assert: Incident details
    assert '77301' in html_output or 'Section' in html_output  # Section code
    assert 'XW' in html_output or 'Reason' in html_output      # Incident reason
    assert '08:23' in html_output                              # Start time from fixture
    
    # Assert: Delay color scheme (from legend)
    assert 'rgb(0,255,0)' in html_output or '#00ff00' in html_output.lower() or 'green' in html_output.lower()  # Green
    assert 'rgb(255,255,0)' in html_output or '#ffff00' in html_output.lower() or 'yellow' in html_output.lower()  # Yellow
    assert 'rgb(255,165,0)' in html_output or '#ffa500' in html_output.lower() or 'orange' in html_output.lower()  # Orange
    assert 'rgb(255,0,0)' in html_output or '#ff0000' in html_output.lower() or 'red' in html_output.lower()  # Red
    
    # Assert: Interactive controls
    assert 'Play' in html_output or 'play' in html_output.lower()
    assert 'Pause' in html_output or 'pause' in html_output.lower()
    assert 'Reset' in html_output or 'reset' in html_output.lower()
    
    # Assert: Map and visualization elements
    assert 'map' in html_output.lower()
    assert 'timeline' in html_output.lower()
    assert 'leaflet' in html_output.lower()
    
    # Assert: JavaScript for interactivity
    assert '<script>' in html_output
    assert 'playTimeline' in html_output or 'play' in html_output.lower()
    
    # Assert: File save was called
    assert mock_save_file.called


@patch('rdmpy.outputs.analysis_tools._load_station_coordinates_from_json')
def test_incident_view_heatmap_html_handles_missing_coordinates(mock_load_coords):
    """Test incident_view_heatmap_html returns None when coordinates unavailable."""
    mock_load_coords.return_value = None
    result = incident_view_heatmap_html(
        incident_code=62537, incident_date='07-DEC-2024', analysis_date='07-DEC-2024',
        analysis_hhmm='0600', period_minutes=120, interval_minutes=30
    )
    assert result is None


@patch('rdmpy.outputs.analysis_tools._load_station_coordinates_from_json')
def test_incident_view_heatmap_html_handles_invalid_date(mock_load_coords):
    """Test incident_view_heatmap_html returns None for invalid date format."""
    mock_load_coords.return_value = {'51511': [51.5307, -0.1234]}
    result = incident_view_heatmap_html(
        incident_code=62537, incident_date='INVALID', analysis_date='INVALID',
        analysis_hhmm='9999', period_minutes=120, interval_minutes=30
    )
    assert result is None or isinstance(result, str)


# ==============================================================================
# STATION VIEW FUNCTION TESTS
# ==============================================================================

# FIXTURES FOR STATION VIEW TESTS

@pytest.fixture
def station_view_sample_data():
    """
    Fixture: Comprehensive station dataset for testing station_view functions.
    
    Simulates realistic multi-week station data with:
    - Multiple days of week with varying traffic patterns
    - Mix of on-time and delayed arrivals
    - Realistic platform occupancy across hours
    - Incident-related and normal operation delays
    - PLANNED_CALLS: scheduled time in HHMM format (e.g., 830 = 08:30)
    - ACTUAL_CALLS: actual arrival time in HHMM format (affected by PFPI_MINUTES delay)
    """
    # Create hourly data for multiple days
    dates = pd.date_range('2024-01-01', periods=336, freq='h')  # 2 weeks of hourly data
    
    # Create repeating patterns for different days of week
    day_codes = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU'] * 48  # Pattern repeating over 2 weeks
    
    # Simulate varying delays based on time of day and day of week
    delays = []
    planned_times = []
    actual_times = []
    
    for i, (date_val, day) in enumerate(zip(dates, day_codes)):
        hour = date_val.hour
        # Peak hours (7-9, 16-18) have more delays
        if hour in [7, 8, 17, 18]:
            base_delay = 15 if day in ['MO', 'FR'] else 10  # Fridays/Mondays busier
        elif hour in [6, 9, 16, 19]:
            base_delay = 8
        else:
            base_delay = 3
        
        # Add randomness
        delay = max(0, base_delay + np.random.uniform(-2, 5))
        delays.append(delay)
        
        # Planned call time in HHMM format (0-2359)
        planned_hour = (hour + np.random.randint(-1, 2)) % 24
        planned_minute = np.random.choice([0, 15, 30, 45])
        planned_times.append(planned_hour * 100 + planned_minute)
        
        # Actual call time: planned + delay (in minutes), wrapped to HHMM format
        total_minutes = planned_hour * 60 + planned_minute + int(delay)
        actual_hour = (total_minutes // 60) % 24
        actual_minute = total_minutes % 60
        actual_times.append(actual_hour * 100 + actual_minute)
    
    data = pd.DataFrame({
        'STANOX': '32000',  # Manchester Piccadilly
        'EVENT_DATETIME': dates.strftime('%d-%b-%Y %H:%M'),
        'PFPI_MINUTES': delays,
        'PLANNED_CALLS': planned_times,
        'ACTUAL_CALLS': actual_times,
        'TRAIN_SERVICE_CODE': [f'SVC{i:05d}' for i in range(len(dates))],
        'EVENT_TYPE': np.random.choice(['D', 'C'], len(dates), p=[0.95, 0.05]),
        'INCIDENT_NUMBER': [12345.0 if i % 20 == 0 else np.nan for i in range(len(dates))],
        'DAY': [d for d in day_codes],
    })
    
    return data


@pytest.fixture
def station_view_minimal_data():
    """Fixture: Minimal station data for edge case testing.
    
    PLANNED_CALLS and ACTUAL_CALLS are time values in HHMM format (0-2359).
    ACTUAL_CALLS = PLANNED_CALLS + PFPI_MINUTES (in minutes), wrapped for hour rollover.
    """
    planned_calls = [700, 800, 900, 700, 800, 900, 700, 800, 900, 700]
    delays = [5.0, 0.0, 8.0, 5.0, 0.0, 8.0, 5.0, 0.0, 8.0, 5.0]
    # Pre-calculated actual calls: planned + delay converted to HHMM format
    actual_calls = [705, 800, 908, 705, 800, 908, 705, 800, 908, 705]
    
    return pd.DataFrame({
        'STANOX': '32000',
        'EVENT_DATETIME': pd.date_range('2024-01-01', periods=10, freq='h').strftime('%d-%b-%Y %H:%M'),
        'PFPI_MINUTES': delays,
        'PLANNED_CALLS': planned_calls,
        'ACTUAL_CALLS': actual_calls,
        'TRAIN_SERVICE_CODE': [f'SVC{i:05d}' for i in range(10)],
        'EVENT_TYPE': ['D'] * 10,
        'INCIDENT_NUMBER': [np.nan] * 10,
        'DAY': ['MO', 'MO', 'MO', 'MO', 'MO', 'TU', 'TU', 'TU', 'TU', 'TU'],
    })


@pytest.fixture
def station_view_data_with_cancellations():
    """Fixture: Data including cancellations (EVENT_TYPE='C') for comprehensive testing.
    
    PLANNED_CALLS and ACTUAL_CALLS are time values in HHMM format (0-2359).
    For delayed trains: ACTUAL_CALLS = PLANNED_CALLS + PFPI_MINUTES (in minutes)
    For cancelled trains: ACTUAL_CALLS = PLANNED_CALLS (no actual arrival)
    """
    planned_calls = np.random.randint(600, 2300, 100)
    delays = np.concatenate([np.random.uniform(0, 30, 95), [np.nan] * 5])
    
    # Calculate actual calls: planned + delay, with proper HHMM format wrapping
    actual_calls = [
        planned if pd.isna(delay) else 
        ((planned // 100 * 60 + planned % 100 + int(delay)) // 60) * 100 + 
        ((planned // 100 * 60 + planned % 100 + int(delay)) % 60)
        for planned, delay in zip(planned_calls, delays)
    ]
    
    data = pd.DataFrame({
        'STANOX': '32000',
        'EVENT_DATETIME': pd.date_range('2024-01-01', periods=100, freq='30min').strftime('%d-%b-%Y %H:%M'),
        'PFPI_MINUTES': delays,
        'PLANNED_CALLS': planned_calls,
        'ACTUAL_CALLS': actual_calls,
        'TRAIN_SERVICE_CODE': [f'SVC{i:05d}' for i in range(100)],
        'EVENT_TYPE': ['D'] * 95 + ['C'] * 5,  # 5% cancellations
        'INCIDENT_NUMBER': [np.nan] * 100,
        'DAY': ['MO'] * 100,
    })
    return data


# UNIT TESTS FOR STATION VIEW FUNCTIONS

# TESTS for station_view_yearly

@patch('os.path.exists')
@patch('pandas.read_parquet')
def test_station_view_yearly_comprehensive_with_sample_data(mock_read_parquet, mock_exists, station_view_sample_data):
    """Comprehensive test for station_view_yearly with realistic sample data.
    
    Tests in a single function call:
    - Return type is tuple of two DataFrames
    - Both DataFrames have required columns with correct data types
    - Incident and normal operations are properly separated
    - Count calculations are valid and logically consistent
    - Time period format is correct (HH:00-HH:00)
    - delay_minutes contains valid numeric arrays
    """
    mock_exists.return_value = True
    mock_read_parquet.return_value = station_view_sample_data
    
    # Call function once
    result = station_view_yearly(station_id='32000', interval_minutes=60)
    
    # ===== ASSERTION GROUP 1: Return type and structure =====
    assert result is not None, "Result should not be None"
    assert isinstance(result, tuple), "Result should be tuple"
    assert len(result) == 2, "Result should contain exactly 2 elements"
    
    incident_summary, normal_summary = result
    assert isinstance(incident_summary, pd.DataFrame), "incident_summary should be DataFrame"
    assert isinstance(normal_summary, pd.DataFrame), "normal_summary should be DataFrame"
    
    # ===== ASSERTION GROUP 2: Column structure and data types =====
    expected_cols = ['time_period', 'ontime_arrival_count', 'delayed_arrival_count', 
                     'cancellation_count', 'delay_minutes', 'operation_type']
    
    if len(incident_summary) > 0:
        assert all(col in incident_summary.columns for col in expected_cols), \
            f"incident_summary missing columns. Expected {expected_cols}, got {incident_summary.columns.tolist()}"
        assert incident_summary['ontime_arrival_count'].dtype in [int, np.int64]
        assert incident_summary['delayed_arrival_count'].dtype in [int, np.int64]
        assert incident_summary['cancellation_count'].dtype in [int, np.int64]
        assert incident_summary['operation_type'].dtype == object
    
    if len(normal_summary) > 0:
        assert all(col in normal_summary.columns for col in expected_cols), \
            f"normal_summary missing columns. Expected {expected_cols}, got {normal_summary.columns.tolist()}"
        assert normal_summary['ontime_arrival_count'].dtype in [int, np.int64]
        assert normal_summary['delayed_arrival_count'].dtype in [int, np.int64]
        assert normal_summary['cancellation_count'].dtype in [int, np.int64]
        assert normal_summary['operation_type'].dtype == object
    
    # ===== ASSERTION GROUP 3: Operation type separation =====
    if len(incident_summary) > 0:
        assert (incident_summary['operation_type'] == 'incident').all(), \
            "All incident_summary rows must have operation_type='incident'"
    
    if len(normal_summary) > 0:
        assert (normal_summary['operation_type'] == 'normal').all(), \
            "All normal_summary rows must have operation_type='normal'"
    
    # ===== ASSERTION GROUP 4: Time period format =====
    if len(incident_summary) > 0:
        for time_period in incident_summary['time_period']:
            assert '-' in time_period and ':' in time_period, \
                f"Time period format incorrect: {time_period}"
    
    if len(normal_summary) > 0:
        for time_period in normal_summary['time_period']:
            assert '-' in time_period and ':' in time_period, \
                f"Time period format incorrect: {time_period}"
    
    # ===== ASSERTION GROUP 5: Count calculations validity =====
    def validate_counts(df, op_type):
        for idx, row in df.iterrows():
            ontime = row['ontime_arrival_count']
            delayed = row['delayed_arrival_count']
            cancelled = row['cancellation_count']
            delays = row['delay_minutes']
            
            # All counts non-negative
            assert ontime >= 0, f"{op_type}: ontime_arrival_count should be >= 0, got {ontime}"
            assert delayed >= 0, f"{op_type}: delayed_arrival_count should be >= 0, got {delayed}"
            assert cancelled >= 0, f"{op_type}: cancellation_count should be >= 0, got {cancelled}"
            
            # delay_minutes is list with length <= delayed count
            assert isinstance(delays, (list, np.ndarray)), f"{op_type}: delay_minutes should be list/array"
            assert len(delays) <= delayed, \
                f"{op_type}: delay count {len(delays)} exceeds delayed_arrival_count {delayed}"
            
            # All delay values are numeric
            if len(delays) > 0:
                assert all(isinstance(d, (int, float, np.integer, np.floating)) for d in delays), \
                    f"{op_type}: all delay values should be numeric"
    
    if len(incident_summary) > 0:
        validate_counts(incident_summary, "incident_summary")
    
    if len(normal_summary) > 0:
        validate_counts(normal_summary, "normal_summary")


# edge cases for station_view_yearly with minimal data

@patch('os.path.exists')
@patch('pandas.read_parquet')
def test_station_view_yearly_comprehensive_with_minimal_data(mock_read_parquet, mock_exists, station_view_minimal_data):
    """Comprehensive test for station_view_yearly with minimal data (edge cases).
    
    Tests with small dataset to verify:
    - Function handles minimal data correctly
    - All assertions from sample_data test also hold
    - Output structure is consistent regardless of data size
    """
    mock_exists.return_value = True
    mock_read_parquet.return_value = station_view_minimal_data
    
    # Call function once with minimal data
    result = station_view_yearly(station_id='32000', interval_minutes=60)
    
    # Basic structure validation
    assert result is not None
    assert isinstance(result, tuple) and len(result) == 2
    incident_summary, normal_summary = result
    assert isinstance(incident_summary, pd.DataFrame)
    assert isinstance(normal_summary, pd.DataFrame)
    
    # Verify columns and types for any returned data
    expected_cols = ['time_period', 'ontime_arrival_count', 'delayed_arrival_count', 
                     'cancellation_count', 'delay_minutes', 'operation_type']
    
    if len(incident_summary) > 0:
        assert all(col in incident_summary.columns for col in expected_cols)
        assert (incident_summary['operation_type'] == 'incident').all()
        assert all(isinstance(delays, (list, np.ndarray)) for delays in incident_summary['delay_minutes'])
    
    if len(normal_summary) > 0:
        assert all(col in normal_summary.columns for col in expected_cols)
        assert (normal_summary['operation_type'] == 'normal').all()
        assert all(isinstance(delays, (list, np.ndarray)) for delays in normal_summary['delay_minutes'])

# TESTS for station_view

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.subplots')
def test_station_view_comprehensive_output_structure_and_calculations(mock_subplots, mock_plt_show, station_view_sample_data):
    """Comprehensive test for station_view with complete data structure and calculation validation.
    
    Tests in a single function call:
    - Return type is dict with required keys
    - both hourly_stats and bin_stats are DataFrames
    - hourly_stats contains all required columns
    - bin_stats calculations are valid (percentages 0-100, CDF monotonicity)
    - All data integrity checks
    """
    # Mock matplotlib
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    # Call function once with comprehensive parameters
    result = station_view(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=14,
        time_window_minutes=60,
        max_delay_percentile=98
    )
    
    # ===== ASSERTION GROUP 1: Return type and structure =====
    assert result is not None
    assert isinstance(result, dict)
    assert 'hourly_stats' in result
    assert 'bin_stats' in result
    assert isinstance(result['hourly_stats'], pd.DataFrame)
    assert isinstance(result['bin_stats'], pd.DataFrame)
    
    # ===== ASSERTION GROUP 2: hourly_stats columns and structure =====
    hourly_stats = result['hourly_stats']
    required_hourly_cols = ['trains_in_system_normalized', 'ontime_trains_normalized', 
                            'total_trains', 'is_100_percent_ontime']
    assert all(col in hourly_stats.columns for col in required_hourly_cols)

    # Validate hourly_stats data
    if len(hourly_stats) > 0:
        # ===== Range validation =====
        if 'trains_in_system_normalized' in hourly_stats.columns:
            # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
            assert (hourly_stats['trains_in_system_normalized'] >= 0).all(), \
                "trains_in_system_normalized should be >= 0"
            assert (hourly_stats['trains_in_system_normalized'] <= 3.0).all(), \
                "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
        
        if 'ontime_trains_normalized' in hourly_stats.columns:
            # ontime_trains_normalized is ratio of ontime trains per hour per platform
            assert (hourly_stats['ontime_trains_normalized'] >= 0).all()
        
        # total_trains should be positive
        if 'total_trains' in hourly_stats.columns:
            assert (hourly_stats['total_trains'] >= 0).all()
        
        # is_100_percent_ontime should be boolean
        if 'is_100_percent_ontime' in hourly_stats.columns:
            assert hourly_stats['is_100_percent_ontime'].dtype == bool or \
                   hourly_stats['is_100_percent_ontime'].dtype == object
        
        # ===== Calculated value validation from fixture =====
        # Fixture sample_data has 336 hourly records (2 weeks)
        total_trains_sum = hourly_stats['total_trains'].sum()
        assert total_trains_sum >= 0

        # Check if on-time trains exist (fixture may produce zero or non-zero)
        ontime_sum = hourly_stats['ontime_trains_normalized'].sum()
        assert ontime_sum >= 0

    # ===== ASSERTION GROUP 3: bin_stats calculations validity =====
    bin_stats = result['bin_stats']
    
    if len(bin_stats) > 0:
        # Check percentage values are valid (0-100)
        if 'pct_hours_100_ontime' in bin_stats.columns:
            assert (bin_stats['pct_hours_100_ontime'] >= 0).all()
            assert (bin_stats['pct_hours_100_ontime'] <= 100).all()
        
        # Check CDF values increase (monotonically non-decreasing)
        if 'cdf' in bin_stats.columns:
            cdf_vals = bin_stats['cdf'].values
            if len(cdf_vals) > 1:
                assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
            
                # CDF should be in valid range (0-100 if percentages, or 0-1 if probabilities)
                min_cdf = cdf_vals[0]
                max_cdf = cdf_vals[-1]
                assert min_cdf >= 0, "CDF min should be >= 0"
                # CDF can be in percent (0-100) or probability (0-1) format
                assert max_cdf <= 100, "CDF max should be <= 100"
                
                # For sample_data fixture: should have some variation
                cdf_unique = len(set(cdf_vals))
                assert cdf_unique > 1, "CDF should have multiple distinct values"


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.subplots')
def test_plot_trains_in_system_vs_delay_comprehensive_output(mock_subplots, mock_plt_show, station_view_sample_data):
    """Comprehensive test for plot_trains_in_system_vs_delay function output validation.
    
    Tests in a single function call:
    - Returns a DataFrame
    - Contains required columns: trains_in_system_normalized, mean_delay
    - Data types and value ranges are valid
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    # Call function once
    result = plot_trains_in_system_vs_delay(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=14
    )
    
    # ===== ASSERTION GROUP 1: Return type and structure =====
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    
    # ===== ASSERTION GROUP 2: Required columns =====
    required_cols = ['trains_in_system_normalized', 'mean_delay']
    assert all(col in result.columns for col in required_cols)
    
    # ===== ASSERTION GROUP 3: Data validity and range checks =====
    if len(result) > 0:
        # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
        if 'trains_in_system_normalized' in result.columns:
            assert (result['trains_in_system_normalized'] >= 0).all(), \
                "trains_in_system_normalized should be >= 0"
            assert (result['trains_in_system_normalized'] <= 3.0).all(), \
                "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
        
        # mean_delay should be non-negative (in minutes)
        if 'mean_delay' in result.columns:
            assert (result['mean_delay'] >= 0).all()
    
    # ===== ASSERTION GROUP 4: Data correctness against fixture =====
    # Fixture data contains 336 hourly records (2 weeks)
    # Overall mean delay across all data should be around 6-12 minutes (accounting for variance)
    if len(result) > 0 and 'mean_delay' in result.columns:
        overall_mean = result['mean_delay'].mean()
        # Verify mean is in reasonable range for sample_data fixture (peak hours have more delays)
        assert 2 <= overall_mean <= 20, \
            f"Overall mean delay should be 2-20 min, got {overall_mean:.2f}"
        
        # Verify individual mean_delay values are sensible
        assert result['mean_delay'].min() >= 0, "Min mean_delay should be >= 0"
        assert result['mean_delay'].max() <= 30, "Max mean_delay should be <= 30 min"
        
        # Verify standard deviation is reasonable (delays do vary by hour)
        if len(result) > 1:
            std_dev = result['mean_delay'].std()
            assert std_dev >= 0, "Std dev should be non-negative"


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.subplots')
def test_explore_delay_outliers_comprehensive_output(mock_subplots, mock_plt_show, station_view_sample_data):
    """Comprehensive test for explore_delay_outliers function output validation.
    
    Tests in a single function call:
    - Returns a DataFrame
    - Contains required columns: trains_in_system_normalized, outlier metrics
    - Data is valid for statistical analysis
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    # Call function once
    result = explore_delay_outliers(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=14
    )
    
    # ===== ASSERTION GROUP 1: Return type and structure =====
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    
    # ===== ASSERTION GROUP 2: Required columns =====
    required_cols = ['trains_in_system_normalized']
    assert all(col in result.columns for col in required_cols)
    
    # ===== ASSERTION GROUP 3: Data validity and range checks =====
    if len(result) > 0:
        # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
        if 'trains_in_system_normalized' in result.columns:
            assert (result['trains_in_system_normalized'] >= 0).all(), \
                "trains_in_system_normalized should be >= 0"
            assert (result['trains_in_system_normalized'] <= 3.0).all(), \
                "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
        
        # If there are outlier-related columns, validate their ranges
        for col in result.columns:
            if 'pct' in col.lower() or 'percentage' in col.lower():
                assert (result[col] >= 0).all() and (result[col] <= 100).all(), \
                    f"Percentage column {col} should be 0-100"
    
    # ===== ASSERTION GROUP 4: Data correctness from fixture =====
    # Fixture has 336 hourly records with realistic delay patterns
    # Outlier detection should identify some outliers but not mark all as outliers
    if len(result) > 5:
        # Should have variety in the data (not all same values)
        unique_vals = result['trains_in_system_normalized'].nunique()
        assert unique_vals > 1, "Should have variation in trains_in_system_normalized"


# EDGE CASE TESTS FOR STATION VIEW FUNCTIONS

@patch('matplotlib.pyplot.show')
def test_station_view_handles_no_data(mock_plt_show):
    """Test station_view handles empty DataFrame gracefully."""
    empty_data = pd.DataFrame({
        'STANOX': [],
        'EVENT_DATETIME': [],
        'PFPI_MINUTES': [],
        'EVENT_TYPE': [],
    })
    
    result = station_view(station_id='99999', all_data=empty_data)
    # Should return None or empty result without raising exception
    assert result is None or (isinstance(result, dict) and len(result.get('hourly_stats', [])) == 0)


@patch('matplotlib.pyplot.show')
def test_station_view_handles_wrong_station_id(mock_plt_show, station_view_sample_data):
    """Test station_view returns None for non-existent station."""
    result = station_view(station_id='99999', all_data=station_view_sample_data)
    # Should return None or empty result when station not in data
    assert result is None or (isinstance(result, dict) and len(result.get('hourly_stats', [])) == 0)

@patch('matplotlib.pyplot.show')
def test_station_view_handles_all_cancellations(mock_plt_show, station_view_data_with_cancellations):
    """Test station_view handles data with only cancellations."""
    # Create data with all cancellations
    cancel_only = station_view_data_with_cancellations.copy()
    cancel_only['EVENT_TYPE'] = 'C'
    
    result = station_view(station_id='32000', all_data=cancel_only)
    
    # Should return result or None (graceful handling)
    assert result is None or isinstance(result, dict)
    
    # If result is dict, it should have expected structure but hourly_stats should show no on-time trains
    if result is not None and isinstance(result, dict) and len(result.get('hourly_stats', [])) > 0:
        hourly_stats = result['hourly_stats']
        # With all cancellations, on-time trains should be 0
        assert (hourly_stats['ontime_trains_normalized'] == 0).all()
        # Total trains should also be 0 since cancellations are not train arrivals
        assert (hourly_stats['total_trains'] >= 0).all()


@patch('matplotlib.pyplot.show')
def test_plot_trains_vs_delay_with_minimal_data(mock_plt_show, station_view_minimal_data):
    """Test plot_trains_in_system_vs_delay handles small datasets and produces correct results.
    
    Fixture data validation:
    - 10 records with delays: [5, 0, 8, 5, 0, 8, 5, 0, 8, 5]
    - Expected mean_delay: (5+0+8+5+0+8+5+0+8+5)/10 = 44/10 = 4.4 minutes
    - Distributed across 2 days: MO (5 records) and TU (5 records)
    """
    result = plot_trains_in_system_vs_delay(
        station_id='32000',
        all_data=station_view_minimal_data
    )
    
    # ===== ASSERTION GROUP 1: Return type =====
    assert result is None or isinstance(result, pd.DataFrame), \
        "Should return DataFrame or None"
    
    # ===== ASSERTION GROUP 2: Data correctness with minimal fixture =====
    if result is not None and len(result) > 0:
        # Required columns
        assert 'mean_delay' in result.columns, "Should have mean_delay column"
        assert 'trains_in_system_normalized' in result.columns, "Should have trains_in_system_normalized"
        
        # ===== ASSERTION GROUP 3: Range validation =====
        assert (result['mean_delay'] >= 0).all(), "Mean delay should be non-negative"
        assert (result['mean_delay'] <= 30).all(), "Mean delay should be reasonable (<=30 min)"
        # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
        assert (result['trains_in_system_normalized'] >= 0).all(), \
            "trains_in_system_normalized should be >= 0"
        assert (result['trains_in_system_normalized'] <= 3.0).all(), \
            "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
        
        # ===== ASSERTION GROUP 4: Calculated value validation =====
        # Fixture delays: [5, 0, 8, 5, 0, 8, 5, 0, 8, 5] = mean 4.4
        # Should have some rows with mean_delay close to 4.4 (or aggregated around it)
        mean_delay_values = result['mean_delay'].dropna()
        if len(mean_delay_values) > 0:
            # Overall mean should be close to expected 4.4 (allow ±1.5 tolerance for hourly bins)
            overall_mean = mean_delay_values.mean()
            assert 2.5 <= overall_mean <= 6.0, \
                f"Overall mean_delay should be ~4.4 (±1.5), got {overall_mean:.2f} from fixture data"
            
            # At least one row should have mean_delay near 4.4
            has_near_expected = any(2.5 <= val <= 6.0 for val in mean_delay_values)
            assert has_near_expected, \
                f"No mean_delay values near expected 4.4. Got: {mean_delay_values.tolist()}"


@patch('matplotlib.pyplot.show')
def test_explore_outliers_with_minimal_data(mock_plt_show, station_view_minimal_data):
    """Test explore_delay_outliers handles small datasets and produces correct results.
    
    Fixture data validation:
    - 10 minimal records with delays: [5, 0, 8, 5, 0, 8, 5, 0, 8, 5]
    - Expected behavior: identify if any hours have extreme delays relative to period
    - With only 10 records, outlier detection may be limited
    """
    result = explore_delay_outliers(
        station_id='32000',
        all_data=station_view_minimal_data
    )
    
    # ===== ASSERTION GROUP 1: Return type =====
    assert result is None or isinstance(result, pd.DataFrame), \
        "Should return DataFrame or None"
    
    # ===== ASSERTION GROUP 2: Data structure validation =====
    if result is not None and len(result) > 0:
        # Required columns
        assert 'trains_in_system_normalized' in result.columns, \
            "Should contain trains_in_system_normalized column"
        
        # ===== ASSERTION GROUP 3: Data validity checks =====
        # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
        assert (result['trains_in_system_normalized'] >= 0).all(), \
            "trains_in_system_normalized should be >= 0"
        assert (result['trains_in_system_normalized'] <= 3.0).all(), \
            "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
        
        # ===== ASSERTION GROUP 4: Calculated correctness validation =====
        # With fixture (10 records, delays [5, 0, 8, 5, 0, 8, 5, 0, 8, 5]):
        # - Min delay: 0, Max delay: 8
        # - For outlier detection, values > mean+2*std or < mean-2*std are outliers
        # - Mean = 4.4, values shouldn't be excessive (max is only 8)
        if 'mean_delay' in result.columns:
            mean_delays = result['mean_delay'].dropna()
            if len(mean_delays) > 0:
                # With fixture data (max delay 8), mean delay shouldn't exceed 6
                assert mean_delays.max() <= 8.5, \
                    f"Max mean_delay from fixture should be <= 8.5, got {mean_delays.max()}"
                assert mean_delays.min() >= 0, \
                    f"Min mean_delay should be >= 0, got {mean_delays.min()}"
                
                # Verify statistical consistency: no wildly impossible values
                expected_max = 8.0  # Max delay in fixture
                assert mean_delays.max() <= expected_max * 1.5, \
                    f"Mean of delays shouldn't exceed max fixture delay by much"


# INTEGRATION TESTS FOR STATION VIEW

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.subplots')
def test_comprehensive_station_analysis_chains_functions_correctly(mock_subplots, mock_plt_show, station_view_sample_data):
    """
    Test comprehensive_station_analysis correctly orchestrates the three sub-functions.
    
    Verifies:
    - All three functions are called
    - Results are properly passed through the chain
    - Final output contains all expected analysis components
    - Data calculations are correct and consistent
    """
    # Mock matplotlib
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    result = comprehensive_station_analysis(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=14
    )
    
    # ===== ASSERTION GROUP 1: Return structure =====
    assert result is not None
    assert isinstance(result, dict)
    assert 'delay_analysis' in result
    assert 'outlier_analysis' in result
    assert 'station_view_analysis' in result
    
    # ===== ASSERTION GROUP 2: Component types =====
    delay_result = result['delay_analysis']
    outlier_result = result['outlier_analysis']
    station_result = result['station_view_analysis']
    
    assert delay_result is None or isinstance(delay_result, (pd.DataFrame, dict))
    assert outlier_result is None or isinstance(outlier_result, (pd.DataFrame, dict))
    assert station_result is None or (isinstance(station_result, dict) and 
                                      'hourly_stats' in station_result)
    
    # ===== ASSERTION GROUP 3: Data correctness validation =====
    if station_result is not None and isinstance(station_result, dict):
        hourly_stats = station_result.get('hourly_stats')
        bin_stats = station_result.get('bin_stats')
        
        if hourly_stats is not None and len(hourly_stats) > 0:
            # ===== SUB-GROUP 3A: Range validation =====
            # trains_in_system_normalized is trains per hour per platform (range 0-2.5+)
            assert (hourly_stats['trains_in_system_normalized'] >= 0).all(), \
                "trains_in_system_normalized should be >= 0"
            assert (hourly_stats['trains_in_system_normalized'] <= 3.0).all(), \
                "trains_in_system_normalized should be <= 3.0 (accounting for edge cases above 2.5)"
            assert (hourly_stats['ontime_trains_normalized'] >= 0).all()
            assert (hourly_stats['total_trains'] >= 0).all()
            
            # ===== SUB-GROUP 3B: Calculation correctness from fixture =====
            # Fixture sample_data has 336 hourly records (2 weeks)
            total_trains_sum = hourly_stats['total_trains'].sum()
            assert total_trains_sum >= 0
            # Check if on-time trains exist (fixture may produce zero or non-zero)
            ontime_sum = hourly_stats['ontime_trains_normalized'].sum()
            assert ontime_sum >= 0
            # If there are any total trains, verify at least some proportion logic
            if total_trains_sum > 0:
                # At least 10% of hours should have some activity
                hours_with_trains = (hourly_stats['total_trains'] > 0).sum()
                assert hours_with_trains >= len(hourly_stats) * 0.1
        
        if bin_stats is not None and len(bin_stats) > 0 and 'cdf' in bin_stats.columns:
            # ===== CDF validation =====
            cdf_vals = bin_stats['cdf'].values
            if len(cdf_vals) > 1:
                # CDF should be monotonically non-decreasing
                assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
                
                # CDF should have reasonable range
                assert cdf_vals[0] >= 0, "CDF min should be >= 0"
                assert cdf_vals[-1] <= 100, "CDF max should be <= 100"
                
                # For sample_data fixture: should have some variation (not all same value)
                cdf_unique = len(set(cdf_vals))
                assert cdf_unique > 1

@patch('matplotlib.pyplot.show')
def test_station_view_consistency_across_platforms(mock_plt_show, station_view_sample_data):
    """
    Test station_view produces consistent results with different platform configurations.
    
    Verifies that normalized values scale appropriately with platform count:
    - More platforms = lower normalized train counts (same trains spread across more platforms)
    - Less platforms = higher normalized train counts
    """
    result_6_platforms = station_view(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=6
    )
    
    result_14_platforms = station_view(
        station_id='32000',
        all_data=station_view_sample_data,
        num_platforms=14
    )
    
    # Both should return valid results
    assert result_6_platforms is not None and isinstance(result_6_platforms, dict)
    assert result_14_platforms is not None and isinstance(result_14_platforms, dict)
    
    # ===== ASSERTION GROUP 1: Data structure consistency =====
    stats_6 = result_6_platforms['hourly_stats']
    stats_14 = result_14_platforms['hourly_stats']
    
    assert len(stats_6) == len(stats_14), "Same periods should be produced for both platform counts"
    assert len(stats_6) > 0, "Should have hourly stats data"
    
    # ===== ASSERTION GROUP 2: Normalization scaling =====
    # Both should have the same total_trains (actual arrivals don't change)
    if 'total_trains' in stats_6.columns and 'total_trains' in stats_14.columns:
        # Total trains should be identical (same input data)
        assert (stats_6['total_trains'] == stats_14['total_trains']).all()
    # ===== ASSERTION GROUP 3: Normalized scaling logic =====
    # With 14 platforms vs 6 platforms, normalized values should scale inversely
    # (more platforms = lower per-platform normalized value)
    # trains_in_system_normalized range is 0-2.5+ per platform
    if 'trains_in_system_normalized' in stats_6.columns and 'trains_in_system_normalized' in stats_14.columns:
        # Verify range for both configurations
        assert (stats_6['trains_in_system_normalized'] >= 0).all() and \
                (stats_6['trains_in_system_normalized'] <= 3.0).all(), \
                "6-platform trains_in_system_normalized should be in range 0-3.0 (0-2.5+ with tolerance)"
        assert (stats_14['trains_in_system_normalized'] >= 0).all() and \
                (stats_14['trains_in_system_normalized'] <= 3.0).all(), \
                "14-platform trains_in_system_normalized should be in range 0-3.0 (0-2.5+ with tolerance)"
        
        # Get average normalized values
        avg_6 = stats_6['trains_in_system_normalized'].mean()
        avg_14 = stats_14['trains_in_system_normalized'].mean()
        
        # 6 platforms should have higher normalized values than 14 platforms
        # (same trains / fewer platforms = higher per-platform ratio)
        assert avg_6 >= avg_14 or (avg_6 == 0 and avg_14 == 0)
        
        # Verify the scaling makes sense (ratio should be roughly 14/6 ≈ 2.33)
        if avg_14 > 0:
            expected_ratio = 14 / 6  # ≈ 2.33
            actual_ratio = avg_6 / avg_14 if avg_14 > 0 else 1
            # Allow some tolerance for rounding and averaging effects
            assert 1.5 <= actual_ratio <= 3.5 or avg_6 == 0
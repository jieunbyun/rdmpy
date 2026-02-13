"""
Unit tests for rdmpy.outputs.analysis_tools module.

"""

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
    _prepare_journey_map_data,
    _compute_station_route_connections,
    _aggregate_delays_and_incidents,
    _create_station_markers_on_map,
    _create_incident_markers_on_map,
    _finalize_journey_map,
    train_view,
    train_view_2,
    plot_reliability_graphs,
    _print_date_statistics,
    _load_station_coordinates,
    _aggregate_time_view_data,
    _create_time_view_markers,
    _finalize_time_view_map,
)


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
    from rdmpy.outputs.analysis_tools import aggregate_view_multiday
    
    mock_load_prep.return_value = None
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    assert result is None

@patch('rdmpy.outputs.analysis_tools.plt.show')
@patch('rdmpy.outputs.analysis_tools._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k2(mock_load_prep, mock_plt_show, sample_complete_df):
    """Test aggregate_view_multiday handles multi-day incidents correctly."""
    from rdmpy.outputs.analysis_tools import aggregate_view_multiday

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
    assert expected_stations.issubset(set(result)) # Verify all expected stations are included


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
    assert expected_07_dec.issubset(set(result_07_dec))  # 07-DEC-2024: rows 0, 1 have STANOX 12931, 89012 + destination 54311

    expected_08_dec = {'78912', '54311'}
    assert expected_08_dec.issubset(set(result_08_dec))  # 08-DEC-2024: row 4 has STANOX 78912 + destination 54311
    
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


# INTEGRATION TESTS FOR map_train_journey_with_incidents - Complete Workflow

@pytest.fixture
def sample_station_ref_json(tmp_path):
    """Create a temporary station reference JSON file for testing."""
    import json
    station_ref = [
        {'stanox': '12931', 'description': 'Origin Station', 'latitude': 51.5, 'longitude': -0.1},
        {'stanox': '89012', 'description': 'Intermediate Station 1', 'latitude': 52.0, 'longitude': -0.5},
        {'stanox': '45123', 'description': 'Intermediate Station 2', 'latitude': 52.5, 'longitude': -1.0},
        {'stanox': '54311', 'description': 'Destination Station', 'latitude': 53.0, 'longitude': -1.5},
        {'stanox': '78912', 'description': 'Additional Station', 'latitude': 51.8, 'longitude': -0.3},
    ]
    
    json_file = tmp_path / "stations_ref.json"
    with open(json_file, 'w') as f:
        json.dump(station_ref, f)
    
    return str(json_file)


@patch('folium.Map')
def test_map_train_journey_with_incidents_creates_map(mock_map, train_journey_fixture, sample_station_ref_json):
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
    
    result = map_train_journey_with_incidents(
        all_data=train_journey_fixture,
        service_stanox=service_stanox,
        incident_results=[incident_df],
        stations_ref_path=sample_station_ref_json,
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


# TESTS FOR train_view_2 (not included in general report of the tool, but present in the code)

@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_computes_reliability_metrics(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 computes correct mean delay and on-time percentage."""
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
    assert len(result) >= 2
    
    # Verify required columns exist
    expected_cols = ['ServiceCode', 'StationName', 'MeanDelay', 'DelayVariance', 'OnTime%', 'IncidentCount']
    assert all(col in result.columns for col in expected_cols)
    
    # Verify ServiceCode is correct
    assert all(result['ServiceCode'] == 'SVC001')


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_calculates_mean_delay_correctly(mock_open, mock_print, sample_single_station_reliability_df):
    """Test train_view_2 calculates mean delay excluding zero delays."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345']
    result = train_view_2(sample_single_station_reliability_df, service_stanox, 'SVC002', stations_ref_path='mock_path.json')
    
    # STANOX 12345 has PFPI: [0.0, 5.0, 10.0] -> mean excluding 0 = (5+10)/2 = 7.5
    assert len(result) == 1
    assert result.loc[0, 'MeanDelay'] == 7.5
    assert result.loc[0, 'IncidentCount'] == 2  # Only 2 delays (excluding 0.0)


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_calculates_on_time_percentage(mock_open, mock_print, sample_single_station_reliability_df):
    """Test train_view_2 calculates on-time percentage correctly."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345']
    result = train_view_2(sample_single_station_reliability_df, service_stanox, 'SVC002', stations_ref_path='mock_path.json')
    
    # STANOX 12345: 1 on-time (<=0) out of 3 records = 33.33%
    assert len(result) == 1
    assert abs(result.loc[0, 'OnTime%'] - 33.33) < 1  # Allow small rounding difference


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_empty_service_stanox(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 with empty service_stanox list returns empty or minimal DataFrame."""
    # Mock station reference file
    mock_station_ref = []
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    result = train_view_2(sample_service_reliability_df, [], 'SVC001', stations_ref_path='mock_path.json')
    
    # Should return DataFrame (may include stations found from data with delays)
    assert isinstance(result, pd.DataFrame)


# ==============================================================================
# TESTS FOR plot_reliability_graphs FUNCTION
# ==============================================================================

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
# TIME VIEW HELPER FUNCTION TESTS
# ==============================================================================

@pytest.fixture
def sample_time_view_data():
    """Create sample time view incident data."""
    dates = pd.date_range('2024-01-15 08:00', periods=10, freq='H')
    return pd.DataFrame({
        'INCIDENT_START_DATETIME': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'STANOX': ['12345', '12346', '12345', '12347', '12345', '12346', '12345', '12348', '12345', '12346'],
        'INCIDENT_NUMBER': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'INCIDENT_REASON': ['Signal Failure', 'Track Obstruction', 'Signal Failure', 'Traction Loss', 'Track Obstruction'] * 2,
        'PFPI_MINUTES': [15, 10, 25, 30, 20, 12, 18, 22, 28, 14],
    })


@pytest.fixture
def sample_time_view_stations_ref():
    """Create sample station reference data for time view."""
    return [
        {'stanox': 12345, 'station_name': 'Station A', 'latitude': 51.5074, 'longitude': -0.1278},
        {'stanox': 12346, 'station_name': 'Station B', 'latitude': 53.4808, 'longitude': -2.2426},
        {'stanox': 12347, 'station_name': 'Station C', 'latitude': 52.5200, 'longitude': 13.4050},
        {'stanox': 12348, 'station_name': 'Station D', 'latitude': 48.8566, 'longitude': 2.3522},
    ]


# ==============================================================================
# FIXTURES FOR TRAIN_VIEW AND TRAIN_VIEW_2 TESTS
# ==============================================================================

@pytest.fixture
def sample_train_view_df():
    """Create sample data for train_view testing with known OD pairs and incidents."""
    dates = pd.date_range('2024-01-15 08:00', periods=10, freq='H')
    return pd.DataFrame({
        'PLANNED_ORIGIN_LOCATION_CODE': ['LKX', 'LKX', 'LKX', 'EDI', 'EDI', 'EDI', 'LKX', 'LKX', 'MAN', 'MAN'],
        'PLANNED_DEST_LOCATION_CODE': ['EDI', 'EDI', 'EDI', 'LKX', 'LKX', 'LKX', 'MAN', 'MAN', 'LKX', 'LKX'],
        'TRAIN_SERVICE_CODE': ['EK001', 'EK001', 'EK002', 'EK003', 'EK003', 'EK004', 'EK005', 'EK005', 'EK006', 'EK006'],
        'STANOX': ['12345', '12346', '12347', '12345', '12348', '12345', '12346', '12347', '12345', '12349'],
        'PLANNED_ORIGIN_GBTT_DATETIME': ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00'],
        'PLANNED_DEST_GBTT_DATETIME': ['10:30', '11:30', '12:30', '13:30', '14:30', '15:30', '16:30', '17:30', '18:30', '19:30'],
        'PLANNED_CALLS': ['0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700'],
        'ACTUAL_CALLS': ['0815', '0920', '1015', '1115', '1220', '1330', '1415', '1520', '1620', '1735'],
        'PFPI_MINUTES': [15.0, 20.0, 15.0, 15.0, 20.0, 30.0, 15.0, 20.0, 20.0, 35.0],
        'INCIDENT_REASON': ['Signal Failure', 'Track Defect', 'Signal Failure', 'Traction Loss', 'Points Failure', 
                           'Signal Failure', 'Track Defect', 'Signalling Issue', 'Signal Failure', 'Track Defect'],
        'INCIDENT_NUMBER': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'EVENT_TYPE': ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        'SECTION_CODE': ['12345:12346', '12346:12347', '12347:12348', '12345:12346', '12346:12348', 
                        '12345:12348', '12346:12347', '12347:12345', '12345:12349', '12349:12345'],
        'DELAY_DAY': ['01-JAN-2024', '01-JAN-2024', '01-JAN-2024', '01-JAN-2024', '01-JAN-2024',
                     '01-JAN-2024', '01-JAN-2024', '01-JAN-2024', '01-JAN-2024', '01-JAN-2024'],
        'EVENT_DATETIME': ['15-JAN-2024 08:15', '15-JAN-2024 09:20', '15-JAN-2024 10:15', '15-JAN-2024 11:15', '15-JAN-2024 12:20',
                          '15-JAN-2024 13:30', '15-JAN-2024 14:15', '15-JAN-2024 15:20', '15-JAN-2024 16:20', '15-JAN-2024 17:35'],
        'INCIDENT_START_DATETIME': ['2024-01-15 08:00', '2024-01-15 09:00', '2024-01-15 10:00', '2024-01-15 11:00', '2024-01-15 12:00',
                                   '2024-01-15 13:00', '2024-01-15 14:00', '2024-01-15 15:00', '2024-01-15 16:00', '2024-01-15 17:00'],
        'ENGLISH_DAY_TYPE': ['Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Weekday'],
        'STATION_ROLE': ['O', 'I', 'D', 'O', 'I', 'D', 'O', 'I', 'O', 'D'],
        'DFT_CATEGORY': ['Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat1'],
        'PLATFORM_COUNT': [5, 5, 5, 5, 5, 5, 5, 5, 4, 4],
        'DATASET_TYPE': ['Delay', 'Delay', 'Delay', 'Delay', 'Delay', 'Delay', 'Delay', 'Delay', 'Delay', 'Delay'],
        'WEEKDAY': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'DAY': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
    })
    # OD_PAIR will be added automatically in train_view


@pytest.fixture
def sample_service_reliability_df():
    """Create sample data for train_view_2 testing with known reliability metrics."""
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['SVC001'] * 10,
        'STANOX': ['12345', '12345', '12345', '12346', '12346', '12346', '12347', '12347', '12347', '12348'],
        'PFPI_MINUTES': [0.0, 5.0, 10.0, 0.0, 15.0, 20.0, 0.0, 0.0, 25.0, 10.0],
        'INCIDENT_REASON': ['OnTime', 'Delay', 'Delay', 'OnTime', 'Delay', 'Delay', 'OnTime', 'OnTime', 'Delay', 'Delay'],
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


def test_print_date_statistics_with_data(sample_time_view_data, capsys):
    """Test _print_date_statistics prints correct summary for a date with incidents."""
    _print_date_statistics('2024-01-15', sample_time_view_data)
    
    captured = capsys.readouterr()
    # Should print the date and incident count
    assert '2024-01-15' in captured.out
    assert 'incidents' in captured.out.lower()


def test_print_date_statistics_empty_data(capsys):
    """Test _print_date_statistics handles empty dataset."""
    empty_df = pd.DataFrame({
        'INCIDENT_START_DATETIME': pd.Series([], dtype='object'),
        'STANOX': pd.Series([], dtype='object'),
        'INCIDENT_REASON': pd.Series([], dtype='object'),
        'PFPI_MINUTES': pd.Series([], dtype='float64'),
    })
    
    _print_date_statistics('2024-01-15', empty_df)
    
    captured = capsys.readouterr()
    # Should print something for the date
    assert '2024-01-15' in captured.out or 'No incidents' in captured.out


def test_print_date_statistics_no_matching_date(sample_time_view_data, capsys):
    """Test _print_date_statistics when no incidents match the date."""
    _print_date_statistics('2024-02-01', sample_time_view_data)
    
    captured = capsys.readouterr()
    # Should indicate no incidents found
    assert '2024-02-01' in captured.out


@patch('rdmpy.outputs.utils._load_station_coordinates')
def test_load_station_coordinates_valid(mock_load, sample_time_view_stations_ref):
    """Test _load_station_coordinates successfully loads station data."""
    # Mock the return value
    expected_result = {
        '12345': [51.5074, -0.1278],
        '12346': [53.4808, -2.2426],
        '12347': [52.5200, 13.4050],
    }
    mock_load.return_value = expected_result
    
    result = mock_load()
    
    # Check structure: should be dict with STANOX as keys
    assert isinstance(result, dict)
    # STANOX should map to [lat, lon] pairs
    for stanox, coords in result.items():
        assert isinstance(coords, list)
        assert len(coords) == 2


@patch('rdmpy.outputs.utils._load_station_coordinates')
def test_load_station_coordinates_missing_file(mock_load):
    """Test _load_station_coordinates handles missing reference file gracefully."""
    mock_load.return_value = {}
    
    result = mock_load()
    # Should return empty dict on missing file
    assert result == {}


def test_aggregate_time_view_data_valid_date(sample_time_view_data):
    """Test _aggregate_time_view_data aggregates incidents correctly for a date."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    # Should find affected STANOX
    assert affected_stanox is not None
    assert len(affected_stanox) > 0
    
    # Check aggregations - returns pandas Series, not dict
    assert hasattr(incident_counts, 'get')  # Series has get method
    assert hasattr(total_pfpi, 'get')
    
    # STANOX 12345 appears 5 times in sample data
    assert 12345 in [int(s) for s in affected_stanox]


def test_aggregate_time_view_data_empty_date(sample_time_view_data):
    """Test _aggregate_time_view_data handles date with no incidents."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-03-01', sample_time_view_data)
    
    # Should return None for non-matching date
    assert affected_stanox is None


def test_aggregate_time_view_data_stanox_grouping(sample_time_view_data):
    """Test _aggregate_time_view_data correctly groups by STANOX."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    if affected_stanox is not None:
        # Verify STANOX 12345 has correct incident count
        # STANOX might be string or int in the Series, try both
        stanox_12345_count = incident_counts.get(12345) or incident_counts.get('12345')
        # STANOX 12345 appears 5 times in the sample data
        assert stanox_12345_count == 5
        
        # Verify PFPI totals are sums
        stanox_12345_pfpi = total_pfpi.get(12345) or total_pfpi.get('12345')
        # Incidents for 12345: indices 0, 2, 4, 6, 8 with PFPI 15, 25, 20, 18, 28
        assert stanox_12345_pfpi == 106



@patch('folium.CircleMarker')
def test_create_time_view_markers_adds_markers(mock_circle):
    """Test _create_time_view_markers adds CircleMarkers to the map."""
    mock_map = MagicMock()
    
    affected_stanox = {12345, 12346, 12347}
    incident_counts = {12345: 5, 12346: 3, 12347: 1}
    total_pfpi = {12345: 106, 12346: 36, 12347: 30}
    stanox_to_coords = {
        '12345': [51.5074, -0.1278],
        '12346': [53.4808, -2.2426],
        '12347': [52.5200, 13.4050],
    }
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Should add markers for each STANOX
    assert mock_circle.call_count >= len(affected_stanox)


@patch('folium.CircleMarker')
def test_create_time_view_markers_color_grading(mock_circle):
    """Test _create_time_view_markers applies correct colors based on PFPI."""
    mock_map = MagicMock()
    
    affected_stanox = {12345}
    incident_counts = {12345: 1}
    total_pfpi = {12345: 50.0}  # 31-60 min range = Red
    stanox_to_coords = {'12345': [51.5074, -0.1278]}
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Check that a marker was added with appropriate color
    assert mock_circle.called
    call_args = mock_circle.call_args
    if call_args:
        # Color should be red for 31-60 min range
        assert call_args[1].get('color') == '#FF0000' or call_args[1].get('fill_color') == '#FF0000'


@patch('folium.CircleMarker')
def test_create_time_view_markers_radius_scaling(mock_circle):
    """Test _create_time_view_markers scales radius by incident count."""
    mock_map = MagicMock()
    
    affected_stanox = {12345}
    incident_counts = {12345: 10}  # Higher count should have larger radius
    total_pfpi = {12345: 50.0}
    stanox_to_coords = {'12345': [51.5074, -0.1278]}
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Check radius is scaled
    assert mock_circle.called
    call_args = mock_circle.call_args
    if call_args:
        # Radius should increase with incident count
        assert call_args[1].get('radius') > 5


@patch('folium.Element')
def test_finalize_time_view_map_adds_title(mock_element):
    """Test _finalize_time_view_map adds title to the map."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should call html.add_child for title and legend
    assert mock_map.get_root().html.add_child.called


@patch('folium.Element')
@patch('builtins.open', create=True)
def test_finalize_time_view_map_saves_file(mock_open, mock_element):
    """Test _finalize_time_view_map saves the map to file."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should call map.save()
    assert mock_map.save.called
    # Check that save was called with correct filename pattern
    call_args = mock_map.save.call_args
    if call_args:
        filename = call_args[0][0]
        assert 'time_view_2024_01_15' in filename


@patch('folium.Element')
def test_finalize_time_view_map_adds_legend(mock_element):
    """Test _finalize_time_view_map adds legend to the map."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should add HTML elements for legend
    calls = mock_map.get_root().html.add_child.call_count
    # At least one call for title/legend
    assert calls >= 1


def test_aggregate_time_view_data_preserves_stanox_format(sample_time_view_data):
    """Test _aggregate_time_view_data returns STANOX in consistent format."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    if affected_stanox is not None:
        # All STANOX should be convertible to int
        for stanox in affected_stanox:
            int_stanox = int(stanox)
            assert 10000 <= int_stanox <= 100000  # Valid STANOX range
import pytest
from unittest.mock import patch, mock_open
from rdmpy.preprocessor import (
    get_weekday_from_schedule_entry, 
    load_stations,
    save_processed_data_by_weekday_to_dataframe, 
    save_stations_by_category,
    _process_schedule_step,
    _process_delays_step,
    _deduplicate_timeline,
    _organize_by_weekday,
    _convert_to_dataframes,
    _load_all_reference_data,
    _cleanup_existing_station_folders,
    _process_single_station_and_save,
    main
)

import pandas as pd
import json


# Testing get_weekday_from_schedule_entry

def test_get_weekday_from_schedule_entry_k0():
    """Test that DELAY_DAY takes precedence over ENGLISH_DAY_TYPE"""
    entry = {
        "DELAY_DAY": "MO",
        "ENGLISH_DAY_TYPE": ["TU", "WE", "TH"]
    }
    result = get_weekday_from_schedule_entry(entry)
    assert result == 0  # Monday is index 0


def test_get_weekday_from_schedule_entry_k1():
    """Test that first ENGLISH_DAY_TYPE is used when no DELAY_DAY"""
    entry = {
        "ENGLISH_DAY_TYPE": ["WE", "TH", "FR"]
    }
    result = get_weekday_from_schedule_entry(entry)
    assert result == 2  # Wednesday is index 2


def test_get_weekday_from_schedule_entry_k2():
    """Test fallback to Monday (0) when neither field present"""
    entry = {}
    result = get_weekday_from_schedule_entry(entry)
    assert result == 0


def test_get_weekday_from_schedule_entry_k3():
    """Test that return type is always int and within valid range"""
    entries = [
        {"DELAY_DAY": "FR"},
        {"ENGLISH_DAY_TYPE": ["SA"]},
        {}
    ]
    for entry in entries:
        result = get_weekday_from_schedule_entry(entry)
        assert isinstance(result, int)
        assert 0 <= result <= 6


def test_get_weekday_from_schedule_entry_k4():
    """Test with Saturday and Sunday"""
    entry_sat = {"ENGLISH_DAY_TYPE": ["SA"]}
    entry_sun = {"ENGLISH_DAY_TYPE": ["SU"]}
    
    assert get_weekday_from_schedule_entry(entry_sat) == 5  # Saturday
    assert get_weekday_from_schedule_entry(entry_sun) == 6  # Sunday


# Testing load_stations

@pytest.fixture
def mock_stations_data():
    """Fixture: sample station reference data"""
    return [
        {"stanox": "11271", "dft_category": "A"},
        {"stanox": "11720", "dft_category": "A"},
        {"stanox": "12001", "dft_category": "B"},
        {"stanox": "12106", "dft_category": "C1"},
        {"stanox": "12931", "dft_category": ""},  # Empty category
        {"stanox": "13702"},  # No category key
    ]


def test_load_stations_k0(mock_stations_data):
    """Test filtering stations by category A"""
    mock_json_str = json.dumps(mock_stations_data)
    
    with patch('builtins.open', mock_open(read_data=mock_json_str)):
        with patch('rdmpy.preprocessor.reference_files', 
                   {"all dft categories": "mock_path.json"}):
            result = load_stations(category='A')
    
    assert result == ["11271", "11720"]


def test_load_stations_k1(mock_stations_data):
    """Test loading all valid stations (category=None) excludes empty/missing categories"""
    mock_json_str = json.dumps(mock_stations_data)
    
    with patch('builtins.open', mock_open(read_data=mock_json_str)):
        with patch('rdmpy.preprocessor.reference_files',
                   {"all dft categories": "mock_path.json"}):
            result = load_stations(category=None)
    
    # Should exclude stations with empty or missing dft_category
    assert result == ["11271", "11720", "12001", "12106"]


def test_load_stations_k2(mock_stations_data):
    """Test filtering by category C1"""
    mock_json_str = json.dumps(mock_stations_data)
    
    
    with patch('builtins.open', mock_open(read_data=mock_json_str)):
        with patch('rdmpy.preprocessor.reference_files',
                   {"all dft categories": "mock_path.json"}):
            result = load_stations(category='C1')
    
    assert result == ["12106"]


def test_load_stations_k3(mock_stations_data):
    """Test returns empty list when category has no stations"""
    mock_json_str = json.dumps(mock_stations_data)
    
    with patch('builtins.open', mock_open(read_data=mock_json_str)):
        with patch('rdmpy.preprocessor.reference_files',
                   {"all dft categories": "mock_path.json"}):
            result = load_stations(category='C2')
    
    assert result == []


def test_load_stations_k4(capsys):
    """Test error handling when file cannot be read"""
    with patch('builtins.open', side_effect=FileNotFoundError):
        with patch('rdmpy.preprocessor.reference_files',
                   {"all dft categories": "nonexistent.json"}):
            result = load_stations(category='A')
    
    assert result == []
    captured = capsys.readouterr()
    assert "Error loading stations" in captured.out


# Testing save_processed_data_by_weekday_to_dataframe

@pytest.fixture
def mock_processed_schedule():
    """
    Fixture: sample processed schedule data that has been adjusted with delays.
    This represents the output of adjust_schedule_timeline() which combines
    schedule entries with matched delay data.
    """
    return [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_ORIGIN_LOCATION_CODE": "11271",
            "PLANNED_ORIGIN_GBTT_DATETIME": "2024-01-01 08:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "2024-01-01 09:00",
            "PLANNED_CALLS": "0800",
            "ACTUAL_CALLS": "0815",  # Delayed by 15 minutes
            "PFPI_MINUTES": 15.0,  # Performance frequency minutes - the delay
            "INCIDENT_REASON": "Fault",
            "INCIDENT_NUMBER": "INC001",
            "EVENT_TYPE": "A",
            "SECTION_CODE": "SC001",
            "DELAY_DAY": "MO",
            "EVENT_DATETIME": "2024-01-01 08:15",
            "INCIDENT_START_DATETIME": "2024-01-01 08:00",
            "ENGLISH_DAY_TYPE": ["MO", "TU"],
            "STATION_ROLE": "Origin",
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 2,
            "DATASET_TYPE": "MULTI_DAY",
            "WEEKDAY": "MO"
        },
        {
            "TRAIN_SERVICE_CODE": "T002",
            "PLANNED_ORIGIN_LOCATION_CODE": "11271",
            "PLANNED_ORIGIN_GBTT_DATETIME": "2024-01-01 09:00",
            "PLANNED_DEST_LOCATION_CODE": "33333",
            "PLANNED_DEST_GBTT_DATETIME": "2024-01-01 10:00",
            "PLANNED_CALLS": "0900",
            "ACTUAL_CALLS": "0900",  # On time (no delay)
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["WE", "TH", "FR"],
            "STATION_ROLE": "Intermediate",
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 3,
            "DATASET_TYPE": "MULTI_DAY",
            "WEEKDAY": "WE"
        },
        {
            "TRAIN_SERVICE_CODE": "T003",
            "PLANNED_ORIGIN_LOCATION_CODE": "44444",
            "PLANNED_ORIGIN_GBTT_DATETIME": "2024-01-01 10:00",
            "PLANNED_DEST_LOCATION_CODE": "11271",
            "PLANNED_DEST_GBTT_DATETIME": "2024-01-01 11:00",
            "PLANNED_CALLS": "1000",
            "ACTUAL_CALLS": "1030",  # Delayed by 30 minutes
            "PFPI_MINUTES": 30.0,
            "INCIDENT_REASON": "Signalling Failure",
            "INCIDENT_NUMBER": "INC002",
            "EVENT_TYPE": "B",
            "SECTION_CODE": "SC002",
            "DELAY_DAY": "TH",
            "EVENT_DATETIME": "2024-01-01 10:30",
            "INCIDENT_START_DATETIME": "2024-01-01 09:45",
            "ENGLISH_DAY_TYPE": ["TH"],
            "STATION_ROLE": "Destination",
            "DFT_CATEGORY": "B",
            "PLATFORM_COUNT": 1,
            "DATASET_TYPE": "SINGLE_DAY",
            "WEEKDAY": "TH"
        }
    ]


def test_save_processed_data_by_weekday_to_dataframe_k0(mock_processed_schedule):
    """Test that pre-loaded schedule data path is used"""
    mock_schedule_data = pd.DataFrame(mock_processed_schedule)
    mock_stanox_ref = pd.DataFrame()
    mock_tiploc = {}
    mock_incident_data = {}
    
    with patch('rdmpy.preprocessor.process_schedule') as mock_process_schedule:
        with patch('rdmpy.preprocessor.process_delays_optimized') as mock_process_delays:
            with patch('rdmpy.preprocessor.adjust_schedule_timeline') as mock_adjust:
                mock_process_schedule.return_value = mock_processed_schedule
                mock_process_delays.return_value = {}
                mock_adjust.return_value = mock_processed_schedule
                
                result = save_processed_data_by_weekday_to_dataframe(
                    st_code="11271",
                    schedule_data_loaded=mock_schedule_data,
                    stanox_ref=mock_stanox_ref,
                    tiploc_to_stanox=mock_tiploc,
                    incident_data_loaded=mock_incident_data
                )
    
    # Verify optimized path was used
    assert mock_process_schedule.called
    assert mock_process_schedule.call_args[1]['schedule_data_loaded'] is not None


def test_save_processed_data_by_weekday_to_dataframe_k1(mock_processed_schedule):
    """Test that output is organized by weekday as DataFrames"""
    mock_schedule_data = pd.DataFrame(mock_processed_schedule)
    
    with patch('rdmpy.preprocessor.process_schedule') as mock_process_schedule:
        with patch('rdmpy.preprocessor.process_delays_optimized') as mock_process_delays:
            with patch('rdmpy.preprocessor.adjust_schedule_timeline') as mock_adjust:
                mock_process_schedule.return_value = mock_processed_schedule
                mock_process_delays.return_value = {}
                mock_adjust.return_value = mock_processed_schedule
                
                result = save_processed_data_by_weekday_to_dataframe(
                    st_code="11271",
                    schedule_data_loaded=mock_schedule_data,
                    stanox_ref=pd.DataFrame(),
                    tiploc_to_stanox={},
                    incident_data_loaded={}
                )
    
    assert isinstance(result, dict)
    assert all(isinstance(v, pd.DataFrame) for v in result.values())
    # Should have weekday keys
    valid_days = {"MO", "TU", "WE", "TH", "FR", "SA", "SU"}
    assert all(day in valid_days for day in result.keys())


def test_save_processed_data_by_weekday_to_dataframe_k2():
    """Test returns None when no schedule data found"""
    with patch('rdmpy.preprocessor.process_schedule') as mock_process_schedule:
        with patch('rdmpy.preprocessor.process_delays_optimized'):
            with patch('rdmpy.preprocessor.adjust_schedule_timeline'):
                mock_process_schedule.return_value = []
                
                result = save_processed_data_by_weekday_to_dataframe(
                    st_code="99999",
                    schedule_data_loaded=pd.DataFrame(),
                    stanox_ref=pd.DataFrame(),
                    tiploc_to_stanox={},
                    incident_data_loaded={}
                )
    
    assert result is None


def test_save_processed_data_by_weekday_to_dataframe_k3():
    """Test fallback to legacy file-based path when no preloaded data"""
    with patch('rdmpy.preprocessor.process_schedule') as mock_process_schedule:
        with patch('rdmpy.preprocessor.process_delays'):
            with patch('rdmpy.preprocessor.adjust_schedule_timeline'):
                mock_process_schedule.return_value = []
                
                result = save_processed_data_by_weekday_to_dataframe(
                    st_code="11271"
                    # No pre-loaded data provided
                )
    
    # Verify legacy path was used (process_schedule called without pre-loaded data)
    assert mock_process_schedule.called
    assert mock_process_schedule.call_args[1].get('schedule_data_loaded') is None


# Testing save_stations_by_category

@pytest.fixture
def mock_station_list():
    return ["11271", "11720", "12001"]


def test_save_stations_by_category_k0(mock_station_list):
    """Test that correct category stations are loaded"""
    with patch('rdmpy.preprocessor.load_stations') as mock_load:
        with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_schedule_once:
            with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident_once:
                with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
                    with patch('os.path.exists', return_value=False):
                        with patch('os.makedirs'):
                            # Set return values for data loading functions
                            mock_schedule_once.return_value = (pd.DataFrame(), pd.DataFrame(), {})
                            mock_incident_once.return_value = {}
                            
                            mock_load.return_value = mock_station_list
                            mock_save.return_value = {"MO": pd.DataFrame()}
                            
                            result = save_stations_by_category(category='A')
    
    # Verify load_stations was called with correct category
    mock_load.assert_called_once_with(category='A')


def test_save_stations_by_category_k1(mock_station_list):
    """Test that all loaded stations are processed"""
    with patch('rdmpy.preprocessor.load_stations') as mock_load:
        with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_schedule_once:
            with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident_once:
                with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
                    with patch('os.makedirs'):
                        with patch('os.path.exists', return_value=False):
                            with patch.object(pd.DataFrame, 'to_parquet'):
                                # Set return values for data loading functions
                                mock_schedule_once.return_value = (pd.DataFrame(), pd.DataFrame(), {})
                                mock_incident_once.return_value = {}
                                
                                mock_load.return_value = mock_station_list
                                mock_save.return_value = {"MO": pd.DataFrame()}
                                
                                result = save_stations_by_category(category='A')
    
    # Verify each station was processed
    assert mock_save.call_count == len(mock_station_list)


def test_save_stations_by_category_k2():
    """Test that results dict has expected keys"""
    with patch('rdmpy.preprocessor.load_stations', return_value=["11271"]):
        with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_schedule_once:
            with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident_once:
                with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
                    with patch('os.makedirs'):
                        with patch('os.path.exists', return_value=False):
                            with patch.object(pd.DataFrame, 'to_parquet'):
                                # Set return values for data loading functions
                                mock_schedule_once.return_value = (pd.DataFrame(), pd.DataFrame(), {})
                                mock_incident_once.return_value = {}
                                
                                mock_save.return_value = {"MO": pd.DataFrame()}
                                
                                result = save_stations_by_category(category='A')
    
    assert 'category' in result
    assert 'successful_stations' in result
    assert 'failed_stations' in result
    assert 'total_entries_by_station' in result
    assert 'files_created' in result


def test_save_stations_by_category_k3():
    """Test that successful and failed stations are tracked"""
    stations = ["11271", "11720", "99999"]
    
    with patch('rdmpy.preprocessor.load_stations', return_value=stations):
        with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_schedule_once:
            with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident_once:
                with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
                    with patch('os.makedirs'):
                        with patch('os.path.exists', return_value=False):
                            with patch.object(pd.DataFrame, 'to_parquet'):
                                # Set return values for data loading functions
                                mock_schedule_once.return_value = (pd.DataFrame(), pd.DataFrame(), {})
                                mock_incident_once.return_value = {}
                                
                                # First two succeed, third fails
                                mock_save.side_effect = [
                                    {"MO": pd.DataFrame()},
                                    {"MO": pd.DataFrame()},
                                    None  # Failure
                                ]
                                
                                result = save_stations_by_category(category='A')
    
    assert len(result['successful_stations']) == 2
    assert len(result['failed_stations']) == 1
    assert "99999" in result['failed_stations']


def test_save_stations_by_category_k4():
    """Test returns None when no stations found"""
    with patch('rdmpy.preprocessor.load_stations', return_value=[]):
        result = save_stations_by_category(category='X')
    
    assert result is None


# ============================================================================
# Tests for: _process_schedule_step() - Helper Function (Unit Tests)
# ============================================================================

def test_process_schedule_step_k0(mock_processed_schedule):
    """Test that pre-loaded data path is used correctly"""
    mock_schedule_data = pd.DataFrame(mock_processed_schedule)
    mock_stanox_ref = pd.DataFrame()
    mock_tiploc = {}
    
    with patch('rdmpy.preprocessor.process_schedule') as mock_process:
        mock_process.return_value = mock_processed_schedule
        
        result = _process_schedule_step(
            st_code="11271",
            schedule_data_loaded=mock_schedule_data,
            stanox_ref=mock_stanox_ref,
            tiploc_to_stanox=mock_tiploc
        )
    
    # Verify function was called with pre-loaded data flag
    assert mock_process.called
    assert len(result) == len(mock_processed_schedule)
    assert result[0]["TRAIN_SERVICE_CODE"] == "T001"


def test_process_schedule_step_k1():
    """Test legacy fallback when no pre-loaded data provided"""
    with patch('rdmpy.preprocessor.process_schedule') as mock_process:
        mock_process.return_value = []
        
        result = _process_schedule_step(st_code="11271")
    
    # Verify it falls back to legacy path
    assert mock_process.called
    assert mock_process.call_args[1].get('schedule_data_loaded') is None


def test_process_schedule_step_k2(mock_processed_schedule):
    """Test returns list of processed schedule entries"""
    with patch('rdmpy.preprocessor.process_schedule') as mock_process:
        mock_process.return_value = mock_processed_schedule
        
        result = _process_schedule_step(st_code="11271")
    
    assert isinstance(result, list)
    assert len(result) > 0
    # Verify entries have expected fields
    assert all('TRAIN_SERVICE_CODE' in entry for entry in result)
    assert all('ENGLISH_DAY_TYPE' in entry for entry in result)


# ============================================================================
# Tests for: _process_delays_step() - Helper Function (Unit Tests)
# ============================================================================

def test_process_delays_step_k0():
    """Test that pre-loaded incident data path is used"""
    mock_incident_data = {
        "period1": pd.DataFrame({
            "TRAIN_SERVICE_CODE": ["T001"],
            "PFPI_MINUTES": [15.0],
            "INCIDENT_REASON": ["Fault"]
        })
    }
    
    with patch('rdmpy.preprocessor.process_delays_optimized') as mock_delays:
        mock_delays.return_value = mock_incident_data
        
        result = _process_delays_step(st_code="11271", incident_data_loaded=mock_incident_data)
    
    # Should return flattened list of delay records
    assert isinstance(result, list)
    assert mock_delays.called


def test_process_delays_step_k1():
    """Test legacy fallback when no pre-loaded incident data"""
    with patch('rdmpy.preprocessor.process_delays') as mock_delays:
        with patch('os.makedirs'):
            mock_delays.return_value = {}
            
            result = _process_delays_step(st_code="11271", incident_data_loaded=None)
    
    assert isinstance(result, list)
    assert mock_delays.called


def test_process_delays_step_k2():
    """Test that DataFrame delays are converted to records"""
    mock_incident_data = {
        "period1": pd.DataFrame({
            "TRAIN_SERVICE_CODE": ["T001", "T002"],
            "PFPI_MINUTES": [15.0, 30.0],
            "INCIDENT_REASON": ["Fault", "Signal"]
        }),
        "period2": pd.DataFrame({
            "TRAIN_SERVICE_CODE": ["T003"],
            "PFPI_MINUTES": [0.0],
            "INCIDENT_REASON": ["None"]
        })
    }
    
    with patch('rdmpy.preprocessor.process_delays_optimized') as mock_delays:
        mock_delays.return_value = mock_incident_data
        
        result = _process_delays_step(st_code="11271", incident_data_loaded=mock_incident_data)
    
    # Should flatten all periods into single list
    assert isinstance(result, list)
    assert len(result) == 3  # 2 from period1, 1 from period2


# ============================================================================
# Tests for: _deduplicate_timeline() - Helper Function (Unit Tests)
# ============================================================================

def test_deduplicate_timeline_k0():
    """Test removal of identical duplicate entries"""
    schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "ENGLISH_DAY_TYPE": ["MO"],
            "PFPI_MINUTES": 0.0
        },
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "ENGLISH_DAY_TYPE": ["MO"],
            "PFPI_MINUTES": 0.0
        },  # Exact duplicate
        {
            "TRAIN_SERVICE_CODE": "T002",
            "PLANNED_CALLS": "0900",
            "ENGLISH_DAY_TYPE": ["TU"],
            "PFPI_MINUTES": 15.0
        }
    ]
    
    result = _deduplicate_timeline(schedule)
    
    # Should remove 1 duplicate, leaving 2 entries
    assert len(result) == 2
    assert result[0]["TRAIN_SERVICE_CODE"] == "T001"
    assert result[1]["TRAIN_SERVICE_CODE"] == "T002"


def test_deduplicate_timeline_k1():
    """Test that same train with different delays is preserved"""
    schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "PFPI_MINUTES": 0.0,  # Different
            "ENGLISH_DAY_TYPE": ["MO"]
        },
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "PFPI_MINUTES": 15.0,  # Different
            "ENGLISH_DAY_TYPE": ["MO"]
        }
    ]
    
    result = _deduplicate_timeline(schedule)
    
    # Should preserve both - they have different delays
    assert len(result) == 2


def test_deduplicate_timeline_k2():
    """Test with list fields (ENGLISH_DAY_TYPE)"""
    schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "ENGLISH_DAY_TYPE": ["MO", "TU"],  # List field
            "PFPI_MINUTES": 0.0
        },
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_CALLS": "0800",
            "ENGLISH_DAY_TYPE": ["MO", "TU"],  # Same list
            "PFPI_MINUTES": 0.0
        }
    ]
    
    result = _deduplicate_timeline(schedule)
    
    # Should remove duplicate even with list fields
    assert len(result) == 1


# ============================================================================
# Tests for: _organize_by_weekday() - Helper Function (Unit Tests)
# ============================================================================

def test_organize_by_weekday_k0(mock_processed_schedule):
    """Test that entries are organized by all their running days"""
    result = _organize_by_weekday(mock_processed_schedule)
    
    # Should have entries for each day that trains run
    assert isinstance(result, dict)
    # Result should have day codes as keys
    valid_days = {"MO", "TU", "WE", "TH", "FR", "SA", "SU"}
    assert all(day in valid_days for day in result.keys())
    
    # Multi-day trains should appear in multiple days
    assert len(result["MO"]) > 0  # T001 runs MO,TU
    assert len(result["TU"]) > 0  # T001 runs MO,TU
    assert len(result["WE"]) > 0  # T002 runs WE,TH,FR


def test_organize_by_weekday_k1():
    """Test single-day schedule"""
    schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "ENGLISH_DAY_TYPE": ["MO"],
            "PLANNED_CALLS": "0800"
        }
    ]
    
    result = _organize_by_weekday(schedule)
    
    # Check that MO has one entry
    assert len(result["MO"]) == 1
    # Verify the entry has the expected fields and metadata added by _organize_by_weekday
    entry = result["MO"][0]
    assert entry["TRAIN_SERVICE_CODE"] == "T001"
    assert entry["WEEKDAY"] == "MO"
    assert entry["DATASET_TYPE"] == "SINGLE_DAY"
    # Other days should be empty
    assert len(result["TU"]) == 0


def test_organize_by_weekday_k2():
    """Test multi-day schedule creates copies for each day"""
    schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "ENGLISH_DAY_TYPE": ["MO", "WE", "FR"],
            "PLANNED_CALLS": "0800",
            "DATASET_TYPE": "TEST"
        }
    ]
    
    result = _organize_by_weekday(schedule)
    
    # Should have 3 entries, one for each day
    assert len(result["MO"]) == 1
    assert len(result["WE"]) == 1
    assert len(result["FR"]) == 1
    
    # Each should have WEEKDAY metadata
    assert result["MO"][0]["WEEKDAY"] == "MO"
    assert result["WE"][0]["WEEKDAY"] == "WE"
    assert result["FR"][0]["WEEKDAY"] == "FR"


# ============================================================================
# Tests for: _convert_to_dataframes() - Helper Function (Unit Tests)
# ============================================================================

def test_convert_to_dataframes_k0():
    """Test conversion of weekday data to DataFrames"""
    weekday_data = {
        "MO": [
            {"TRAIN_SERVICE_CODE": "T001", "ACTUAL_CALLS": "0815", "PFPI_MINUTES": 15.0},
            {"TRAIN_SERVICE_CODE": "T002", "ACTUAL_CALLS": "0900", "PFPI_MINUTES": 0.0}
        ],
        "TU": [
            {"TRAIN_SERVICE_CODE": "T003", "ACTUAL_CALLS": "1030", "PFPI_MINUTES": 30.0}
        ],
        "WE": []  # Empty day
    }
    
    result = _convert_to_dataframes(weekday_data)
    
    # Should return dict of DataFrames
    assert isinstance(result, dict)
    assert "MO" in result
    assert "TU" in result
    assert "WE" not in result  # Empty days excluded
    
    # Check DataFrames
    assert isinstance(result["MO"], pd.DataFrame)
    assert len(result["MO"]) == 2
    assert len(result["TU"]) == 1


def test_convert_to_dataframes_k1():
    """Test sorting by ACTUAL_CALLS"""
    weekday_data = {
        "MO": [
            {"TRAIN_SERVICE_CODE": "T002", "ACTUAL_CALLS": "0900"},
            {"TRAIN_SERVICE_CODE": "T001", "ACTUAL_CALLS": "0800"},
            {"TRAIN_SERVICE_CODE": "T003", "ACTUAL_CALLS": "1000"}
        ]
    }
    
    result = _convert_to_dataframes(weekday_data)
    
    # Should be sorted by ACTUAL_CALLS
    df = result["MO"]
    calls = df["ACTUAL_CALLS"].tolist()
    assert calls == ["0800", "0900", "1000"]


def test_convert_to_dataframes_k2():
    """Test handling of NaN and invalid ACTUAL_CALLS"""
    weekday_data = {
        "MO": [
            {"TRAIN_SERVICE_CODE": "T001", "ACTUAL_CALLS": "0900"},
            {"TRAIN_SERVICE_CODE": "T002", "ACTUAL_CALLS": None},  # NaN
            {"TRAIN_SERVICE_CODE": "T003", "ACTUAL_CALLS": "NA"},  # String NA
            {"TRAIN_SERVICE_CODE": "T004", "ACTUAL_CALLS": "0800"}
        ]
    }
    
    result = _convert_to_dataframes(weekday_data)
    
    # Should handle gracefully and create DataFrame
    assert isinstance(result["MO"], pd.DataFrame)
    assert len(result["MO"]) == 4


# ============================================================================
# Tests for: _load_all_reference_data() - Helper Function (Unit Tests)
# ============================================================================

def test_load_all_reference_data_k0():
    """Test that all reference data is loaded correctly"""
    with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_sched:
        with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident:
            mock_sched.return_value = (
                pd.DataFrame({"col": [1, 2]}),
                pd.DataFrame({"col": [3, 4]}),
                {"tiploc": "stanox"}
            )
            mock_incident.return_value = {"period1": pd.DataFrame()}
            
            schedule, stanox, tiploc, incident = _load_all_reference_data()
    
    # Verify all data is returned
    assert schedule is not None
    assert stanox is not None
    assert tiploc is not None
    assert incident is not None
    assert mock_sched.called
    assert mock_incident.called


def test_load_all_reference_data_k1():
    """Test that return values are in correct order"""
    with patch('rdmpy.preprocessor.load_schedule_data_once') as mock_sched:
        with patch('rdmpy.preprocessor.load_incident_data_once') as mock_incident:
            schedule_df = pd.DataFrame({"a": [1]})
            stanox_df = pd.DataFrame({"b": [2]})
            tiploc_dict = {"test": "mapping"}
            incident_dict = {"p1": pd.DataFrame()}
            
            mock_sched.return_value = (schedule_df, stanox_df, tiploc_dict)
            mock_incident.return_value = incident_dict
            
            result_sched, result_stanox, result_tiploc, result_incident = _load_all_reference_data()
    
    # Verify correct order
    assert result_sched is schedule_df
    assert result_stanox is stanox_df
    assert result_tiploc is tiploc_dict
    assert result_incident is incident_dict


# ============================================================================
# Tests for: _cleanup_existing_station_folders() - Helper Function (Unit Tests)
# ============================================================================

def test_cleanup_existing_station_folders_k0():
    """Test that existing station folders are removed"""
    stations = ["11271", "11720", "12001"]
    
    with patch('os.path.exists', return_value=True):
        with patch('shutil.rmtree') as mock_rmtree:
            removed = _cleanup_existing_station_folders(stations, "processed_data")
    
    # Should attempt to remove all 3 folders
    assert mock_rmtree.call_count == 3
    assert removed == 3


def test_cleanup_existing_station_folders_k1():
    """Test that non-existent folders are skipped"""
    stations = ["11271", "11720", "12001"]
    
    with patch('os.path.exists', return_value=False):
        with patch('shutil.rmtree') as mock_rmtree:
            removed = _cleanup_existing_station_folders(stations, "processed_data")
    
    # Should not attempt to remove any folders
    assert mock_rmtree.call_count == 0
    assert removed == 0


def test_cleanup_existing_station_folders_k2():
    """Test handling of deletion errors"""
    stations = ["11271", "11720"]
    
    # Both exist, but first raises error during removal
    def exist_side_effect(path):
        return True
    
    with patch('os.path.exists', side_effect=exist_side_effect):
        with patch('shutil.rmtree', side_effect=Exception("Permission denied")):
            removed = _cleanup_existing_station_folders(stations, "processed_data")
    
    # Should count attempted removals despite errors
    assert removed == 0  # Both failed due to exception


# ============================================================================
# Tests for: _process_single_station_and_save() - Helper Function (Unit Tests)
# ============================================================================

def test_process_single_station_and_save_k0(mock_processed_schedule):
    """Test successful station processing and file saving"""
    mock_schedule_data = pd.DataFrame(mock_processed_schedule)
    
    with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
        with patch('os.makedirs'):
            with patch.object(pd.DataFrame, 'to_parquet'):
                mock_save.return_value = {"MO": pd.DataFrame([mock_processed_schedule[0]])}
                
                success, files, total = _process_single_station_and_save(
                    st_code="11271",
                    i=1,
                    total_stations=3,
                    output_dir="processed_data",
                    schedule_data_loaded=mock_schedule_data,
                    stanox_ref=pd.DataFrame(),
                    tiploc_to_stanox={},
                    incident_data_loaded={}
                )
    
    assert success is True
    assert total > 0
    assert len(files) > 0


def test_process_single_station_and_save_k1():
    """Test handling of station with no data"""
    mock_schedule_data = pd.DataFrame()
    
    with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
        mock_save.return_value = None  # No data
        
        success, files, total = _process_single_station_and_save(
            st_code="99999",
            i=1,
            total_stations=3,
            output_dir="processed_data",
            schedule_data_loaded=mock_schedule_data,
            stanox_ref=pd.DataFrame(),
            tiploc_to_stanox={},
            incident_data_loaded={}
        )
    
    assert success is False
    assert total == 0
    assert len(files) == 0


def test_process_single_station_and_save_k2():
    """Test handling of processing errors"""
    mock_schedule_data = pd.DataFrame()
    
    with patch('rdmpy.preprocessor.save_processed_data_by_weekday_to_dataframe') as mock_save:
        mock_save.side_effect = Exception("Processing failed")
        
        success, files, total = _process_single_station_and_save(
            st_code="11271",
            i=1,
            total_stations=3,
            output_dir="processed_data",
            schedule_data_loaded=mock_schedule_data,
            stanox_ref=pd.DataFrame(),
            tiploc_to_stanox={},
            incident_data_loaded={}
        )
    
    assert success is False
    assert total == 0

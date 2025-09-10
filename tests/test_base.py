import json

import pytest

from soilgen_ai.base import Soil, SoilBorehole


# --- Fixtures for Mock Data ---
@pytest.fixture
def clay_soil() -> Soil:
    """Provides a sample Stiff Clay Soil object."""
    properties = {"Vs": 300, "Density": 1900, "Poisson": 0.3}
    return Soil(name_definition="Stiff Clay", properties=properties)


@pytest.fixture
def sand_soil() -> Soil:
    """Provides a sample Dense Sand Soil object."""
    properties = {"Vs": 450, "Density": 2000, "Poisson": 0.25}
    return Soil(name_definition="Dense Sand", properties=properties)


@pytest.fixture
def sample_borehole(clay_soil, sand_soil) -> SoilBorehole:
    """Provides a sample SoilBorehole using other fixtures."""
    layers = [
        {"soil": clay_soil, "depth": 10.0},
        {"soil": sand_soil, "depth": 25.0},
    ]
    return SoilBorehole(layers=layers)


# --- Tests for Soil Class ---
def test_soil_initialization(clay_soil):
    """Tests if the Soil object is initialized correctly."""
    assert clay_soil.name_definition == "Stiff Clay"
    assert clay_soil.properties["Vs"] == 300
    assert clay_soil.properties["Density"] == 1900


def test_soil_to_dict(clay_soil):
    """Tests the dictionary representation of a Soil object."""
    expected_dict = {
        "name_definition": "Stiff Clay",
        "properties": {"Vs": 300, "Density": 1900, "Poisson": 0.3},
    }
    assert clay_soil.to_dict() == expected_dict


def test_properties_from_file(tmp_path):
    """
    Tests loading properties from a JSON file using pytest's tmp_path fixture
    to create a temporary file.
    """
    soil_data = {
        "description": "Sample soil data",
        "properties": {"Vs": 500, "Density": 2100},
    }
    # Create a temporary file in a temporary directory
    file_path = tmp_path / "soil.json"
    file_path.write_text(json.dumps(soil_data))

    # Run the method and check the result
    properties = Soil.properties_from_file(file_path)
    assert properties == {"Vs": 500, "Density": 2100}


def test_properties_from_file_no_properties_key(tmp_path):
    """
    Tests the case where the JSON file is valid but lacks the 'properties' key.
    It should return an empty dictionary.
    """
    soil_data = {"description": "Data without a properties key"}
    file_path = tmp_path / "soil_no_props.json"
    file_path.write_text(json.dumps(soil_data))

    properties = Soil.properties_from_file(file_path)
    assert properties == {}


# --- Tests for SoilBorehole Class ---
def test_borehole_initialization(sample_borehole, clay_soil):
    """Tests if the SoilBorehole object is initialized correctly."""
    assert len(sample_borehole.layers) == 2
    assert sample_borehole.layers[0]["soil"] == clay_soil
    assert sample_borehole.layers[0]["depth"] == 10.0


def test_borehole_total_depth(sample_borehole):
    """Tests the total_depth property calculation."""
    assert sample_borehole.total_depth == 25.0


def test_borehole_total_depth_empty():
    """Tests the total_depth property for a borehole with no layers."""
    empty_borehole = SoilBorehole(layers=[])
    assert empty_borehole.total_depth == 0.0


def test_borehole_total_depth_single_layer(clay_soil):
    """Tests the total_depth property for a borehole with a single layer."""
    single_layer_borehole = SoilBorehole(layers=[{"soil": clay_soil, "depth": 15.0}])
    assert single_layer_borehole.total_depth == 15.0


def test_borehole_to_dict(sample_borehole, clay_soil, sand_soil):
    """
    Tests the dictionary representation of a SoilBorehole, checking the integration
    between SoilBorehole and Soil objects.
    """
    expected_dict = {
        "layers": [
            {
                "soil": clay_soil.to_dict(),  # Uses the Soil object's method
                "depth": 10.0,
            },
            {
                "soil": sand_soil.to_dict(),
                "depth": 25.0,
            },
        ],
        "total_depth": 25.0,
    }
    assert sample_borehole.to_dict() == expected_dict

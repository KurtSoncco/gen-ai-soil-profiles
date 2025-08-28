import json
from typing import Any, Dict, List


class Soil:
    """
    Represents a soil type with its name and properties.

    Attributes:
        name_definition (str): The name or definition of the soil type.
        properties (Dict[str, Any]): A dictionary of soil properties.
            Examples of properties include:
            - Shear Wave velocity (Vs), in meters/second
            - Compressional Wave velocity (Vp), in meters/second
            - Density (ρ), in kilograms/cubic meter
            - Poisson's ratio (ν)
    """

    def __init__(self, name_definition: str, properties: Dict[str, Any]):
        self.name_definition = name_definition
        self.properties = properties

    @staticmethod
    def properties_from_file(file: str) -> Dict[str, Any]:
        """Loads soil properties from a JSON file."""
        with open(file, "r") as f:
            data = json.load(f)
        return data.get("properties", {})

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the Soil object."""
        return {"name_definition": self.name_definition, "properties": self.properties}


class SoilBorehole:
    """
    Represents a borehole profile, which is a sequence of soil layers.

    Each layer in the borehole is defined by a Soil object and its depth.

    Attributes:
        layers (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary represents a soil layer with its 'soil' type and 'depth'.
    """

    def __init__(self, layers: List[Dict[str, Any]]):
        self.layers = layers

    @property
    def total_depth(self) -> float:
        """Calculates the total depth of the borehole."""
        if not self.layers:
            return 0.0
        return max(layer.get("depth", 0.0) for layer in self.layers)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the SoilBorehole object."""
        return {
            "layers": [
                {
                    "soil": layer["soil"].to_dict(),
                    "depth": layer["depth"],
                }
                for layer in self.layers
            ],
            "total_depth": self.total_depth,
        }

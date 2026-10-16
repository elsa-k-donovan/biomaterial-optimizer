"""Shared test data. pytest loads this file automatically."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def literature_dir() -> Path:
    return REPO / "data" / "literature"


@pytest.fixture
def aliases_path() -> Path:
    return REPO / "data" / "aliases.yaml"


@pytest.fixture
def materiom_recipe() -> dict:
    """A small recipe in the same shape as Materiom's API returns."""
    return {
        "recipe": {
            "_id": "abc123",
            "data": {
                "name": "Eggshell bio-ceramic",
                "group_id": "g1",
                "details": {
                    "sources": [{"source": "https://doi.org/10.1000/Example.1"}]
                },
                "ingredients": [
                    {
                        "label": "Eggshell powder",
                        "value": {"amount": "60"},
                        "unit": {"value": "g", "label": "grams"},
                    },
                    {
                        "label": "Sodium Alginate",
                        "value": {"amount": "5"},
                        "unit": {"value": "g", "label": "grams"},
                    },
                ],
                "physicalProperties": {
                    "properties": [
                        {
                            "name": "Flexural Modulus",
                            "items": [{"value": 1477, "unit": "Mpa"}],
                        }
                    ]
                },
                "mechanicalTensionProperties": {
                    "properties": [
                        {
                            "name": "Tensile Strength",
                            "items": [{"value": "43.9 ± 2.3", "unit": "MPa"}],
                        },
                        {
                            "name": "Elongation at Break (%)",
                            "items": [{"value": "3.2 ± 0.2", "unit": ""}],
                        },
                        {"name": "Yield Strength", "items": []},  # empty template slot
                    ]
                },
                "opticalProperties": {
                    "properties": [
                        {
                            "name": "UV transmittance",
                            "items": [{"value": "<0.1", "unit": "%"}],
                        }
                    ]
                },
            },
        }
    }

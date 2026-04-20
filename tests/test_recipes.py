"""Recipe definitions — smoke tests."""

import pytest

from samhita.core.recipes import RECIPES, get_recipe, list_recipes
from samhita.core.schemas import EntityType, RecipeName, RelationType


def test_both_recipes_registered() -> None:
    assert RecipeName.DRUG_TARGET_DISEASE in RECIPES
    assert RecipeName.DISEASE_GENE_PATHWAY in RECIPES
    assert len(list_recipes()) == 2


def test_drug_target_disease_shape() -> None:
    r = get_recipe(RecipeName.DRUG_TARGET_DISEASE)
    assert r["name"] == RecipeName.DRUG_TARGET_DISEASE
    assert EntityType.DRUG in r["entity_types"]
    assert EntityType.DISEASE in r["entity_types"]
    assert RelationType.TARGETS in r["relation_types"]
    assert RelationType.TREATS in r["relation_types"]


def test_disease_gene_pathway_shape() -> None:
    r = get_recipe(RecipeName.DISEASE_GENE_PATHWAY)
    assert EntityType.PATHWAY in r["entity_types"]
    assert RelationType.PART_OF in r["relation_types"]


def test_unknown_recipe_raises() -> None:
    with pytest.raises(KeyError):
        get_recipe("nonsense")  # type: ignore[arg-type]

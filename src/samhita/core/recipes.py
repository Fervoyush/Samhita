"""Schema recipes for v1.

A recipe bundles (entity types, relation types, sources) so the planner
can map a natural-language request onto a concrete KGSpec without
inventing entity/relation vocabularies on the fly.
"""

from __future__ import annotations

from typing import TypedDict

from samhita.core.schemas import EntityType, RecipeName, RelationType, SourceName


class Recipe(TypedDict):
    name: RecipeName
    description: str
    entity_types: list[EntityType]
    relation_types: list[RelationType]
    sources: list[SourceName]


DRUG_TARGET_DISEASE: Recipe = {
    "name": RecipeName.DRUG_TARGET_DISEASE,
    "description": (
        "Drugs and their molecular targets, mapped to the diseases they treat or are "
        "indicated for. Useful for drug repurposing, mechanism-of-action exploration, "
        "and target identification."
    ),
    "entity_types": [
        EntityType.DRUG,
        EntityType.GENE,
        EntityType.PROTEIN,
        EntityType.DISEASE,
    ],
    "relation_types": [
        RelationType.TARGETS,
        RelationType.INHIBITS,
        RelationType.ACTIVATES,
        RelationType.TREATS,
        RelationType.INDICATED_FOR,
    ],
    "sources": [
        SourceName.PUBMED_CENTRAL,
        SourceName.OPENTARGETS,
        SourceName.CHEMBL,
        SourceName.DRUGBANK,
    ],
}


DISEASE_GENE_PATHWAY: Recipe = {
    "name": RecipeName.DISEASE_GENE_PATHWAY,
    "description": (
        "Disease-associated genes and the biological pathways they participate in. "
        "Useful for biomarker exploration, understanding disease mechanisms, and "
        "identifying druggable pathways."
    ),
    "entity_types": [
        EntityType.DISEASE,
        EntityType.GENE,
        EntityType.PROTEIN,
        EntityType.PATHWAY,
    ],
    "relation_types": [
        RelationType.ASSOCIATED_WITH,
        RelationType.CAUSES,
        RelationType.PART_OF,
        RelationType.REGULATES,
        RelationType.INTERACTS_WITH,
    ],
    "sources": [
        SourceName.PUBMED_CENTRAL,
        SourceName.OPENTARGETS,
    ],
}


RECIPES: dict[RecipeName, Recipe] = {
    RecipeName.DRUG_TARGET_DISEASE: DRUG_TARGET_DISEASE,
    RecipeName.DISEASE_GENE_PATHWAY: DISEASE_GENE_PATHWAY,
}


def get_recipe(name: RecipeName) -> Recipe:
    if name not in RECIPES:
        available = ", ".join(sorted(r.value for r in RECIPES))
        raise KeyError(f"Unknown recipe: {name!r}. Available: {available}")
    return RECIPES[name]


def list_recipes() -> list[Recipe]:
    return list(RECIPES.values())

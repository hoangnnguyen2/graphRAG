from pydantic import Field, BaseModel
from typing import List

class Entity(BaseModel):
    entity_name: str = Field(
        description=(
            "Canonical normalized entity name used for deduplication. "
            "Lowercase, stripped, and normalized where appropriate."
        )
    )
    entity_type: str = Field(
        description=(
            "Entity category, for example PERSON, ORGANIZATION, LOCATION, "
            "CONCEPT, EVENT, METHOD, TECHNOLOGY, DISEASE, DRUG, OTHER."
        )
    )

class Relationship(BaseModel):
    source: str = Field(
        description="entity_name of the source entity."
    )
    target: str = Field(
        description="entity_name of the target entity."
    )
    relationshipType: str = Field(
        description=(
            "Short semantic relationship type such as TREATS, CAUSES, "
            "PART_OF, DEVELOPED_BY, LOCATED_IN, ASSOCIATED_WITH."
        )
    )
    weight: float = Field(
            default=1.0,
            ge=0.0,
            description=(
                "Relationship confidence or strength. "
                "Normally use 1.0 for explicitly stated relationships."
            )
    )

class GraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

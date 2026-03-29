from pydantic import BaseModel, Field
from typing import Optional

# -----------------------------------------------------------------------
# Define the data schema for layer 1 and layer 2
# -----------------------------------------------------------------------

class PartSchema(BaseModel):
    family: str = Field(
        description="Mechanical part family, e.g. spur gear / Stirnrad, bevel gear/Kegelrad, etc.. Extract only if available from title/heading."
    )
    spur_gear_material: str = Field(description="The material from which the spur gear was manufactured (e.g. Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc.)")
    module: float = Field(description="The gear module of a gear represents the ratio of the pitch (distance between teeth) to pi (\\(\\pi \\)), effectively defining how thick a gear tooth is and, consequently, how strong it is.")
    straight_toothed: bool = Field(description="It indicates whether, the teeth are aligned longitudinally with the shaft, meaning there is no \"helix angle\".")
    angle_of_engagement: int = Field(description="It refers to the angular position, or the arc, during which two gear teeth are in contact and transmitting power. It is often written in Degrees (°).")

class TableRowSchema(BaseModel):
    ZZ: float = Field(description="ZZ (German for Zähnezahl) represents the number of teeth of the spur gear")
    ZB: float = Field(description="ZB (German for Zahn-breite) represents the width of the spur gear")
    ØB: float = Field(description="Represents the inner diameter of the spur gear")
    ØTK: float = Field(description="Represents the Pitch circle diameter of the spur gear")
    ØKK: float = Field(description="Represents the tip diameter of the spur gear")
    ØN: float = Field(description="Represents the Hub diameter of the spur gear")
    L: float = Field(description="The length of the spur gear")
    ØFM: float = Field(description="Represents the diameter of the ring gear")
    WS: float = Field(description="Represents the girder width")
    G: float = Field(description="The weight of the spur gear indicated in unit of gramms ([g]).")
    DM: float = Field(description="Represents the maximum permissible torque applied to the indicated in ([Ncm]).")
    art_nr: str = Field(description="'Art.-Nr.' is an abbreviation for the German term Artikelnummer, which translates to Article Number. It distinguishes a particular rack based on its specifications.")

    # extend layer 2
    family: str = Field(
        description="Mechanical part family, e.g. spur gear / Stirnrad, bevel gear/Kegelrad, etc.. Extract only if available from title/heading."
    )
    module: float = Field(
        description="Gear module for the mechanical part this row belongs to, e.g. 0.5, 0.7, 1.0."
    )
    spur_gear_material: str = Field(
        description="Material of the spur gear for this row, Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc."
    )

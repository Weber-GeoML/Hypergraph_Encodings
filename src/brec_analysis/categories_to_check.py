# Define valid categories with their ranges
VALID_CATEGORIES = [
    ("basic", (0, 60), "Basic"),
    ("regular", (60, 85), "Regular"),
    ("str", (85, 135), "Strongly Regular"),
    ("cfi", (135, 185), "CFI"),
    ("extension", (185, 235), "Extension"),
    ("4vtx", (235, 255), "4-Vertex Condition"),
    ("dr", (255, 275), "Distance Regular"),
]


# for the ones I fetched myself
PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 85),
    "strongly regular": (85, 135),
    "CFI": (135, 185),
    "Extension": (185, 235),
    "4-Vertex_Condition": (235, 255),
    "Distance_Regular": (255, 275),
}

"""Categories to check for BREC dataset"""

# Define valid categories with their ranges
# (0,60) is number of pairs (ie 120 graphs)
VALID_CATEGORIES = [
    ("basic", (0, 60), "Basic"),
    ("regular", (60, 110), "Regular"),
    ("str", (110, 160), "Strongly Regular"),
    ("cfi", (160, 260), "CFI"),
    ("extension", (260, 360), "Extension"),
    ("4vtx", (360, 380), "4-Vertex Condition"),
    ("dr", (380, 400), "Distance Regular"),
]


# for the ones I fetched myself
PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 110),
    "Strongly_Regular": (110, 160),
    "CFI": (160, 260),
    "Extension": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}

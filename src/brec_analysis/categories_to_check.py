"""Categories to check for BREC dataset."""

# Define valid categories with their ranges
# (0,60) is number of pairs (ie 120 graphs)
VALID_CATEGORIES = [
    ("basic", (0, 56), "Basic"),  # 56 pairs
    ("regular", (56, 106), "Regular"),  # 50 pairs
    ("str", (106, 156), "Strongly Regular"),  # 50 pairs
    ("cfi", (156, 253), "CFI"),  # 97 pairs
    ("extension", (253, 350), "Extension"),  # 97 pairs
    ("4vtx", (350, 370), "4-Vertex Condition"),  # 20 pairs
    ("dr", (370, 390), "Distance Regular"),  # 20 pairs
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

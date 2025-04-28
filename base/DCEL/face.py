class Face:
    def __init__(self):
        self.outer_component = None  # 指向该面上任一半边

    def __repr__(self):
        return f"Face(outer_component={self.outer_component})"

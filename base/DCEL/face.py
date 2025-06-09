class Face:
    def __init__(self):
        self.outer_component = None  # 指向该面上任一半边
        self.neighbourhoods = []
        self.visit = None
    def __repr__(self):
        return f"Face(outer_component={self.outer_component,self.neighbourhoods})"

    # --- add these three methods ---
    def __hash__(self):
        # use the Python object id as a stable hash
        return id(self)

    def __eq__(self, other):
        # two Face objects are equal iff they are the same instance
        return self is other

    def __lt__(self, other):
        # define a total order so we can sort them
        return id(self) < id(other)
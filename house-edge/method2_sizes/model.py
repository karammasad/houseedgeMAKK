"""Method 2 model: Method 1's gate with a size model chosen here. Person B edits sizes.py and this file."""
from method1_gate.model import Model as GateModel


class Model(GateModel):
    def __init__(self, sizes="npmle", **kw):
        super().__init__(sizes=sizes, **kw)

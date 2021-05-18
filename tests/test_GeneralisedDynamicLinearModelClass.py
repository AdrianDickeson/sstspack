import unittest

from sstspack.GeneralisedDynamicLinearModelClass import (
    GeneralisedDynamicLinearModel as GDLM,
)


class TestGeneralisedDynamicLinearModel(unittest.TestCase):
    def SetupLocalModel(self):
        pass

    def test__init__(self):
        local_model = GDLM()


if __name__ == "__main__":
    unittest.main()
import unittest

from app.services.vision import _class_name, _names_to_list


class VisionServiceTests(unittest.TestCase):
    def test_model_names_support_dicts_and_sequences(self) -> None:
        self.assertEqual(_class_name({0: "person"}, 0), "person")
        self.assertEqual(_class_name(["head", "helmet"], 1), "helmet")
        self.assertEqual(_names_to_list(("head", "helmet")), ["head", "helmet"])


if __name__ == "__main__":
    unittest.main()

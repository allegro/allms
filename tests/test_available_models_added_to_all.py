# import html.entities
# from unittest.mock import patch
#
# import pytest
from allms import models


class TestAvailableModelsAddedToAll:

    def test_available_models_added_to_all(self):
        for model in models.get_available_models().values():
            assert model.__name__ in models.__all__

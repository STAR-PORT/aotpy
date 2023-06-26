# TODO: Add module docstring

from collections import namedtuple
from . import AOFTranslator, CIAOTranslator, ERISTranslator, NAOMITranslator


InitParameter = namedtuple("InitParameter", ["name", "type"])


def get_available_translators() -> dict:
    # TODO: Add docstring
    # TODO: Define type for available_translators
    available_translators = {"naomi": {"cls": NAOMITranslator,
                                       "params": [InitParameter(name="path", type=str),
                                                  InitParameter(name="at_number", type=int)]},
                             "ciao": {"cls": CIAOTranslator,
                                      "params": [InitParameter(name="path", type=str),
                                                 InitParameter(name="at_number", type=int)]},
                             "eris": {"cls": ERISTranslator,
                                      "params": [InitParameter(name="path", type=str)]},
                             "aof": {"cls": AOFTranslator,
                                     "params": [InitParameter(name="path_lgs", type=str),
                                                InitParameter(name="path_ir", type=str),
                                                InitParameter(name="path_pix", type=str)]}
                             }
    return available_translators

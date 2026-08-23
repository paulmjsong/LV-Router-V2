import io
import pickle

import pytest

from app.gist_regulations import _RestrictedUnpickler


class Dangerous:
    def __reduce__(self):
        return (eval, ("1+1",))


def test_restricted_unpickler_rejects_unapproved_globals() -> None:
    payload = pickle.dumps(Dangerous())
    with pytest.raises(pickle.UnpicklingError):
        _RestrictedUnpickler(io.BytesIO(payload)).load()

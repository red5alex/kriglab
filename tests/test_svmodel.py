import svmodels
import pytest


def test_instance_of():
    svmodel = svmodels.instance_of(svmodels.exponential, a=50, C0=0.11, Cn=0.00)

    assert pytest.approx(svmodel(2), 0.0695)


def test_svm2cvm():
    svmodel = svmodels.instance_of(svmodels.exponential, a=50, C0=0.11, Cn=0.00)
    cvmodel = svmodels.svm2cvm(svmodel)

    assert pytest.approx(cvmodel(2), 0.0405)

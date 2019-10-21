import svmodels


def test_instance_of():
    res = svmodels.instance_of(svmodels.exponential, a=50,
                         C0=0.11, Cn=0.00)
    print(res)


import numpy as np
from math import pi
import pandas as pd

from svmodels import hole_New, SV
from correlation_models import cvmodel


def test_cvmodel():
    varmodel = hole_New
    C0 = 100.0
    bw = 25
    hs = np.arange(0, 200, bw)
    df_z = pd.read_excel(r".\example_data\Periodic Sine\example_2_modified.xlsx")
    P = np.array(df_z.dropna()[['x', 'y', 'truez', 'noise']])
    sv = SV(P, hs, bw)

    sp = cvmodel(P, model=varmodel, hs=np.arange(0, 200, bw), bw=bw, Cn=0.0, svrange=250/(2*pi), C0=C0)
    sp(sv[0])

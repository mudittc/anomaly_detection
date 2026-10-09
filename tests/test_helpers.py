import pandas as pd
from astro_anomaly import find_coord_columns, robust_median_scores

def test_find_coordinates():
    df = pd.DataFrame({"RAJ2000": [1.0], "DEJ2000": [2.0], "flux": [3.0]})
    assert find_coord_columns(df) == ("RAJ2000", "DEJ2000")

def test_robust_scores_detect_outlier():
    df = pd.DataFrame({"flux": [10, 10, 11, 9, 10, 10, 12, 500]})
    result = robust_median_scores(df, ["flux"], min_group_size=5, z_threshold=4)
    assert result.loc[7, "robust_max_abs_z"] > result.loc[0, "robust_max_abs_z"]

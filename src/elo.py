TEAMMATE_WEIGHT = 0.8
OVERALL_POSITION_WEIGHT = 0.2
OVERALL_POSITION_SCALE = 100.0


def expected_position_diff(Ra, Rb, psf=500):
    """Return the expected finishing-place gap from the rating difference."""
    return (Ra - Rb) / psf


def calculate_new_elo(Ra, Rb, position_A, position_B, k=5, psf=500):
    """Apply the project's existing teammate, finishing-margin update rule."""
    expected_diff = expected_position_diff(Ra, Rb, psf=psf)
    actual_diff = position_B - position_A
    diff_factor = abs(actual_diff - expected_diff)
    if actual_diff > 0:
        return Ra + k * diff_factor, Rb - k * diff_factor
    else:
        return Ra - k * diff_factor, Rb + k * diff_factor

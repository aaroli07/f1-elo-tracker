def expected_position_diff(Ra, Rb, psf=500):
    return (Ra - Rb) / psf

def calculate_new_elo(Ra, Rb, position_A, position_B, k=5,psf=500):
    expected_diff = expected_position_diff(Ra, Rb, psf=psf)
    actual_diff = position_B - position_A
    diff_factor = abs(actual_diff - expected_diff)
    if actual_diff > 0:
        return Ra + k * diff_factor, Rb - k * diff_factor
    else:
        return Ra - k * diff_factor, Rb + k * diff_factor

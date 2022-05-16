import matplotlib.pyplot as plt
import numpy as np

LIMIT_X_UPPER = 5
LIMIT_X_LOWER = -5
SLOPE_ASPHALT = (-6 - 6) / (LIMIT_X_UPPER-LIMIT_X_LOWER)  # delta_y / delta_x
SLOPE_ICE = (-1 - 1) / (LIMIT_X_UPPER-LIMIT_X_LOWER)  # delta_y / delta_x


def main():
    print("Slope asphalt:", SLOPE_ASPHALT, "SLOPE_ICE:", SLOPE_ICE)
    x_range = np.linspace(LIMIT_X_LOWER, LIMIT_X_UPPER, 100)
    y_range_asp = x_range*SLOPE_ASPHALT
    y_range_ice = x_range*SLOPE_ICE

    plt.figure()
    plt.plot(x_range, y_range_asp, label="Asphalt")
    plt.plot(x_range, y_range_ice, label="Ice")
    plt.legend()
    plt.xlabel("Tire Slip Angle (alpha) [deg]")
    plt.ylabel("Lateral Tire Force (F_y) [kN]")
    plt.show()

if __name__ == "__main__":
    main()

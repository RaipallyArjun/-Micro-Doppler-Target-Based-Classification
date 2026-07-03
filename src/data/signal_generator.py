import numpy as np


def generate_drone_signal(duration=1.0, fs=1000):
    """
    Generate a synthetic drone micro-Doppler signal.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    carrier = np.sin(2 * np.pi * 50 * t)
    blade1 = 0.5 * np.sin(2 * np.pi * 120 * t)
    blade2 = 0.3 * np.sin(2 * np.pi * 180 * t)

    signal = carrier + blade1 + blade2

    noise = np.random.normal(0, 0.05, len(signal))

    return signal + noise


def generate_bird_signal(duration=1.0, fs=1000):
    """
    Generate a synthetic bird micro-Doppler signal.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    wingbeat = np.sin(2 * np.pi * 20 * t)
    harmonic = 0.4 * np.sin(2 * np.pi * 40 * t)

    signal = wingbeat + harmonic

    noise = np.random.normal(0, 0.05, len(signal))

    return signal + noise


if __name__ == "__main__":

    drone = generate_drone_signal()

    bird = generate_bird_signal()

    print("Drone Signal Shape :", drone.shape)

    print("Bird Signal Shape :", bird.shape)
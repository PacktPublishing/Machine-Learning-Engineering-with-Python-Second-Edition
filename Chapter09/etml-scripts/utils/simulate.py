"""
Simulate taxi ride data for clustering.

"""
import logging
import numpy as np
import pandas as pd
import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))


# Define simulate ride data function
def simulate_ride_distances():
    logging.info("Simulating ride distances ...")
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10),  # long distances
            10 * np.random.random(size=10),  # same distance
            10 * np.random.random(size=10),  # same distance
        )
    )
    return ride_dists


def simulate_ride_speeds():
    logging.info("Simulating ride speeds ...")
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10),  # same speed
            np.random.normal(loc=50, scale=10, size=10),  # high speed
            np.random.normal(loc=15, scale=4, size=10),  # low speed
        )
    )
    return ride_speeds


def simulate_ride_data():
    logging.info("Simulating ride data ...")
    # Simulate some ride data ...
    ride_dists = simulate_ride_distances()
    ride_speeds = simulate_ride_speeds()
    ride_times = ride_dists / ride_speeds

    # Assemble into Data Frame
    df = pd.DataFrame(
        {"ride_dist": ride_dists, "ride_time": ride_times, "ride_speed": ride_speeds}
    )
    ride_ids = datetime.datetime.now().strftime("%Y%m%d") + df.index.astype(str)
    df["ride_id"] = ride_ids

    df = simulate_text_data(df)

    return df


def simulate_text_data(df: pd.DataFrame) -> pd.DataFrame:
    example_news = [
        f"""
        Reports are that there has been an accident on the M8 motorway near Glasgow due to icy conditions on the roads. No one has been seriously injured but the road is blocked and traffic is backed up for miles. The police are on the scene and are expected to aid the clearing of the scene in the next few hours.
        """,
        f"""
        It is expected to be a busy shopping day today as many retailers attempt are offering discounts to try to lure shoppers back into city centre stores after the COVID-19 pandemic and lockdown. Many are expected to make there way into Glasgow city centre today to take advantage of the discounts on offer. There is also an expected surge in activity on online shopping sites.
        """,
        f"""
        Economic conditions are slowly improving for the West of Scotland after a series of targeted investments in the area. Some high profile companies from across the globe have been lured to the Greater Glasgow Area after a targeted campaign by the Scottish Government to attract more investment in the area. The main pitch laid out to the investors has been centered around Scotland's excellent education system and it's very high quality of life for employees and customers.
        """,
    ]

    example_weather = [
        f"""
        The forecast for the West of Scotland over the next few days is for cold and wet weather with icy conditions to remain. Drivers are advised to only make journeys where absolutely necessary, especially since the current cold spell has already led to a number of accidents across the region.
        """,
        f"""
        The weather is expected to be sunny and dry over the next few days, with temperatures in the mid-teens looking set to entice people out and about.
        """,
        f"""
        The forecast for the Greater Glasgow Area today remains overcast with a chance of showers and mild winds. 
        """,
    ]

    example_traffic = [
        f"""
        There is a traffic jam on the M8 motorway near Glasgow. Expect delays.
        """,
        f"""
        Traffic is expected to be heavy on the M8 motorway near Glasgow today due to an influx of shoppers into the city centre.
        """,
        f"""
        Traffic is expected to be normal today in the Greater Glasgow Area.
        """,
    ]

    # Need to select consistent examples for each row
    selection_idx = rs.choice(3, len(df))
    df["selection_idx"] = selection_idx
    df["news"] = df["selection_idx"].apply(lambda x: example_news[x].strip())
    df["weather"] = df["selection_idx"].apply(lambda x: example_weather[x].strip())
    df["traffic"] = df["selection_idx"].apply(lambda x: example_traffic[x].strip())

    return df


if __name__ == "__main__":
    import os

    # If data present, read it in
    # file_path = 'taxi-rides.csv'
    date = datetime.datetime.now().strftime("%Y%m%d")
    file_path = f"../data/taxi-rides-{date}.json"
    if os.path.exists(file_path):
        # df = pd.read_csv(file_path)
        pass
    else:
        logging.info("Simulating ride data")
        df = simulate_ride_data()
        # df.to_csv(file_path, index=False)
        df.to_json(file_path, orient="records")

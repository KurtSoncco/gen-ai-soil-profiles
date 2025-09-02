def depth_processing(df):
    df = df[df["depth"] < 2000]
    return df


def vs_max_processing(df):
    index_vals = df[df["vs_value"] >= 2500]["velocity_metadata_id"].unique()
    df = df[~df["velocity_metadata_id"].isin(index_vals)]
    return df


def preprocessing(df):
    df = depth_processing(df)
    df = vs_max_processing(df)
    df = df.reset_index(drop=True)
    return df

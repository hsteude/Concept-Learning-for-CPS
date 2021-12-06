from solution1.datagen.data_gen import ThreeTankDataGenerator
import solution1.constants as const


if __name__ == "__main__":
    import pandas as pd
    ttdg = ThreeTankDataGenerator()
    x, time, uid_ts_sample, q1, q3, kv12, kv23 = ttdg.generate()
    df = pd.DataFrame(x, columns=const.STATE_COL_NAMES)
    df[const.TIME_COL_NAME] = time
    df[const.UID_SAMPLE_COL_NAME] = uid_ts_sample
    df_meta = pd.DataFrame(
        {
            const.Q1_COL_NAME: q1,
            const.Q3_COL_NAME: q3,
            const.KV12_COL_NAME: kv12,
            const.KV23_COL_NAME: kv23,
            const.UID_SAMPLE_COL_NAME: list(range(const.NUMBER_OF_SAMPLES)),
        }
    )
    df = df.merge(df_meta, how="left", on=const.UID_SAMPLE_COL_NAME)
    df.to_parquet(const.DATA_PATH)

import solution2.constants as const
from solution2.datagen.data_gen import ThreeTankDataGenerator


if __name__ == '__main__':
    import pandas as pd
    ttdg = ThreeTankDataGenerator()
    x, time, uid_ts_sample, q1, q3, kv12, kv23, \
        ans_question_1, ans_question_2, ans_question_3, ans_question_4 \
        = ttdg.generate()
    df = pd.DataFrame(x, columns=const.STATE_COL_NAMES)
    df[const.TIME_COL_NAME] = time
    df[const.UID_SAMPLE_COL_NAME] = uid_ts_sample
    df_meta = pd.DataFrame({const.Q1_COL_NAME: q1, const.Q3_COL_NAME: q3,
                            const.KV12_COL_NAME: kv12, const.KV23_COL_NAME: kv23,
                            const.A1_COLNAME: ans_question_1,
                            const.A2_COLNAME: ans_question_2,
                            const.A3_COLNAME: ans_question_3,
                            const.A4_COLNAME: ans_question_4,
                            const.UID_SAMPLE_COL_NAME: list(range(const.NUMBER_OF_SAMPLES))})
    df = df.merge(df_meta, how='left', on=const.UID_SAMPLE_COL_NAME)

    df.to_parquet(const.DATA_PATH)

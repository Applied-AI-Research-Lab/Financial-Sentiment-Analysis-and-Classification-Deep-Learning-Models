import pandas as pd

print('Model,Mean Prediction Time,Total Time (for 1,000 complaints)')
def sum_and_average_column(dataset: str, column: str) -> tuple:
    df = pd.read_csv(dataset)
    total = round(df[column].sum(), 2)
    average = round(df[column].mean(), 2)
    return column + ',' + str(average) + ',' + str(total)
    # return column, total, average

print(sum_and_average_column("../Datasets/test_set.csv", "time-gpt-4o"))
print(sum_and_average_column("../Datasets/test_set.csv", "time-ft:gpt-4o-2024-08-06:personal::B7K5Hmps"))
print(sum_and_average_column("../Datasets/test_set.csv", "time-gpt-4o-mini"))
print(sum_and_average_column("../Datasets/test_set.csv", "time-ft:gpt-4o-mini-2024-07-18:personal::B7IZokNn"))
print(sum_and_average_column("../Datasets/test_set.csv", "time-bert"))
print(sum_and_average_column("../Datasets/test_set.csv", "time-bert-adamw"))
print(sum_and_average_column("../Datasets/test_set.csv", "time_svm"))
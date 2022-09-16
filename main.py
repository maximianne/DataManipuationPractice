# ----- IMPORTS ----- #
import pandas as pd
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def data_info(dataframe):
    # head function, used to display the first 5 rows,
    # you can change the value by putting a number in the parenthesis
    print("the head: \n: ")
    print(dataframe.head())
    # tail does the same, but the end
    print("\n the tail: \n: ")
    print(dataframe.tail())

    # shape, this shows the dimension of the dataframe
    print("\n the dimension (rows, columns): \n: ")
    print(dataframe.shape)

    # info, this shows the columns names, number of non-nulls, type etc.
    print("\n the info: \n: ")
    print(dataframe.info)

    # also, just to get columns names:
    print("\n the column names: \n: ")
    print(dataframe.columns)

    # types of each column
    print("\n the types of each columns are: \n: ")
    print(dataframe.dtypes)

def data_manipulation(dataframe):
    # to datetime, in this specific instance, we contain date data, so we can turn the "stop_date"
    # column into a date_time. When you convert it to a datetime, that means that you can
    # manipulate using time functions
    dataframe['stop_date'] = pd.to_datetime(dataframe['stop_date'])
    dataframe['stop_time'] = pd.to_datetime(dataframe['stop_time'], format="%H:%M").dt.time
    print("\n the column names: \n: ")
    print(dataframe.columns)

    # is null and sum, this is used to determine the sum of the column if it is a num type
    # Or you can get the total of nulls in a column
    print("\n The number of nulls in each column \n")
    print(dataframe.isnull().sum())
    # Or to find a specific columns number of null:
    print("\n The column driver_gender total number of nulls:")
    print(dataframe['driver_gender'].isnull().sum())

    # once we see the nulls, we can drop the columns that are completely empty
    dataframe.drop(['county_name'], axis=1, inplace=True)
    # or we can do dataframe= dataframe.drop(columns= dataframe.columns[1])

    # we can also drop rows:
    # dataframe = dataframe.drop(labels=1, axis=0)

    # we can drop entire rows when they contain nulls in certain columns
    dataframe = dataframe.dropna(subset=['stop_time'])

    # describe
    print("\n the description of the dataframe:")
    print(dataframe.describe())

def violations_per_race():
    race_violations = df[['driver_race', 'violation']]
    race_violations = race_violations.dropna()
    visited = []
    for i in range(0, len(race_violations['violation'])):
        if race_violations['violation'].iloc[i] not in visited:
            visited.append(race_violations['violation'].iloc[i])
    print(visited)

    asian = race_violations.groupby('driver_race').get_group('Asian')

    arr = [0] * len(visited)
    for i in range(0, len(asian["violation"])):
        check = asian['violation'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.bar(visited, arr)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-small')
    plt.title('Ethnicity: Asian, Violations Frequency')
    plt.xlabel('Violation')
    plt.ylabel('Quantity')
    plt.show()

    white = race_violations.groupby('driver_race').get_group('White')

    arr = [0] * len(visited)
    for i in range(0, len(white["violation"])):
        check = white['violation'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.bar(visited, arr)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-small')
    plt.title('Ethnicity: White, Violations Frequency')
    plt.xlabel('Violation')
    plt.ylabel('Quantity')
    plt.show()

    black = race_violations.groupby('driver_race').get_group('Black')

    arr = [0] * len(visited)
    for i in range(0, len(black["violation"])):
        check = black['violation'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.bar(visited, arr)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-small')
    plt.title('Ethnicity: Black, Violations Frequency')
    plt.xlabel('Violation')
    plt.ylabel('Quantity')
    plt.show()

    hispanic = race_violations.groupby('driver_race').get_group('Hispanic')

    arr = [0] * len(visited)
    for i in range(0, len(hispanic["violation"])):
        check = hispanic['violation'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.bar(visited, arr)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-small')
    plt.title('Ethnicity: Hispanic, Violations Frequency')
    plt.xlabel('Violation')
    plt.ylabel('Quantity')
    plt.show()

    other = race_violations.groupby('driver_race').get_group('Other')

    arr = [0] * len(visited)
    for i in range(0, len(other["violation"])):
        check = other['violation'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.bar(visited, arr)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-small')
    plt.title('Ethnicity: Other, Violations Frequency')
    plt.xlabel('Violation')
    plt.ylabel('Quantity')
    plt.show()

def violations_and_race():
    race_violations = df[['driver_race', 'violation']]
    race_violations = race_violations.dropna()
    visited = []
    for i in range(0, len(race_violations['driver_race'])):
        if race_violations['driver_race'].iloc[i] not in visited:
            visited.append(race_violations['driver_race'].iloc[i])

    # get all the speeding data into a single dataframe
    speeding = race_violations.groupby('violation').get_group('Speeding')

    arr = [0] * len(visited)
    for i in range(0, len(speeding["driver_race"])):
        check = speeding['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Speeding, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()

    MV = race_violations.groupby('violation').get_group('Moving violation')

    arr = [0] * len(visited)
    for i in range(0, len(MV["driver_race"])):
        check = MV['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Moving Violation, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()

    equipment = race_violations.groupby('violation').get_group('Equipment')

    arr = [0] * len(visited)
    for i in range(0, len(equipment["driver_race"])):
        check = equipment['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Equipment, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()

    RP = race_violations.groupby('violation').get_group('Registration/plates')

    arr = [0] * len(visited)
    for i in range(0, len(RP["driver_race"])):
        check = RP['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Registration/plates, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()

    SB = race_violations.groupby('violation').get_group('Seat belt')

    arr = [0] * len(visited)
    for i in range(0, len(SB["driver_race"])):
        check = SB['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Seat Belts, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()

    other = race_violations.groupby('violation').get_group('Other')

    arr = [0] * len(visited)
    for i in range(0, len(other["driver_race"])):
        check = other['driver_race'].iloc[i]
        for j in range(0, len(visited)):
            if check == visited[j]:
                arr[j] += 1

    plt.title('Violation: Other, By Ethnicity')
    plt.pie(arr, labels=visited)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("police.csv")

    # data_info(df)
    data_manipulation(df)

    # ---- show the number of races by number of violations ---- #
    # ---------------------------------------------------------- #

    print("this is the number of different races in 'driver_race' column")
    print(df['driver_race'].value_counts())

    print("this is the number of different violations in 'violations' column")
    print(df['violation'].value_counts())

    violations_per_race()

    # ---- show the number of violations by races ---- #
    # ------------------------------------------------ #
    violations_and_race()

    # ---- show the age distribution ---- #
    # ----------------------------------- #

    # ---- stop duration by race ---- #
    # ------------------------------- #

    # ---- time frequency of stops per race. ---- #
    # ------------------------------------------- #

    '''print(df.columns)
    print(df["driver_gender"].nunique())
    print(df['driver_gender'].isnull().sum())
    print(df["driver_age"].nunique())
    print(df["driver_race"].nunique())
    print(df["violation"].nunique())
    print(df["is_arrested"].nunique())'''

    # print(df.isnull().sum())
    # df.drop(['county_name'], axis=1, inplace=True)
    # df = df.drop(columns= df.columns[2])
    # print(df.columns)

    # print(df.head())
    # df = df.drop(labels= [0], axis=0)
    # print(df.head())

    '''print(df.isnull().sum())
    df = df.dropna(subset=['stop_outcome'])
    print(df.isnull().sum())'''

    # print(df.describe())

    # print(df['search_type'].value_counts())

    # print(df['driver_gender'].isnull().sum())

    # print(df.sample(7))
    # print(df.nsmallest(6,"driver_age"))
    # print(df.nlargest(6,"driver_age"))

    # print(df.groupby('driver_race').agg(np.mean))

    # print(df.groupby('driver_age').get_group("Asian"))

    # print(df.loc[:5,['driver_age', 'violation']])
    # print(df.loc[(df.driver_age<16) & (df.violation=='Speeding')])

    # print(df.iloc[:8,:5])
    # print(df[['stop_date', 'driver_age', 'driver_race', 'violation', 'stop_outcome']].sort_values(by='driver_age'))

    # print(df.query('45< driver_age < 50').head())
    # print(df.query('driver_age == 17 and driver_gender == "F"'))

    # print(df.set_index('stop_date').head(4))

    '''newdf = pd.get_dummies(df.driver_race, prefix = "driver_race_").iloc[:,0:]
    print(newdf)
    newdf2 = pd.get_dummies(df.driver_race, prefix="driver_race_")
    print(newdf2)

    numerical_df = df.select_dtypes(include=[np.numerical])
    categorical_df = df.select_dtypes(include='object')

    comb_df = pd.concat([numerical_df, categorical_df], axis=1)'''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

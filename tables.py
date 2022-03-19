import pandas as p
import numpy as np
import features as f
import psycopg2 as db
import random




# builds the table's that are queried in actual table form
# cursor: the cursor to execute and fetch queries
# tables:  list of table ids
def build_table(cursor, tables):
    df_list = []
    series_list = []
    temp_list = []
    for table in tables:
        cell = sql_commands(cursor, table,
                            'SELECT tableid, tokenized, colid, rowid FROM main_tokenized WHERE tableid=%s')
        # cell is a list of tuples representing the table
        col = 0
        for tuple in cell:
            if col == tuple[2]:
                temp_list.append(tuple[1])
            else:
                series_list.append(temp_list)
                temp_list = [tuple[1]]  # to avoid losing entries
                col += 1
        series_list.append(temp_list)  # to avoid losing the last column
        df_list.append((table, p.DataFrame(series_list).T))
        series_list = []
        temp_list = []  # to avoid bloating the table

    return df_list


#  saves a frame to designated destination
#  path: the path to save to
#  frame: the Dataframe representing the Data that needs to be saved
def write_table(path, frame, index=False):
    try:
        if path.lower().endswith(('.xlsx', '.ods')):
            with p.ExcelWriter(path) as writer:
                frame.to_excel(writer, index=index)
        elif path.lower().endswith('.csv'):
            frame.to_csv(path, index=index)
        else:
            print('Filetype is not supported! Please choose between .xlsx, .ods and .csv')
            return True  # to check whether the saving progress failed
        return False  # to check whether the saving progress succeeded
    except (FileNotFoundError, ValueError):
        print('Directory not found! Try again')
        return True


# checks the tables iteratively whether they are applicable or not
# df_list: list of the tables in Dataframes
# threshold: the threshold of accepted None values
def check_tables(cursor, df_list, threshold, name=None, index=False):
    none_counter = 0
    new_df_list = []
    for table in df_list:
        # check whether there are known headers to begin with
        headers = sql_commands(cursor, table[0],
                               'SELECT tableid, colid, header, header_tokenized FROM columns_tokenized WHERE tableid=%s')
        if len(headers) > 0:
            #insert_header_row(table[1], headers)
            for row in table[1].itertuples(name=name, index=index):
                for entry in row:
                    if entry in ('', ' ', 'blank', 'null', 'Null', 'NULL', 'None', None):
                        none_counter += 1

            ratio = none_counter / table[1].size
            if not (ratio >= threshold or len(table[1]) < 4 or len(
                    table[1].columns) < 4):
                new_df_list.append(table)
            none_counter = 0
    return new_df_list


# creates a list of tableid's. the tables are not checked yet
# start: lower bounder (inclusive)
# end: upper bounder (inclusive)
# n: number of tableid's to be generated
# path: if tableid's are saved somewhere as simple file, this is the path to it
def tableid_gen(start, end, n, path):
    lines = []
    if path:  # if we don't save the tableid's somewhere keep path to None
        try:
            with open(path, 'r') as file:
                lines = [int(i) for i in file.readlines()]
        except FileNotFoundError:
            print('file has not been found')
            lines = []

    while len(lines) < n:
        rng = random.randint(start, end)
        while lines.count(rng) >= 1:
            rng = random.randint(start, end)
        lines.append(rng)

    return lines


# saves all the tables for training in xlsx files
def create_train_tables(cursor, threshold, start=1, end=41200, n=50, path='train/tableids.txt'):
    tables = tableid_gen(start, end, n, path)
    df_list = build_table(cursor, tables)  # tables might not yet be applicable
    df_list = check_tables(cursor, df_list, threshold)  # list of applicable tables

    for table in df_list:
        print(table[1].to_markdown())
        print()

    if path:
        try:
            with open(path, 'w') as file:
                for id in df_list:
                    file.write(str(id[0]) + '\n')
        except FileNotFoundError:
            print('hat der das wohl doch nicht geschrieben')

    actual_df_list = []  # df_list returns a tuple of table id and table but at this point forward
    # only tables are needed

    for tuple in df_list:
        actual_df_list.append(tuple[1])
    return actual_df_list


#  reads in tabular data of various type
#  path: the path to read from
#  return: the Dataframe representing the read data
def read_table(path):
    frame = None  # to prevent UnboundLocalError
    try:
        if path.lower().endswith(('.xlsx', 'ods')):
            frame = p.read_excel(path)
        elif path.lower().endswith('.csv'):
            frame = p.read_csv(path)
        # else:
        #     frame = p.read_html(path)
        return frame
    except (FileNotFoundError, IsADirectoryError):
        print('file does not exist')
        return None


#  iterate through DF of train/test data to build corresponding featurevectors (np.arrays)
#  stores them in a list
#  data: the data from the queries as a list
#  return: a list of np.arrays representing the featurevectors
def iterate_data(data, name=None, index=False, header=False):
    vector_list = list()
    for frame in data:
        if header:
            vector = np.empty(len(f.build_feat_dict()))
            columns = extract_header_values(frame.columns)
            vector_list.append(f.calc_features(columns, vector))
        for row in frame.itertuples(name=name, index=index):
            vector = np.empty(len(f.build_feat_dict()))
            vector_list.append(f.calc_features(row, vector))
    return vector_list


# swaps the desired rows in given frame. swaps inplace
# frame: the given frame
# row: the index of new
# index: the other index
def swap_rows(frame, row, index, header):
    if header:
        if row != -1:  # if row is -1 then the header is already correctly placed
            new = frame.iloc[row].copy()
            header = frame.columns.copy()
            frame.iloc[row] = header
            frame.columns = new
            frame.reset_index(drop=True)
    else:
        new = frame.iloc[row].copy()
        old = frame.iloc[index].copy()
        frame.iloc[row] = old
        frame.iloc[index] = new


# it uses the vectors' classification to order the rows new. Swaps inplace but also returns
# pred: the predicted array
# frame: the frame that got classified
def newly_ordered_frame(pred, frame):
    indices = np.where(pred == 0)  # returns tuple of arrays
    old = -1
    header = True
    for i in indices[0]:
        swap_rows(frame, i - 1, old, header)
        old += 1
        header = False
    return frame


# iterate through the DF to rebuild the vectors
# stores them in a list
# frame: the Dataframe representing the featurevectors
# return: list of np.arrays representing the featurevectors
def iterate_feature_vectors(frame, name=None, index=False):
    vectors = list()
    for row in frame.itertuples(name=name, index=index):
        vectors.append(np.asarray(row))
    return vectors


# iterate through the DF to rebuild the ground_truth
# stores them in a list
# frame: the Dataframe representing the featurevectors
# return: list of np.arrays representing the featurevectors
def iterate_ground_truth(frame, name=None, index=False):
    vectors = list()
    for row in frame.itertuples(name=name, index=index):
        vectors.append(row[0])
    return vectors


# inserts the remaining header rows in the dataframe's last row if it is not present already
# headers is a list of tuple: (id, colid, header, header_token)
def insert_header_row(df, headers):
    n = m = len(df)
    i = 0
    header = [value[3] for value in headers]
    while i < n:
        if df.iloc[i].tolist() != header:
            m -= 1
        i += 1
    if m == 0:
        df.loc[-1] = header  # adding a row
        df.index = df.index + 1  # shifting index


# returns True if there is a valid DataFrame
def empty(frame):
    if isinstance(frame, p.DataFrame):
        return True

    return False


# converts all values of the frame into a String
def convert_values_to_string(frame):
    i = 0
    n = len(frame.columns)
    while i < n:
        frame[frame.columns[i]] = frame[frame.columns[i]].astype(str)
        i += 1


# takes the values of the header column in the DataFrame and stores them in a tuple
def extract_header_values(frame_header):
    header_row = []
    for i in frame_header:
        i = str(i)
        header_row.append(i)
    return tuple(header_row)


# helper to build the ground-truth
def build_ground_truth(path, path2):
    lines = []
    series = []
    try:
        with open('train/kopfzeilen.txt', 'r') as file:
            lines = [int(i) for i in file.readlines()]
    except FileNotFoundError:
        print('file has not been found')
        return None

    vectors = read_table(path)
    n = len(vectors)
    i = 0
    while i < n:
        value = int(vectors.iloc[i][0])
        if value in lines and value != 119:
            series.append(0)
        elif value in lines:
            series.append(2)
        else:
            series.append(1)
        i += 1

    print(series)
    write_table(path2, p.DataFrame(series))


import pandas as p
import tables as t
import numpy as np
import re


# helper to determine whether value is a numeric or not
def is_num(value):
    try:
        num = float(value)
        return True, str(num)
    except:
        return False, None


# helper to determine whether numbers in a string are separated with whitespace
def split_string(value):
    to_return = value.replace(' ', '')
    if to_return.isdigit():
        return True, to_return
    else:
        worked, number = is_num(to_return)
    return worked, number


# helper to remove all empty strings from list
def remove_n_times(lst):
    n = lst.count('')
    i = 0
    while i < n:
        lst.remove('')
        i += 1


# blanks ration- laut Mitlöhner sind alle Spalten entweder nicht oder so ziemilch vollständig besetzt
# einige meta und footter Daten besetzten aber nur wenige
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def blank_ratio(row, vector, index):
    blanks = 0
    n = len(row)
    for cell in row:
        if cell in ('None', '', ' ', None, 'NaN', 'NAN'):  # WIE NaN KLÄREN
            blanks += 1
    ratio = blanks / n
    vector[index] = ratio


# string_num_ratio: similar used in Jiang et al.
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def string_num_ratio(row, vector, index):
    String = 0
    numeric = 0
    for cell in row:
        if isinstance(cell, str):
            cell_ = cell.replace(',', '')
            condition, number = is_num(cell_)
            if condition:
                numeric += 1
            elif (re.search("\d{1,2}[. ]\d{1,2}[. ]\d{2,4}", cell)
                  or re.search("\d{1,2}[. ]\w{3,9}[. ]\d{2,4}", cell)) and len(cell) <= 17:
                numeric += 1
            elif cell not in ('None', ' ', '', 'NaN', 'inf', '-inf', None):
                String += 1

    if String != 0 or numeric != 0:
        ratio = String / (String + numeric)
    else:
        ratio = 0
    vector[index] = ratio


# measure the similarity between the adjacent rows. Gedanke hier: header Zeilen sind in der Regel nur einzeln
# und wenn hiearcisch dann aber nicht alle Spalten umfassend
# VIELLEICHT SOWAS WIE WIE NAH SIND DIE WERTE AN EINANDER QUASI WIE NAH DIE VEKTOREN IM KOORDINATEN SYSTEM SIND
# KÖNNTE GEWICHTUINGEN MITEINBEZIEHEN
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def similarity_score(row, vector, index):
    return 0


# used in koci et al and other papers
# counts all words and calcs the mean over length of row
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def word_count(row, vector, index):
    cells = []
    for cell in row:
        if isinstance(cell, str):
            buf = re.split(r"[\W_]+", cell)
            remove_n_times(buf)  # remove empty Strings since they don't count
            cells.append(buf)  # list of words in the cell

    n_cells = len(row)
    count = 0
    for words in cells:  # words is a list of str
        count += len(words)

    norm_count = count / n_cells  # average words per cell in a row
    vector[index] = norm_count


# sets a flag if the string has too many words to be considered a header
def set_flag_when_too_many(row, vector, index):
    flag = 0
    for cell in row:
        if isinstance(cell, str):
            if len(cell.split()) > 3:
                flag = 1
    vector[index] = flag


# checks whether the cells have multiple datatypes within
def mixed_data_types_within_cells(row, vector, index):
    types = {'integer': 0, 'float': 0, 'string': 0, 'dates': 0}
    celltypes = []
    for cell in row:
        if isinstance(cell, str):
            cell_ = cell.replace(',', '')
            words = cell_.split()
            for word in words:
                condition, _ = is_num(word)
                if word.isdigit():
                    types['integer'] = 1
                elif condition:
                    types['float'] = 1

                elif (re.search("\d{1,2}[. ]\d{1,2}[. ]\d{2,4}", cell)
                      or re.search("\d{1,2}[. ]\w{3,9}[. ]\d{2,4}", cell)) \
                        and len(cell) <= 17:  # has to be cell to include dates like: '25 sep 2021'
                    types['dates'] = 1
                else:
                    types['string'] = 1
        celltypes.append(types['integer'] + types['float'] + types['string'] + types['dates'])
        types['integer'] = 0
        types['float'] = 0
        types['string'] = 0
        types['dates'] = 0
    # max per cell will always be 4 normalize by length will give certain insight of type distribution
    vector[index] = sum(celltypes) / len(row)


# checks whether the cells in the row have multiple datatypes within
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def mixed_data_types(row, vector, index):
    types = {'integer': 0, 'float': 0, 'string': 0, 'dates': 0}
    for cell in row:
        if isinstance(cell, str):
            cell_ = cell.replace(',', '')
            condition, num = split_string(cell_)  # fuse numbers that are separated by whitespaces
            if condition:
                if num.isdigit():
                    types['integer'] = 1
                else:
                    types['float'] = 1
            elif (re.search("\d{1,2}[. ]\d{1,2}[. ]\d{2,4}", cell)
                  or re.search("\d{1,2}[. ]\w{3,9}[. ]\d{2,4}", cell)) \
                    and len(cell) <= 17:
                types['dates'] = 1
            else:
                types['string'] = 1
    vector[index] = types['integer'] + types['float'] + types['string'] + types['dates']


# checks whether the cell is a date value
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# index: the index of the position in the featurevector in that the feature will be stored
def is_date(row, vector, index):
    dates = 0
    for cell in row:
        if isinstance(cell, str) and (re.search("\d{1,2}[. ]\d{1,2}[. ]\d{2,4}", cell)
                                      or re.search("\d{1,2}[. ]\w{3,9}[. ]\d{2,4}", cell)):
            dates = 1

    vector[index] = dates


# calls the feature methods above using a dictionary
# row: a tuple representing the row of tabular data
# vector: a np.array that represents the featurevector
# return: a featurevector as a np.array
def calc_features(row, vector):
    i = 0
    dict = build_feat_dict()
    for feature in dict:
        dict[feature](row, vector, i)
        i += 1
    return vector


# build selected feature dicts to be used to classify with
# return: a dictionary of features
def build_feat_dict():
    dict = {
        'word_count': word_count,
        #'set_flag_when_too_many': set_flag_when_too_many,
        'string_num_ratio': string_num_ratio,
        'blank_ratio': blank_ratio,
        'is_date': is_date,
        'mixed_data_types_within_cells': mixed_data_types_within_cells,
        #'mixed_data_types': mixed_data_types,

    }
    return dict

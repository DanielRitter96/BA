import numpy as np
import time
import scikit
import tables as t


# parses the input file
# returns: the tabular data in a list with one single Dataframe representing the data
def parse_input_file():
    frame = None  # None to prevent BoundError if no input provided
    file = input('Please enter the path of the file that you want to be classified: ')
    if file != '':
        frame = t.read_table(file)
        while not t.empty(frame) and file != '':
            print('File not found! try again.')
            file = input('Please enter the path of the file that you want to be classified: ')
            frame = t.read_table(file)

    return [frame]  # the function that calcs the feature vectors needs a list


# keeps the application running
def keep_alive(clf, eval=False, y_test=None):
    frame_list = parse_input_file()
    while t.empty(frame_list[0]):
        start = time.process_time_ns()
        t.convert_values_to_string(frame_list[0])
        vectors = t.iterate_data(frame_list, header=True)
        print(vectors)
        prediction = clf.predict(np.asarray(vectors))
        print(prediction)
        if eval:
            scikit.calc_scores(clf, y_test=y_test, y_pred=prediction, X_test=vectors)
        t.newly_ordered_frame(prediction, frame_list[0])
        path = input('please enter the path in which you would like to save the file at: ')
        saved = t.write_table(path, frame_list[0])
        while saved:  # keep asking for a correct link
            path = input('it looks like it didn\'t work as planned!'
                         '\nplease enter the path in which you would like to save the file at: ')
            saved = t.write_table(path, frame_list[0])
        print('The process finished, if you would like to continue please enter a new file'
              ' \nif you would like to stop the application please just press enter'
              ' if you don\'t want to do anything please just do nothing!')

        frame_list = parse_input_file()
        end = time.process_time_ns()
        print('time: %s' % (end - start))

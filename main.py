import tables as t
import numpy as np
import scikit as s
import input_output as io
import features as f
import time



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ids = t.tableid_gen(30001, 40000, 250, None)
    # print(ids)
    print('connecting to database')
    cursor, connection = t.connect_postgres_database()


    print('create tables to train the model with')
    df_list = t.create_train_tables(cursor, .5, start=30001, end=40000, n=250, path=None)


    print('disconnect from database')
    t.disconnect_database(connection)

    print(len(df_list))
    #for table in df_list:
        # check whether there are known headers to begin with

    # print('create actual vectors to train the model')
    # vectors = t.iterate_data(df_list)
    #
    # print('save the featurevectors to train the Model with the SAME data ')
    # vector_frame = f.p.DataFrame(vectors, columns=f.build_feat_dict())
    # t.write_table('train/11/vectors.xlsx', vector_frame, index=False)
    # vector_frame_without = vector_frame.drop_duplicates()
    # t.write_table('train/11/vectors_without.xlsx', vector_frame_without, index=False)

    # t.build_ground_truth('train/11/vectors_without.xlsx', 'train/11/ground-truth2.xlsx')

    # print('reading the feature vectors data')
    # vector_frame = t.read_table('train/3/vectors_without.xlsx')
    #
    # vectors = t.iterate_feature_vectors(vector_frame)
    #
    # print('reading the ground truth data')
    # y_frame = t.read_table('./train/3/ground-truth2.xlsx')
    # y = t.iterate_ground_truth(y_frame)
    #
    # print('create the classifier using auto-SKlearn')
    # print('train the model')
    #
    # try:
    #     clf = s.load_model('train/3/model2.joblib')
    #     fit = False
    # except FileNotFoundError:
    #     clf = s.create_classifier(total_time=3600, resample='cv', resample_args={'folds': 10}, include={'classifier': ['k_nearest_neighbors']})
    #     fit = True
    #
    # clf, X_test, y_test = s.train_model(clf, np.asarray(vectors), np.asarray(y), 'train/3/model2.joblib', fit=fit, save=False)
    #
    # print('leaderboard:')
    # print(clf.leaderboard(top_k=1))
    # print('stats')
    # print(clf.sprint_statistics())
    # # print('models')
    # # print(clf.show_models())
    #
    #
    # y_pred = clf.predict(X_test)
    #
    #
    # print(y_test)
    # print(y_pred)
    #
    # acc, pred, rec, f1 = s.calc_score(y_test, y_pred)
    # print('printing the scores of the train process')
    # print('Accuracy = %s, prediction = %s, recall = %s, f1 = %s' % (acc, pred, rec, f1))
    #s.performance_over_time(clf)
    #s.print_confusion_matrix(clf, X_test, y_test)
    # # s.print_confusion_matrix(clf, X_test, y_pred)
    #io.keep_alive(clf, eval=False, y_test=np.asarray([1,1,0,0,1,1,1,0,1]))





# See PyCharm help at https://www.jetbrains.com/help/pycharm/

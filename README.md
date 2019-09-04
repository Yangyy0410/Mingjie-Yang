1. Different kinds of models are stored in different directories which have the same name with corresponding models.
2. For example, GRU1L64 stands for bidirectional GRU with one layer and 64-dimensional hidden states.
3. There is no complex procedure, just use the command line: python3 filename.py.
4. Run EVA-* will directly evaluate the model on the training set using trained.
5. Run the files named after model will start to train the model from the very beginning.
6. "shuffle_list.npy" stored the information regarding how the dataset was split into training set, validation set and test set, so please don't delete them.
7. "Data" stores the data file.
8. "DeepLearning_data.h5" and "Traditional_data.h5" are processed data.

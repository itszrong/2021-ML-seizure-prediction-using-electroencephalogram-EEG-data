import scipy
from scipy.fft import fft, ifft, fftfreq
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import figure
import os

#Specify information about files
print('0: Specify information about files')
path_data = ("CHB-MIT")
ids_data = np.array(os.listdir(path_data))  # list of images filenames to be used

patients = np.arange(1, 47)
# missing = [11, 12, 13, 14, 17, 19, 20, 38, 35, 44, 45]
missing = [11, 12, 13, 14, 17, 19, 20] + list(np.arange(22,47))

# missing = [38, 35, 44, 45]
N = np.where(np.isin(patients, missing))
remaining_patients = np.delete(patients, N)

#Above if downloaded full data set, use below with the data in CHB-MIT file
remaining_patients = [1, 3, 4, 10, 15]

#Read data from CHB-MIT
print('1: Read data from CHB-MIT')
bool = 0
print(len(ids_data))
for i in range(len(ids_data)):
    path_EEG = os.path.join(path_data, ids_data[i])
    data = mne.io.read_raw_edf(path_EEG)
    if bool == 0:
        size = np.asarray(data.get_data())
        print(size.shape[0], size.shape[1])
        raw_data_array = np.zeros((46, size.shape[0], size.shape[1]))
        print(raw_data_array.size)
        bool = 1
    raw_data_array[remaining_patients[i]] = np.asarray(data.get_data())

print('Raw data array shape', raw_data_array.shape) #46, 23, 926100

#Create labels, create indices to insert 1s and 2s
print('2: Create labels, create indices to insert 1s and 2s')
labels = np.zeros((46, raw_data_array.shape[2]))
labels_one = np.array([[3, 2996, 3036], [4, 1467, 1494], [15, 1732, 1772], [16, 1015, 1066], [18, 1720, 1810], [21, 327, 420],[26, 1862, 1963]])
labels_twos = labels_one
preictal_period_length = 5
print(labels_one.shape)
for i in range(labels_one.shape[0]):
    labels_twos[i][1] = labels_one[i][1] - preictal_period_length
    labels_twos[i][2] = labels_one[i][1] - 1

#Insert 1s and 2s
print('3: Insert 1s and 2s')
for i in range(labels_one.shape[0]):
    #insert 1s
    labels[labels_one[i][0]][(labels_one[i][1] - 1) * 256:(labels_one[i][2] * 256)] = 1
    labels[labels_one[i][0]][(labels_one[i][1] - 1) * 256:(labels_one[i][2] * 256)] = 1

    #insert 2s
    # labels[labels_one[i][0]][(labels_twos[i][1] - 1) * 256:(labels_twos[i][2] * 256)] = 2
    # labels[labels_one[i][0]][(labels_twos[i][1] - 1) * 256:(labels_twos[i][2] * 256)] = 2
print('Labels shape is', labels.shape) #46, 926100

#function to make windows
def window(x, initial, final):
    window_data = x[:,initial:final]
    return window_data

#Specify information about windows
print('4: Specify information about windows')
initial = 0
total_data_length = 926100
seconds = 5
interval_length = seconds*256
no_windows = 10*total_data_length//seconds//256
shift = int((total_data_length-2*seconds*256)//(1.1*no_windows))
print('Shift is', shift)

#Create windows for data
print('5: Create windows for data')
window_total = np.zeros((no_windows, raw_data_array.shape[1], interval_length))
window_total_fft = np.zeros((no_windows, raw_data_array.shape[1], interval_length))
for j in remaining_patients:
    print(j)
    for i in range(no_windows):
        # print(i)
        window_data_provided = window(raw_data_array[j],i*shift,(i)*shift+interval_length)
        window_total[i] = window_data_provided
        # window_total[i] = window(raw_data1[j], i * shift, (i) * shift + interval_length)
        # window_total_fft[i] = np.fft.fft(window_data_provided)

        # print('window data provided shape', window_data_provided.shape)

    np.save('WINDOW DATA\window_data' + str(j), window_total)

#Create windows for labels
print('6: Create windows for labels')
labels_total = np.zeros((no_windows, 46, interval_length))
print('Labels shape is ', labels.shape)
for i in range(no_windows):
    # print(i)
    label_window_data = window(labels,i*shift,(i)*shift+interval_length)
    # print('label window data shape', label_window_data.shape)
    labels_total[i] = label_window_data
print(labels_total.shape)
print('Labels window size', labels_total.shape)
np.save('WINDOW LABELS\Label data', labels_total)

print('7: All data in files')


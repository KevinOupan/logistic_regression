from numpy import *
import csv
import pandas as pd


def train_data(fn):
    data = []
    label = []
    fm = open('./' + fn)
    fd = csv.reader(fm)
    for line in fd:
        data.append([1, float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
                     float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]),
                     float(line[12]), float(line[13]), float(line[14]), float(line[15]), float(line[16]), float(line[17]),
                     float(line[18]), float(line[19]), float(line[20]), float(line[21]),  float(line[22]), float(line[23]),
                     float(line[24]), float(line[25]), float(line[26]), float(line[27]), float(line[28])])
        label.append(int(line[29]))
    return data, label


def test_data(fn):
    data = []
    fm = open('./' + fn)
    fd = csv.reader(fm)
    for line in fd:
        data.append([1, float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
                     float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]),
                     float(line[12]), float(line[13]), float(line[14]), float(line[15]), float(line[16]), float(line[17]),
                     float(line[18]), float(line[19]), float(line[20]), float(line[21]),  float(line[22]), float(line[23]),
                     float(line[24]), float(line[25]), float(line[26]), float(line[27]), float(line[28])])
    return data


def sigmoid(h):
    return 1/(1+exp(-h))


filename_1 = 'data_train.csv'
filename_2 = 'data_test.csv'

x, y = train_data(filename_1)
z = test_data(filename_2)


data_matrix = mat(x)
label_matrix = mat(y).transpose()
p, q = shape(data_matrix)
# weights = mat([17452.84499884, -1198078.40240807, -6372.55500097, 166301.74512946, 1563200.04067343, -1958.06889557,
#                327696.99515424, -29334.75094218, -248795.10973833, 9803.34036749, -173889.10510739, -33143.00486912,
#                104306.59012718, -326005.27040052, 30501.49462872, -14561.65656663, 410831.73187279, 334255.34023624,
#                381271.53511174, 87518.83499631, -102877.6655051, 1396148.59990669, -279755.95449989, -1008752.59879625,
#                -125651.85547566, -28004.66024027, -677665.68479238, -659196.93418658, -10449.00513083,
#                -679535.95047195]).transpose()
weights = mat([9.05320362e+03, -1.39568990e+06, -1.89435397e+04, 1.57802866e+05, 1.43076331e+06, -2.13352408e+04,
               3.13289179e+05, -1.61328643e+04, -2.75073294e+05, -1.99539746e+03, -1.99107643e+05, -3.73537261e+04,
               7.73845560e+04, -3.53572738e+05, 1.18391693e+04, -1.31990279e+03, 4.02923755e+05, 3.27683939e+05,
               2.44681006e+05, 6.04767055e+04, -1.29278790e+05, 7.24177290e+05, -2.89364913e+05, -1.01207924e+06,
               -1.52980148e+05, -5.19802052e+04, -6.83630525e+05, -6.66056360e+05, -6.70307254e+03,
               -6.86986997e+05]).transpose()
alpha = 0.00001
max_loop = 120000
for i in range(max_loop):
    y_pre = sigmoid(data_matrix * weights)
    error = label_matrix - y_pre
    grad = data_matrix.transpose() * error
    weights = weights + alpha * grad

w = weights


def  prediction(v, ww):
    pre = []
    for line in v:
        if line * ww > 0:
            pre.append(1)
        else:
            pre.append(0)
    return pre


outcome = prediction(z, w)
print(outcome)
test = pd.DataFrame(data=outcome)
test.to_csv('test3.csv')

# print(w)



# [[ 8.44630561e+03]
#  [-1.40644521e+06]
#  [-1.98217830e+04]
#  [ 1.57207909e+05]
#  [ 1.41651193e+06]
#  [-2.26016876e+04]
#  [ 3.12255103e+05]
#  [-1.40288769e+04]
#  [-2.76997487e+05]
#  [-2.86865272e+03]
#  [-2.00995418e+05]
#  [-3.76724449e+04]
#  [ 7.54326898e+04]
#  [-3.55594315e+05]
#  [ 1.05493433e+04]
#  [-7.77714162e+02]
#  [ 4.02587519e+05]
#  [ 3.27357418e+05]
#  [ 2.34551001e+05]
#  [ 5.85070064e+04]
#  [-1.31203186e+05]
#  [ 6.75625449e+05]
#  [-2.90157592e+05]
#  [-1.01097820e+06]
#  [-1.55016101e+05]
#  [-5.36877602e+04]
#  [-6.83181609e+05]
#  [-6.66078040e+05]
#  [-6.20175575e+03]
#  [-6.87056938e+05]]
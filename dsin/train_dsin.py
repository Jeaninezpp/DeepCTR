import sys
sys.path.append("..")
import argparse
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DSIN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
home = os.environ['HOME']


def auroc(y_true,y_pred):
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)


if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    # arguments
    parser = argparse.ArgumentParser(description='Train dsin model')
    parser.add_argument('--frac', type=float, default=0.001, help='Fraction')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch')
    parser.add_argument('--dataset', type=str, default= home+'/datasets/dsin2/', help='Dataset path.')
    args = parser.parse_args()
    print(args)

    FRAC = args.frac
    EPOCH = args.epoch
    DATASET = args.dataset
    SESS_COUNT = 5
    SESS_MAX_LEN = 10
    ModelPath = DATASET + '/model_input/dsin_input/'

    fd          = pd.read_pickle(ModelPath + '/dsin_fd_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    model_input = pd.read_pickle(ModelPath + '/dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    label       = pd.read_pickle(ModelPath + '/dsin_label_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    sample_sub  = pd.read_pickle(DATASET + '/sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    sample_sub['idx'] = list(range(sample_sub.shape[0])) # add index
    train_idx = sample_sub.loc[sample_sub.time_stamp < 1494633600, 'idx'].values
    test_idx  = sample_sub.loc[sample_sub.time_stamp >= 1494633600, 'idx'].values

    train_input = [i[train_idx] for i in model_input]
    test_input  = [i[test_idx] for i in model_input]

    train_label = label[train_idx]
    test_label  = label[test_idx]

    sess_feature = ['cate_id', 'brand']
    BATCH_SIZE = 4096
    TEST_BATCH_SIZE = 2 ** 14

    model = DSIN(fd, sess_feature, sess_max_count=SESS_COUNT,
                 bias_encoding=False, att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200,80))
    model.compile('adagrad', 'binary_crossentropy', metrics=['binary_crossentropy', auroc])
    hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=EPOCH,
                      initial_epoch=0, verbose=1, validation_data=(test_input, test_label))
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)
    print("")
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
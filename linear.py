import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from generate import GENDER_TO_STR_DICT, create_model, set_shape

from load import MeshMeasurements, load, load_star
from metrics import evaluate
from logs import log
from visualize import visualize


MODEL_PATH_TEMPLATE = './models/{}_{}.sav'


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, 
        help='root data folder'
    )
    parser.add_argument(
        '--dataset_name', type=str, 
        help='dataset name'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'],
        help='train or test'
    )
    parser.add_argument(
        '--gender', type=str, choices=['male', 'female', 'neutral'],
        help='train or test'
    )
    parser.add_argument(
        '--noise', type=float, default=0.,
        help='standard deviation of the Gaussian noise added to weight'
    )

    return parser


def train(args):
    print(f'Preparing {args.dataset_name} dataset...')
    X, y, genders = load(args)
    print('Train/test splitting...')
    X_train, X_test, y_train, y_test, _, gender_test = train_test_split(
        X, y, genders, test_size=0.33, random_state=2021)

    model = LinearRegression()
    reg = model.fit(X_train, y_train)
    pickle.dump(model, open(MODEL_PATH_TEMPLATE.format(args.dataset_name, args.gender), 'wb'))
    print('Predicting...')
    y_predict = reg.predict(X_test)

    smpl_model = create_model(GENDER_TO_STR_DICT[0])
    smpl_output = set_shape(smpl_model, np.zeros(10))
    verts = smpl_output.vertices.detach().cpu().numpy().squeeze()
    faces = smpl_model.faces.squeeze()
    zero_measurements = MeshMeasurements(0, verts, faces, noise_std=args.noise).apmeasurements
    zero_measurements = np.tile(zero_measurements, (y_predict.shape[0], 1))

    print('Evaluating...')
    #params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(y_predict, y_test, gender_test)
    params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(zero_measurements, y_test, gender_test)

    print('Logging to stdout...')
    log(params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    print('Visualizing...')
    visualize(model, args, params_errors, measurement_errors, s2s_dists)


def test(args):
    print(f'Preparing {args.dataset_name} dataset...')
    X_test, y_test, genders = load(args)
    print('Loading model...')
    model = pickle.load(open(MODEL_PATH_TEMPLATE.format('caesar', args.gender), 'rb'))
    print('Predicting...')
    y_predict = model.predict(X_test)

    print('Evaluating...')

    smpl_model = create_model(GENDER_TO_STR_DICT[0])
    smpl_output = set_shape(smpl_model, np.zeros(10))
    verts = smpl_output.vertices.detach().cpu().numpy().squeeze()
    faces = smpl_model.faces.squeeze()
    zero_measurements = MeshMeasurements(0, verts, faces, noise_std=args.noise).apmeasurements
    zero_measurements = np.tile(zero_measurements, (y_predict.shape[0], 1))

    params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(y_predict, y_test, genders)
    #params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(zero_measurements, y_test, genders)

    print('Logging to stdout...')
    log(params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    print('Visualizing...')
    visualize(model, args, params_errors, measurement_errors, s2s_dists)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    locals()[args.mode](args)

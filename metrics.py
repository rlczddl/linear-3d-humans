import numpy as np

from generate import create_model, set_shape, GENDER_TO_STR_DICT
from load import MeshMeasurements


def params_error(pred_params, gt_params):
    return np.abs(pred_params - gt_params)


def measurement_error(model_faces, pred_verts, gt_verts):
    pred_meas = MeshMeasurements(pred_verts, model_faces).apmeasurements
    gt_meas = MeshMeasurements(gt_verts, model_faces).apmeasurements
    return np.abs(pred_meas - gt_meas)


def surface2surface_dist(pred_verts, gt_verts):
    return np.linalg.norm(pred_verts - gt_verts, axis=1)


def evaluate(y_predict, y_target, genders):
    return np.zeros(10), np.zeros(10), np.mean(np.abs(y_predict - y_target), axis=0), \
        np.std(np.abs(y_predict - y_target), axis=0), np.empty(0), np.empty(0)
    '''
    else:
        params_errors = []
        measurement_errors = []
        surface2surface_dists = []

        for sample_idx, pred_params in enumerate(y_predict):
            gender = GENDER_TO_STR_DICT[genders[sample_idx]]
            gt_params = y_target[sample_idx]
            
            model = create_model(gender)
            pred_output = set_shape(model, pred_params)
            pred_verts = pred_output.vertices.detach().cpu().numpy().squeeze()

            gt_output = set_shape(model, gt_params)
            gt_verts = gt_output.vertices.detach().cpu().numpy().squeeze()

            model_faces = model.faces.squeeze()

            params_errors.append(params_error(pred_params, gt_params))
            measurement_errors.append(measurement_error(model_faces, pred_verts, gt_verts))
            surface2surface_dists.append(surface2surface_dist(pred_verts, gt_verts))

        params_errors = np.array(params_errors)
        measurement_errors = np.array(measurement_errors)
        surface2surface_dists = np.array(surface2surface_dists)

        params_means = np.mean(params_errors, axis=0)
        params_stds = np.std(params_errors, axis=0)

        measurement_means = np.mean(measurement_errors, axis=0)
        measurement_stds = np.std(measurement_errors, axis=0)

        surface2surface_means = np.mean(surface2surface_dists, axis=0)
        surface2surface_stds = np.std(surface2surface_dists, axis=0)

        return params_means, params_stds, measurement_means, measurement_stds, surface2surface_means, surface2surface_stds
    '''

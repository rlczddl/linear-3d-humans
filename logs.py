from load import MeshMeasurements
from models import Models


def log_errors(params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds):
    print(f'Mean total errors: [PARAMS: {params_errors.mean():.6f} | MEASURES: {measurement_errors.mean():.6f} | S2S: {s2s_dists.mean():.6f}]')

    print('\nPARAMS\n========')
    for param_idx in range(params_errors.shape[0]):
        print(f'PCA{param_idx}: {params_errors[param_idx]:.6f}, {params_stds[param_idx]:.6f}')

    print('\nMEASURES\n=========')
    measure_labels = MeshMeasurements.aplabels()
    for meas_idx in range(measurement_errors.shape[0]):
        print(f'{measure_labels[meas_idx]}: {(measurement_errors[meas_idx] * 1000):.6f}mm, {(measurement_stds[meas_idx] * 1000):.6f}mm')
    print(f'\nMean: {measurement_errors.mean() * 1000:.6f}, {measurement_stds.mean() * 1000:.6f}')


def log(params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds):
    log_errors(params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)

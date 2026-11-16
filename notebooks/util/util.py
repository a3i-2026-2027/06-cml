import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import igraph as ig
from ortools.linear_solver import pywraplp

def load_communities_data(data_folder, nan_discard_thr=0.05):
    # Read the raw data
    fname = os.path.join(data_folder, 'CommViolPredUnnormalizedData.csv')
    data = pd.read_csv(fname, sep=';', na_values='?')
    # Discard columns
    dcols = list(data.columns[-18:-2]) # directly crime related
    dcols = dcols + list(data.columns[7:12]) # race related
    dcols = dcols + ['nonViolPerPop']
    data = data.drop(columns=dcols)
    # Use relative values
    for aname in data.columns:
        if aname.startswith('pct'):
            data[aname] = data[aname] / 100
        elif aname in ('numForeignBorn', 'persEmergShelt',
                       'persHomeless', 'officDrugUnits',
                       'policCarsAvail', 'policOperBudget', 'houseVacant'):
            data[aname] = data[aname] / (data['pop'] / 100e3)
    # Remove redundant column (a relative columns is already there)
    data = data.drop(columns=['persUrban', 'numPolice',
                              'policeField', 'policeCalls', 'gangUnit'])
    # Discard columns with too many NaN values
    thr = nan_discard_thr * len(data)
    cols = data.columns[data.isnull().sum(axis=0) >= thr]
    cols = [c for c in cols if c != 'violentPerPop']
    data = data.drop(columns=cols)
    # Remove all NaN values
    data = data.dropna()
    # Shuffle
    rng = np.random.default_rng(42)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    return data.iloc[idx]



def build_nn_model(input_shape, output_shape, hidden,
        output_activation='linear', kernel_regularizers=[], scale=None,
        name=None):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for i, h in enumerate(hidden):
        kr = kernel_reguralizers[i] if i < len(kernel_regularizers)-1 else None
        x = layers.Dense(h, activation='relu', kernel_regularizer=kr)(x)
    kr_out = kernel_regularizers[-1] if len(kernel_regularizers) > len(hidden) else None
    model_out = layers.Dense(output_shape[0], activation=output_activation,
            kernel_regularizer=kr_out)(x)
    if scale is not None:
        model_out *= scale
    model = keras.Model(model_in, model_out, name=name)
    return model


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


def plot_training_history(history=None,
        figsize=None, print_final_scores=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


def plot_lr_weights(weights, attributes, cap_num=None,
        figsize=None):
    plt.figure(figsize=figsize)
    # Sort attributes by decreasing absolute weights
    idx = np.argsort(np.abs(weights))[::-1]
    if cap_num is not None:
        idx = idx[:cap_num]
    fontsize = min(8, 300 / len(idx))
    x = np.linspace(0.5, 0.5+len(idx), len(idx))
    plt.bar(x, weights[idx])
    plt.xticks(x, labels=attributes[idx], rotation=45, fontsize=fontsize)
    plt.grid(linestyle=':')
    plt.tight_layout()


def plot_pred_by_protected(data, pred, protected, figsize=None):
    plt.figure(figsize=figsize)
    # Prepare the data for the boxplot
    x, lbls = [], []
    # Append the baseline
    pred = pred.ravel()
    x.append(pred)
    lbls.append('all')
    # Append the sub-datasets
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            x.append(pred[mask])
            lbls.append(f'{aname}={val}')
    plt.boxplot(x, label=lbls)
    plt.grid(linestyle=':')
    plt.tight_layout()


def DIDI_r(data, pred, protected):
    res = 0
    avg = np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            res += abs(avg - np.mean(pred[mask]))
    return res


class CstDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, alpha, thr):
        super(CstDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = alpha
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def call(self, data):
        return self.base_pred(data)

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base_pred(x, training=True)
            # mse = self.compiled_loss(y_true, y_pred)
            mse = self.compute_loss(x, y_true, y_pred)
            # Compute the constraint regularization term
            ymean = tf.math.reduce_mean(y_pred)
            didi = 0
            for aidx, dom in self.protected.items():
                for val in dom:
                    mask = (x[:, aidx] == val)
                    didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
            cst = tf.math.maximum(0.0, didi - self.thr)
            loss = mse + self.alpha * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


class LagDualDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, thr):
        super(LagDualDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = tf.Variable(0., name='alpha')
        self.thr = thr
        # Dual multiplier optimizer
        self.dual_optimizer = keras.optimizers.Adam()
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def compile(self, *pargs, **nargs):
        super().compile(*pargs, **nargs)
        self.dual_optimizer.build([self.alpha])

    def call(self, data):
        return self.base_pred(data)

    def __custom_loss(self, x, y_true, sign=1):
        y_pred = self.base_pred(x, training=True)
        # loss, mse, cst = self.__custom_loss(x, y_true, y_pred)
        # mse = self.compiled_loss(y_true, y_pred)
        mse = self.compute_loss(x, y_true, y_pred)
        # Compute the constraint regularization term
        ymean = tf.math.reduce_mean(y_pred)
        didi = 0
        for aidx, dom in self.protected.items():
            for val in dom:
                mask = (x[:, aidx] == val)
                didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
        cst = tf.math.maximum(0.0, didi - self.thr)
        loss = mse + self.alpha * cst
        return sign*loss, mse, cst

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables

        # Update the network weights
        grads = tape.gradient(loss, tr_vars)
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=-1)

        grads = tape.gradient(loss, self.alpha)

        # self.optimizer.apply_gradients(zip([grads], [self.alpha]))
        self.dual_optimizer.apply_gradients(zip([grads], [self.alpha]))

        # Track the loss change
        self.ls_tracker.update_state(-loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


def load_multilabel_dataset(fname,
                            label_num,
                            is_sparse=False,
                            test_fraction=0.2,
                            standardize=True,
                            seed=42):
    # X, y, in_meta, out_meta = load_from_arff(fname, label_num,
    #                       label_location='end' if ftype == 'mulan' else 'start',
    #                       load_sparse=is_sparse,
    #                       return_attribute_definitions=True)
    data = pd.read_csv(fname)
    # Extrat names
    in_names = data.columns[:-label_num]
    out_names = data.columns[-label_num:]
    # in_names = [t[0] for t in in_meta]
    # out_names = [t[0] for t in out_meta]
    X = pd.DataFrame(data=data[in_names], columns=in_names)
    y = pd.DataFrame(data=data[out_names], columns=out_names)
    # Split training and test data
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_fraction, random_state=seed)
    # Apply standardization
    if standardize:
        X_scaler = StandardScaler()
        X_tr = X_scaler.fit_transform(X_tr)
        X_ts = X_scaler.transform(X_ts)
    return X_tr, X_ts, y_tr, y_ts


def get_multi_label_accuracy(y_true, y_pred, series_name=None):
    data = [accuracy_score(y_true.values[:, c], y_pred[:, c])
            for c in range(y_true.shape[1])]
    res = pd.Series(data=data, name=series_name)
    return res


def get_forbidden_pairs(y, cutoff_quantile=0):
    lnum = y.shape[1]
    # Obtain counts for all pairs
    counts = {}
    for c0 in range(lnum):
        for c1 in range(c0+1, lnum):
            # count occurrences
            counts[c0, c1] = np.sum(y.iloc[:, c0] & y.iloc[:, c1])
    # Compute the cutoff threshold
    thr = np.quantile(list(counts.values()), q=cutoff_quantile)
    # Identify pairs with a count below the threshold
    forbidden = set((c0, c1) for (c0, c1), v in counts.items() if v <= thr)
    return forbidden


def forbidden_pair_violation(y, forbidden, normalize=True):
    labels = []
    violations = []
    y = y.astype('bool')
    for (h, k) in sorted(forbidden):
        labels.append(f'({h},{k})')
        violations.append((y[:, h] & y[:, k]).sum())
    res = pd.Series(index=labels, data=violations)
    if normalize:
        res /= len(y)
    return res


def plot_forbidden_pairs(nclasses, forbidden, figsize=None):
    g = ig.Graph(n=nclasses, edges=forbidden)
    layout = g.layout("kamada_kawai")
    g.vs['color'] = 'bisque'
    fig, ax = plt.subplots(figsize=figsize)
    ig.plot(g, layout=layout, target=ax)


def project_multilabel_classification(p,
                                      forbidden,
                                      solver='SAT',
                                      tlim=10):
    # Extend p into a two-dimensional structure
    if len(p.shape) == 1:
        p = p.reshape(1, -1)
    # Detect the number of labels
    label_num = p.shape[1]
    # Build a solver object
    slv = pywraplp.Solver.CreateSolver(solver)
    if not slv:
        raise ValueError(f'Could not build solver type {solver}')
    # Define the main decision variables
    z = [slv.IntVar(0, 1, 'z_{c}') for c in range(label_num)]
    # Define the constraints
    for c0, c1 in forbidden:
        slv.Add(z[c0] + z[c1] <= 1)
    # Project all prediction vectors
    res = np.zeros(p.shape)
    for ii in range(p.shape[0]):
        lpc = np.log(p[ii, :] + 1e-4)
        clpc = np.log(1 - p[ii, :] + 1e-4)
        # Define the objective
        obj = - sum([z[c] * float(lpc[c]) + (1 - z[c])*float(clpc[c])
                     for c in range(label_num)])
        slv.Minimize(obj)
        # Define a time limit
        slv.SetTimeLimit(1000*tlim)
        # Solve the problem
        status = slv.Solve()
        # Check the status
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise ValueError(f'Invalid projection problem for prediction {ii}')
        # Store the projected values
        for c in range(label_num):
            if np.round(z[c].solution_value()) == 1:
                res[ii, c] = 1
    # Return results
    return res

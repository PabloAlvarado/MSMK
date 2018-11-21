# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gpflow
import tensorflow as tf
import numpy as np
import itertools


def mvhermgauss(means, covs, H, D):
        """
        Return the evaluation locations, and weights for several multivariate Hermite-Gauss quadrature runs.
        :param means: NxD
        :param covs: NxDxD
        :param H: Number of Gauss-Hermite evaluation points.
        :param D: Number of input dimensions. Needs to be known at call-time.
        :return: eval_locations (H**DxNxD), weights (H**D)
        """
        N = tf.shape(means)[0]
        #gh_x, gh_w = gpflow.kernels.hermgauss(H)
        gh_x, gh_w = gpflow.likelihoods.hermgauss(H)
        xn = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
        wn = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
        cholXcov = tf.cholesky(covs)  # NxDxD
        X = 2.0 ** 0.5 * tf.batch_matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)),
                                         adj_y=True) + tf.expand_dims(means, 2)  # NxDxH**D
        Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, D))  # H**DxNxD
        return Xr, wn * np.pi ** (-D * 0.5)


class ModLik(gpflow.likelihoods.Likelihood):
    def __init__(self):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.noise_var = gpflow.param.Param(1.0)

    def logp(self, F, Y):
        f1, f2, g1, g2 = F[:, 0], F[:, 1], F[:,2], F[:,3]
        y = Y[:, 0]

        sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
        sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
        mean = sigma_g1 * f1 + sigma_g2 * f2   #Instead of sigmoid function we use the softmax.
        return gpflow.densities.gaussian(y, mean, self.noise_var).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        D = 4  # Number of input dimensions (increased from 2 to 4)
        H = 5 # number of Gauss-Hermite evaluation points. (reduced from 10 to 3)
        Xr, w = mvhermgauss(Fmu, tf.matrix_diag(Fvar), H, D)
        w = tf.reshape(w, [-1, 1])
        f1, f2, g1, g2 = Xr[:, 0], Xr[:, 1], Xr[:, 2], Xr[:, 3]
        y = tf.tile(Y, [H**D, 1])[:, 0]
        sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
        sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive

        mean =  sigma_g1 * f1 + sigma_g2 * f2  #Instead of sigmoid function we use the softmax.
        evaluations = gpflow.densities.gaussian(y, mean, self.noise_var)
        evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w), tf.shape(Fmu)[0]])))
        return tf.matmul(evaluations, w)

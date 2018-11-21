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
import numpy as np
import gpflow
from gpflow import settings
import tensorflow as tf
from sig_likelihood import ModLik


float_type = settings.dtypes.float_type
jitter = settings.numerics.jitter_level


class ModGP(gpflow.model.Model):
    def __init__(self, X, Y, kern1, kern2, kern3, kern4, Z):
        gpflow.model.Model.__init__(self)
        self.X, self.Y = X, Y
        self.kern1, self.kern2, self.kern3, self.kern4 =  kern1, kern2, kern3, kern4
        self.likelihood = ModLik()
        self.Z = Z
        self.num_inducing = Z.shape[0]
        self.num_data = X.shape[0]

        #K_f1 = kern1.compute_K_symm(self.Z)
        #K_f2 = kern2.compute_K_symm(self.Z)
        #aux1 = np.random.multivariate_normal(np.zeros(self.Z.shape[0]), K_f1).reshape(-1, 1)
        #aux1 = aux1 / np.max(np.abs(aux1))
        #aux2 = np.random.multivariate_normal(np.zeros(self.Z.shape[0]), K_f2).reshape(-1, 1)
        #aux2 = aux2 / np.max(np.abs(aux2))

        #self.q_mu1 = gpflow.param.Param(aux1)
        #self.q_mu2 = gpflow.param.Param(aux2)
        self.q_mu1 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu2 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu3 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu4 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)

        self.q_sqrt1 = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt2 = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt3 = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt4 = gpflow.param.Param(q_sqrt.copy())

    def build_prior_KL(self):
        K1 = self.kern1.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * jitter
        KL1 = gpflow.kullback_leiblers.gauss_kl(self.q_mu1, self.q_sqrt1, K1)

        K2 = self.kern2.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * jitter
        KL2 = gpflow.kullback_leiblers.gauss_kl(self.q_mu2, self.q_sqrt2, K2)

        K3 = self.kern3.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * jitter
        KL3 = gpflow.kullback_leiblers.gauss_kl(self.q_mu3, self.q_sqrt3, K3)

        K4 = self.kern4.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * jitter
        KL4 = gpflow.kullback_leiblers.gauss_kl(self.q_mu4, self.q_sqrt4, K4)

        return KL1 + KL2 + KL3 + KL4

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = gpflow.conditionals.conditional(self.X, self.Z, self.kern1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1, full_cov=False, whiten=False)

        fmean2, fvar2 = gpflow.conditionals.conditional(self.X, self.Z, self.kern2, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2, full_cov=False, whiten=False)

        fmean3, fvar3 = gpflow.conditionals.conditional(self.X, self.Z, self.kern3, self.q_mu3,
                                                        q_sqrt=self.q_sqrt3, full_cov=False, whiten=False)

        fmean4, fvar4 = gpflow.conditionals.conditional(self.X, self.Z, self.kern4, self.q_mu4,
                                                        q_sqrt=self.q_sqrt4, full_cov=False, whiten=False)

        fmean = tf.concat(1, [fmean1, fmean2, fmean3, fmean4])
        fvar  = tf.concat(1, [fvar1, fvar2, fvar3, fvar4])
        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f1(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern1, self.q_mu1,
                                               q_sqrt=self.q_sqrt1, full_cov=False, whiten=False)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f2(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern2, self.q_mu2,
                                               q_sqrt=self.q_sqrt2, full_cov=False, whiten=False)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g1(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern3, self.q_mu3,
                                               q_sqrt=self.q_sqrt3, full_cov=False, whiten=False)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g2(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern4, self.q_mu4,
                                               q_sqrt=self.q_sqrt4, full_cov=False, whiten=False)

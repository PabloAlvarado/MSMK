import numpy as np
import GPflow
import scipy as sp
import scipy.io as sio
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import tensorflow as tf
import siggp




def Lorentzian(p, x):
    '''Lorentzian function
    See http://mathworld.wolfram.com/LorentzianFunction.html'''
    return (p[0]*p[1] / 1.) / ( (4.*np.square(np.pi))*(x - p[2]/(2.*np.pi)) **2. + p[1]**2. )




def Lloss(p, x, y):
    '''Loss function to fit a Lorentzian function to data "y" '''
    f =  np.sqrt(np.square(Lorentzian(p, x) - y).mean())
    return f




def LorM(x, s, l, f):
    '''Mixture of Lorentzian functions'''
    lm = np.zeros((x.shape))
    for i in range(0, s.size):
        lm += (s[i]*l[i] / 1.) / ( (4.*np.square(np.pi))*(x - f[i]/(2.*np.pi)) **2. + l[i]**2. )
    return lm




def MaternSM(x, s, l, f):
    '''Matern spectral mixture function'''
    ker = np.zeros((x.shape))
    for i in range(0, s.size):
        ker += s[i] * np.exp(-l[i]*np.abs(x)) * np.cos(f[i]*x)
    return ker




def ker_msm(s, l, f, Nh):
    '''Matern spectral mixture kernel
    Input:
        s  : variance vector
        l  : Matern lengthscales vector
        f  : frequency vector (Hz)
        Nh : number of components
    Output:
        GPflow kernel object'''
    per = 1./(2.*np.pi*f)
    kexp0 = GPflow.kernels.Matern12(input_dim=1, variance=1.0, lengthscales=l[0])
    kcos0 = GPflow.kernels.Cosine(input_dim=1, variance=s[0], lengthscales=per[0])
    ker = kexp0*kcos0
    for n in range (1, Nh):
        kexp = GPflow.kernels.Matern12(input_dim=1, variance=1.0, lengthscales=l[n])
        kcos = GPflow.kernels.Cosine(input_dim=1, variance=s[n], lengthscales=per[n])
        ker += kexp*kcos
    return ker




def learnparams(X, S, Nh):
    '''Learn parameters in frequency domain.
    Input:
        X: frequency vector (Hz)
        S: Magnitude Fourier transform signal
        Nh: number of maximun harmonic to learn
    Output:
        matrix of parameters'''
    Np = 3 # number of parameters per Lorentzian
    Pstar = np.zeros((Nh,Np))
    Shat = S.copy()
    count = 0
    for i in range(0, Nh):
        idx = np.argmax(Shat)
        if Shat[idx] > 0.025*S.max():
            count += 1
            a = idx - 100
            b = idx + 100
            x = X
            y = Shat
            p0 = np.array([0.1, 0.1, 2.*np.pi*X[idx]])
            phat = sp.optimize.minimize(Lloss, p0, method='L-BFGS-B', args=(x, y), tol = 1e-10, options={'disp': False})
            pstar = phat.x
            Pstar[i,:] = pstar
            learntfun = Lorentzian(pstar, x)
            Shat = np.abs(learntfun -  Shat)
            Shat[a:b,] = 0.
    s_s, l_s, f_s = np.hsplit(Pstar[0:count,:], 3)
    return s_s, 1./l_s, f_s/(2.*np.pi),




def softmax(x,y,z):
    """This function computes the softmax function
    for the vectors x, y, z ."""
    Nor = np.exp(x) + np.exp(y) + np.exp(z) #normalizer
    return np.exp(x)/Nor, np.exp(y)/Nor, np.exp(z)/Nor




def logistic(x):
    return 1./(1+ np.exp(-x))




def pitch(filename, windowsize=16000):
    '''Pitch detector'''
    sf, y = wav.read(filename) #load test data
    N = np.size(y)
    y = y.astype(np.float64)
    y = y / np.max(np.abs(y))
    X = np.linspace(0, (N-1.)/sf, N)
    Xt1, yt1 = X[0:2*sf], y[0:2*sf] # set training data pitch 1
    Xt2, yt2 = X[2*sf:4*sf], y[2*sf:4*sf] # set training data pitch 2

    Xtest1, ytest1 = X[0:4*sf], y[0:4*sf]
    Xtest2, ytest2 = X[6*sf:8*sf], y[6*sf:8*sf]
    Xtest = np.hstack((Xtest1, Xtest2)).reshape(-1,1)
    ytest = np.hstack((ytest1, ytest2)).reshape(-1,1)

    Nt = yt1.size # Number of sample points for training
    y1F, y2F = sp.fftpack.fft(yt1), sp.fftpack.fft(yt2) #FT training data
    T = 1.0 / sf # sample spacing
    F = np.linspace(0.0, 1.0/(2.0*T), Nt/2)
    S1 = 2.0/Nt * np.abs(y1F[0:Nt/2]) # spectral density training data 1
    S2 = 2.0/Nt * np.abs(y2F[0:Nt/2]) # spectral density training data 2

    # Parameters learning
    Nh = 15 # max num of harmonics allowed
    s1, l1, f1 = learnparams(X=F, S=S1, Nh=Nh)
    s2, l2, f2 = learnparams(X=F, S=S2, Nh=Nh)
    par1 = [s1, l1, f1]
    par2 = [s2, l2, f2]

    # define kernel components and activations
    k_f1 = ker_msm(s=s1, l=l1, f=f1, Nh=s1.size) #
    k_f2 = ker_msm(s=s2, l=l2, f=f2, Nh=s2.size)
    k_g1 = GPflow.kernels.Matern12(input_dim=1, lengthscales=1., variance=0.3767)
    k_g2 = GPflow.kernels.Matern12(input_dim=1, lengthscales=1., variance=0.3767)


    ### Pitch detection
    Ns = windowsize #number of samples per window
    winit = 0 #initial window to analyse
    wfinish = Xtest.size/Ns # final window to analyse
    Nw = wfinish - winit # number of windows to analyse
    # initialize arrays to save results
    save_X = np.zeros((Ns,Nw))
    save_y = np.zeros((Ns,Nw))
    save_muf1 = np.zeros((Ns,Nw))
    save_muf2 = np.zeros((Ns,Nw))
    save_mug1 = np.zeros((Ns,Nw))
    save_mug2 = np.zeros((Ns,Nw))
    save_mug0 = np.zeros((Ns,Nw))
    save_varf1 = np.zeros((Ns,Nw))
    save_varf2 = np.zeros((Ns,Nw))
    save_varg1 = np.zeros((Ns,Nw))
    save_varg2 = np.zeros((Ns,Nw))
    save_varg0 = np.zeros((Ns,Nw))
    save_ll = np.zeros((Nw,1))

    noise_var = 1e-3
    count = 0 # count number of windows analysed so far
    for i in range (winit, wfinish):
        count = count + 1
        X = Xtest[ i*Ns : (i+1)*Ns]
        y = ytest[ i*Ns : (i+1)*Ns]
        #Z = X[::100].copy()
        Z = np.vstack(( (X[::100].copy()).reshape(-1,1) ,(X[-1].copy()).reshape(-1,1)  ))


        m = siggp.ModGP(X, y, k_f1, k_f2, k_g1, k_g2, Z)
        m.kern1.fixed = True
        m.kern2.fixed = True
        m.kern3.fixed = False
        m.kern4.fixed = False
        m.likelihood.noise_var = noise_var
        m.likelihood.noise_var.fixed = True
        print('Analysing window number ', count, ' total number of windows to analyse ', Nw)
        m.optimize(disp=1, maxiter = 25)
        mu_f1, var_f1 = m.predict_f1(X)
        mu_f2, var_f2 = m.predict_f2(X)
        mu_g1, var_g1 = m.predict_g1(X)
        mu_g2, var_g2 = m.predict_g2(X)
        save_X[:,i-winit] = X.reshape(-1)
        save_y[:,i-winit] = y.reshape(-1)
        save_muf1[:,i-winit] = mu_f1.reshape(-1)
        save_muf2[:,i-winit] = mu_f2.reshape(-1)
        save_mug1[:,i-winit] = mu_g1.reshape(-1)
        save_mug2[:,i-winit] = mu_g2.reshape(-1)
        save_varf1[:,i-winit] = var_f1.reshape(-1)
        save_varf2[:,i-winit] = var_f2.reshape(-1)
        save_varg1[:,i-winit]= var_g1.reshape(-1)
        save_varg2[:,i-winit]= var_g2.reshape(-1)
        #ll = m.compute_log_likelihood()
        #save_ll[i-winit,0] =  ll


    X = np.reshape(save_X, (-1,1), order = 'F')
    y = np.reshape(save_y, (-1,1), order = 'F')
    mu_f1 = np.reshape(save_muf1, (-1,1), order = 'F')
    mu_f2 = np.reshape(save_muf2, (-1,1), order = 'F')
    mu_g1 = np.reshape(save_mug1, (-1,1), order = 'F')
    mu_g2 = np.reshape(save_mug2, (-1,1), order = 'F')
    var_f1 = np.reshape(save_varf1, (-1,1), order = 'F')
    var_f2 = np.reshape(save_varf2, (-1,1), order = 'F')
    var_g1 = np.reshape(save_varg1, (-1,1), order = 'F')
    var_g2 = np.reshape(save_varg2, (-1,1), order = 'F')
    ll = np.reshape(save_ll, (-1,1), order = 'F')

    np.savez_compressed('SIG_FL_results',
                                  X = X,
                                  y = y,
                                  Xt1 = Xt1,
                                  yt1 = yt1,
                                  Xt2 = Xt2,
                                  yt2 = yt2,
                                  F   = F,
                                  S1 = S1,
                                  S2 = S2,
                                  params = [par1, par2],
                                  mu_f1 = mu_f1, var_f1 = var_f1,
                                  mu_f2 = mu_f2, var_f2 = var_f2,
                                  mu_g1 = mu_g1, var_g1 = var_g1,
                                  mu_g2 = mu_g2, var_g2 = var_g2)

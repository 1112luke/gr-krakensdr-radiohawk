#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 KrakenRF Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
import numpy.linalg as lin
import scipy as scipy

from gnuradio import gr

class doa_music(gr.sync_block):
    """
    docstring for block doa_music
    """
    def __init__(self, vec_len=1048576, freq=433.0, array_dist=0.33, num_elements=5, array_type='UCA', processing_alg='MUSIC'):
        gr.sync_block.__init__(self,
            name="DOA MUSIC",
            in_sig=[(np.complex64, vec_len)] * num_elements,
            out_sig=[(np.float32, 360)])
            
        print("INITIALIZED WITH PROCESSING ALG = ", processing_alg)

        self.cpi_size = vec_len
        self.freq = freq
        self.array_dist = array_dist
        self.num_elements = num_elements
        self.array_type = array_type
        self.processing_alg = processing_alg

        wavelength = 300 / freq
        if array_type == 'UCA':
            inter_elem_spacing = (np.sqrt(2) * array_dist * np.sqrt(1 - np.cos(np.deg2rad(360 / num_elements))))
            wavelength_mult = inter_elem_spacing / wavelength
        else:
            wavelength_mult = array_dist / wavelength
        
        self.scanning_vectors = self.gen_scanning_vectors(self.num_elements, wavelength_mult, self.array_type, 0)

        print("wavelength mult: " + str(wavelength_mult))
        
    def work(self, input_items, output_items):

        #print("input items size: " + str(np.shape(input_items)))
        processed_signal = np.empty((self.num_elements, self.cpi_size), dtype=np.complex64)

        for i in range(self.num_elements):
            processed_signal[i, :] = input_items[i][0][:]

        #decimated_processed_signal = signal.decimate(processed_signal, 100, n=100 * 2, ftype='fir')
        # Doing decimation in GNU Radio blocks, or uncomment to do decimation in scipy
        decimated_processed_signal = processed_signal
       
        
        if self.processing_alg == "MUSIC":
            print("USING STOCK MUSIC")

            #stock music algorithm
            R = self.corr_matrix(decimated_processed_signal)
            DOA_MUSIC_res = self.DOA_MUSIC(R, self.scanning_vectors, signal_dimension=1)

        elif self.processing_alg == "Correlation_MUSIC":
            print("USING CORRELATION MUSIC")

            #feed in correlation matrices
            chirp, _, _ = self.get_chirp(num = 10)
            corr_size = self.crosscorrelate(chirp, processed_signal[0]).size
            correlated_signal = np.empty((self.num_elements, corr_size), dtype = np.complex64)
            for i in range(self.num_elements):
                correlated_signal[i,:] = self.crosscorrelate(chirp, processed_signal[i])
            R = self.corr_matrix(correlated_signal)
            DOA_MUSIC_res = self.DOA_MUSIC(R, self.scanning_vectors, signal_dimension=1)

        elif self.processing_alg == "ULT":
            print("ULT")

            #get y
            y = np.fromfile("/home/krakenrf/gr-krakensdr/python/krakensdr/references/chopped.cfile", dtype = np.complex64)
            x = decimated_processed_signal

            print("Y", y)

            DOA_MUSIC_res = self.DOA_ULT(x, y)


        doa_plot = self.DOA_plot_util(DOA_MUSIC_res)
        output_items[0][0][:] = doa_plot
        return len(output_items[0])


    def corr_matrix(self, X):
        N = X[0, :].size
        R = np.dot(X, X.conj().T)
        R = np.divide(R, N)
        return R
    
    def auto_corr_matrix(self, X, Y):
        N = X[0, :].size
        R = np.dot(X, Y.conj().T)
        R = np.divide(R, N)
        return R


    def gen_scanning_vectors(self, M, DOA_inter_elem_space, type, offset):
        thetas = np.linspace(0, 359, 360)  # Remember to change self.DOA_thetas too, we didn't include that in this function due to memoization cannot work with arrays
        if type == "UCA":
            x = DOA_inter_elem_space * np.cos(2 * np.pi / M * np.arange(M))
            y = -DOA_inter_elem_space * np.sin(2 * np.pi / M * np.arange(M))
        elif "ULA":
            x = np.zeros(M)
            y = -np.arange(M) * DOA_inter_elem_space

        scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
        for i in range(thetas.size):
            scanning_vectors[:, i] = np.exp(
                1j * 2 * np.pi * (x * np.cos(np.deg2rad(thetas[i] + offset)) + y * np.sin(np.deg2rad(thetas[i] + offset))))

        return np.ascontiguousarray(scanning_vectors)

    def ULT_gen_scanning_vector(self, M, theta, phi):

        a = np.zeros(M, dtype=np.complex64)
        beta = 2 * np.pi * np.arange(M) / M
        for m in range(M):
            phase = (2 * np.pi * self.array_dist / (300/self.freq)) * np.sin(theta * np.pi/180) * np.cos((phi * np.pi/180) - beta[m])
            a[m] = np.exp(1j * phase)

        return a
        
        
    def DOA_MUSIC(self, R, scanning_vectors, signal_dimension, angle_resolution=1):
        # --> Input check
        if R[:, 0].size != R[0, :].size:
            print("ERROR: Correlation matrix is not quadratic")
            return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

        if R[:, 0].size != scanning_vectors[:, 0].size:
            print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
            return np.ones(1, dtype=np.complex64) * -2

        ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
        M = R[:, 0].size  # np.size(R, 0)

        # --- Calculation ---
        # Determine eigenvectors and eigenvalues
        sigmai, vi = lin.eig(R)
        sigmai = np.abs(sigmai)

        idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
        vi = vi[:, idx]

        # Generate noise subspace matrix
        noise_dimension = M - signal_dimension

        E = np.zeros((M, noise_dimension), dtype=np.complex64)
        for i in range(noise_dimension):
            E[:, i] = vi[:, i]

        E_ct = E @ E.conj().T
        theta_index = 0
        for i in range(scanning_vectors[0, :].size):
            S_theta_ = scanning_vectors[:, i]
            S_theta_ = np.ascontiguousarray(S_theta_.T)
            ADORT[theta_index] = 1 / np.abs(S_theta_.conj().T @ E_ct @ S_theta_)
            theta_index += 1

        return ADORT

    def DOA_ULT(self, x, y):
        #x is 5 x cpi_size array

        '''
        # --> Input check
        if R[:, 0].size != R[0, :].size:
            print("ERROR: Correlation matrix is not quadratic")
            return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

        if R[:, 0].size != scanning_vectors[:, 0].size:
            print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
            return np.ones(1, dtype=np.complex64) * -2
        

        ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
        M = R[:, 0].size  # np.size(R, 0)

        '''

        # --- Calculation ---
        #compute crosscorrelation and autocorrelation matrices
        R_xy = (1/self.cpi_size) * np.dot(x, np.conj(y).T)
        R_yy = (1/self.cpi_size) * np.dot(y, np.conj(y).T)

        B_hat = R_xy/R_yy

        thetas = np.deg2rad(np.linspace(0, 90, 91)) 
        phis = np.deg2rad(np.linspace(-180, 189, 360))

        outputs = np.zeros((len(thetas), len(phis)), dtype=np.float32)

        for i in range(thetas.size):
            for j in range(phis.size):
                outputs[i, j] = np.abs(np.vdot(self.ULT_gen_scanning_vector(5, i, j), B_hat))**2

        return outputs[90, :]

    def DOA_plot_util(self, DOA_data, log_scale_min=-100):
        """
            This function prepares the calulcated DoA estimation results for plotting.
            - Noramlize DoA estimation results
            - Changes to log scale
        """

        DOA_data = np.divide(np.abs(DOA_data), np.max(np.abs(DOA_data)))  # Normalization
        DOA_data = 10 * np.log10(DOA_data)  # Change to logscale

        for i in range(len(DOA_data)):  # Remove extremely low values
            if DOA_data[i] < log_scale_min:
                DOA_data[i] = log_scale_min

        return DOA_data
    
    def crosscorrelate(self, arr1, arr2):
        #assume arr2 is the larger array
        match = scipy.signal.correlate(arr1, arr2, mode = "valid")

        return np.ascontiguousarray(match)
        '''
        match = np.abs(match) #get positive part

        #smooth
        # Calculate rolling average with a window of samplesize/10
        match = pd.Series(match)
        window_size = len(match)//8
        match_smooth = match.rolling(window=window_size).mean()
        '''
        
        #find peaks -- SUBJECT TO MUCH TUNING!!!
        #peaks = scipy.signal.find_peaks(match_smooth, distance = len(match_smooth)//8, height = 3000)

    def get_chirp(self, sample_rate = 2.048e6, num = 1):
        """
        Generates a LoRa chirp signal at a specified SDR sample rate.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The complex baseband chirp signal.
                - numpy.ndarray: The time vector (for plotting).
                - numpy.ndarray: The k-index vector used in the chirp formula.
        """

        sdr_sample_rate = sample_rate # Simulate sample rate of sdr.sample_rate = 2.048e6

        # Parameters
        B = 62.5e3  # (125 kHz) bandwidth
        T = 1 / B  # This 'T' is the base sampling period for the chirp's definition (1/Bandwidth).
                # It is NOT the SDR's actual sampling period (1/sdr_sample_rate).
        SF = 6  # spreading factor {7,8,9,10,11,12}
        T_s = (2**SF) * T  # symbol period (total duration of one chirp)

        # For a preamble chirp, the symbol value is 0.
        # In a real LoRa system, this 'w' array would represent the bits
        # of the data symbol being transmitted.
        w = [0,0,0,0,0,0,0,0,0,0,0,0]

        # Calculate the decimal symbol value from the binary 'w' array.
        # For a preamble, this will sum to 0.

        chirps = []

        symbol = 0
        for h in range(SF):
            symbol += w[h] * (2**h)

        for i in range(num):

            # Calculate the number of samples for the chirp duration based on the SDR's sample rate.
            num_samples = int(T_s * sdr_sample_rate)

            # Create the time vector 't' for plotting, spanning the symbol period 'T_s'
            # with 'num_samples' points.
            t = np.linspace(start=0, stop=T_s, num=num_samples, endpoint=False)

            # The 'k' in your chirp formula represents a normalized index (or "chip index")
            # that effectively sweeps from 0 to (2**SF - 1) over the duration of the chirp.
            # To maintain the "exact same signal" shape at the new sample rate, 'k' must
            # also span this range (0 to 2**SF) but consist of 'num_samples' points.
            # np.linspace is ideal for this, mapping the 'num_samples' evenly across the
            # conceptual range of 0 to 2**SF.
            k = np.linspace(start=0, stop=(2**SF), num=num_samples, endpoint=False)


            # LoRa chirp formula (baseband representation):
            # This formula generates a complex chirp signal.
            # The term '((symbol + k) % (2**SF))' handles the cyclic shift based on the symbol value,
            # ensuring the chirp wraps around within the 2**SF range.
            # The division by '(2**SF)' normalizes the phase, and the multiplication by 'k'
            # creates the quadratic phase characteristic of a linear chirp.
            chirp = np.exp(1j * 2 * np.pi * ((symbol + k) % (2**SF)) / (2**SF) * k)

            chirps.append(chirp)

        output = np.concatenate(chirps)
        return output, t, k # Return t and k for plotting and verification
        

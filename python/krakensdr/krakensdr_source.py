#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 KrakenRF Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import socket
import _thread
import queue
from threading import Thread
from threading import Lock
from gnuradio import gr
from struct import pack, unpack
import sys

class krakensdr_source(gr.sync_block):
    """
    docstring for block krakensdr_source
    """
    def __init__(self, ipAddr="127.0.0.1", port=5000, ctrlPort=5001, numChannels=5, freq=416.588, gain=[10.0], debug=False):
        gr.sync_block.__init__(self,
            name="KrakenSDR Source",
            in_sig=None,
            out_sig=[np.complex64] * numChannels)

        self.valid_gains = [0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]

        self.ipAddr = ipAddr
        self.port = port
        self.ctrlPort = ctrlPort
        self.numChannels = numChannels
        self.freq = int(freq*10**6)
        self.gain = gain
        self.debug = debug
        self.iq_header = IQHeader()

        # Data Interface
        self.socket_inst = socket.socket()
        self.receiver_connection_status = False
        self.receiverBufferSize = 2 ** 18

        # Control interface
        self.ctr_iface_socket = socket.socket()
        self.ctr_iface_port = self.ctrlPort
        self.ctr_iface_thread_lock = Lock()

        # Custom Output Interface
        self.tcp_connected = False
        self.tcpout_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpout_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcpout_port = 3333
        self.tcpout_socket.bind(('', tcpout_port))
        self.tcpout_socket.listen(5)
        self.tcpout_server_thread = Thread(target=self.tcpout_server)
        self.tcpout_server_thread.start()
        self.tcpout_lock = Lock()

        self.tcp_send_queue = queue.Queue()
        self.tcp_send_thread = Thread(target=self.tcp_send_loop)
        self.tcp_send_thread.start()

        # Init cpi_len from heimdall header
        self.get_iq_online()
        while self.iq_header.cpi_length == 0:
            self.get_iq_online()
            
        self.cpi_len = self.iq_header.cpi_length
        self.total_fetched = self.iq_header.cpi_length

        self.iq_samples = None
        self.iq_sample_queue = queue.Queue(10)
        
        self.stop_threads = False
        self.buffer_thread = Thread(target=self.buffer_iq_samples)
        self.buffer_thread.start()

    def tcpout_server(self):
        self.tcpout_socket.settimeout(1.0)
        while not self.stop_threads:
            try:
                client_socket, addr = self.tcpout_socket.accept()
                with self.tcpout_lock:
                    if self.tcp_connected:
                        client_socket.close()
                        continue
                    self.c = client_socket
                    self.addr = addr
                    self.tcp_connected = True
                    print(f"Got connection from {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"TCP accept error: {e}")
                break

    def tcp_send_loop(self):
        BATCH_SIZE = 5242880  # Define batch size
        while not self.stop_threads:
            batch = []
            batch_size_bytes = 0
            
            # Collect data up to batch size
            while batch_size_bytes < BATCH_SIZE and not self.stop_threads:
                try:
                    data = self.tcp_send_queue.get(timeout=1)
                    data_bytes = data.tobytes()
                    batch_size_bytes += len(data_bytes)
                    batch.append(data_bytes)
                except queue.Empty:
                    continue

            # Send batched data
            if batch and self.tcp_connected:
                with self.tcpout_lock:
                    if self.tcp_connected:
                        try:
                            for data_bytes in batch:
                                self.c.sendall(data_bytes)
                        except Exception as e:
                            print(f"TCP send error: {e}")
                            with self.tcpout_lock:
                                self.c.close()
                                self.tcp_connected = False

    def buffer_iq_samples(self):
        while not self.stop_threads:
            if self.debug:
                self.iq_header.dump_header()
                
            iq_samples = self.get_iq_online()
            
            if self.iq_header.frame_type == self.iq_header.FRAME_TYPE_DATA:
                try:
                    self.iq_sample_queue.put_nowait(iq_samples)
                    # Also add to TCP send queue
                    for n in range(self.numChannels):
                        self.tcp_send_queue.put_nowait(iq_samples[n])
                except Exception as e:
                    print(f"Failed to put IQ Samples into the Queue: {e}")

    def work(self, input_items, output_items):
        if self.total_fetched == self.cpi_len:
            try:
                self.iq_samples = self.iq_sample_queue.get(True, 3)
            except Exception as e:
                print(f"Failed to get IQ Samples: {e}")
                return 0
            self.total_fetched = 0

        fetch_left = self.cpi_len - self.total_fetched
        output_items_req = len(output_items[0])
        output_items_now = min(output_items_req, fetch_left)

        try:
            for n in range(self.numChannels):
                data_slice = self.iq_samples[n, self.total_fetched:self.total_fetched + output_items_now]
                output_items[n][0:output_items_now] = data_slice
                # Add to TCP send queue
                if self.tcp_connected:
                    self.tcp_send_queue.put_nowait(data_slice)
        except Exception as e:
            print(f"Failed to write output_items: {e}")
            return 0

        self.total_fetched += output_items_now
        return output_items_now

    def stop(self):
        self.stop_threads = True
        self.buffer_thread.join()
        self.tcpout_server_thread.join()
        self.tcp_send_thread.join()
        with self.tcpout_lock:
            if self.tcp_connected:
                self.c.close()
                self.tcp_connected = False
        self.tcpout_socket.close()
        self.eth_close()
        return True

    def set_gain(self, gain):
        self.gain = gain
        self.set_if_gain(self.gain)

    def set_freq(self, freq):
        self.freq = freq
        self.set_center_freq(int(freq*10**6))

    def eth_connect(self):
        try:
            if not self.receiver_connection_status:
                self.socket_inst.connect((self.ipAddr, self.port))
                self.socket_inst.sendall(str.encode('streaming'))
                test_iq = self.receive_iq_frame()
                self.ctr_iface_socket.connect((self.ipAddr, self.ctr_iface_port))
                self.receiver_connection_status = True
                self.ctr_iface_init()
                self.set_center_freq(self.freq)
                self.set_if_gain(self.gain)
        except Exception as e:
            self.receiver_connection_status = False
            print(f"Ethernet Connection Failed: {e}")
            return -1
        return 0

    def ctr_iface_init(self):
        if self.receiver_connection_status:
            cmd = "INIT"
            msg_bytes = cmd.encode() + bytearray(124)
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except Exception as e:
                print(f"Unable to start communication thread: {e}")

    def ctr_iface_communication(self, msg_bytes):
        with self.ctr_iface_thread_lock:
            print("Sending control message")
            self.ctr_iface_socket.send(msg_bytes)
            reply_msg_bytes = self.ctr_iface_socket.recv(128)
            print("Control interface communication finished")
            status = reply_msg_bytes[0:4].decode()
            if status == "FNSD":
                print("Reconfiguration successfully finished")
            else:
                print(f"Failed to set the requested parameter, reply: {status}")

    def set_center_freq(self, center_freq):
        if self.receiver_connection_status:
            self.freq = int(center_freq)
            cmd = "FREQ"
            freq_bytes = pack("Q", int(center_freq))
            msg_bytes = cmd.encode() + freq_bytes + bytearray(116)
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except Exception as e:
                print(f"Unable to start communication thread: {e}")

    def set_if_gain(self, gain):
        if self.receiver_connection_status:
            cmd = "GAIN"
            for i in range(len(gain)):
                gain[i] = min(self.valid_gains, key=lambda x: abs(x - gain[i]))
            gain_list = [int(i * 10) for i in gain]
            gain_bytes = pack("I" * self.numChannels, *gain_list)
            msg_bytes = cmd.encode() + gain_bytes + bytearray(128 - (self.numChannels + 1) * 4)
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except Exception as e:
                print(f"Unable to start communication thread: {e}")

    def get_iq_online(self):
        if not self.receiver_connection_status:
            fail = self.eth_connect()
            if fail:
                return -1
        self.socket_inst.sendall(str.encode("IQDownload"))
        return self.receive_iq_frame()

    def receive_iq_frame(self):
        total_received_bytes = 0
        iq_header_bytes = bytearray(self.iq_header.header_size)
        view = memoryview(iq_header_bytes)

        while total_received_bytes < self.iq_header.header_size:
            recv_bytes_count = self.socket_inst.recv_into(view, self.iq_header.header_size - total_received_bytes)
            view = view[recv_bytes_count:]
            total_received_bytes += recv_bytes_count

        self.iq_header.decode_header(iq_header_bytes)
        incoming_payload_size = self.iq_header.cpi_length * self.iq_header.active_ant_chs * 2 * int(self.iq_header.sample_bit_depth / 8)
        if incoming_payload_size > 0:
            total_bytes_to_receive = incoming_payload_size
            receiver_buffer_size = 2 ** 18
            total_received_bytes = 0
            iq_data_bytes = bytearray(total_bytes_to_receive + receiver_buffer_size)
            view = memoryview(iq_data_bytes)

            while total_received_bytes < total_bytes_to_receive:
                recv_bytes_count = self.socket_inst.recv_into(view, receiver_buffer_size)
                view = view[recv_bytes_count:]
                total_received_bytes += recv_bytes_count

            self.iq_samples = np.frombuffer(iq_data_bytes[0:total_bytes_to_receive], dtype=np.complex64).reshape(self.iq_header.active_ant_chs, self.iq_header.cpi_length)
            self.iq_frame_bytes = bytearray() + iq_header_bytes + iq_data_bytes
            return self.iq_samples
        else:
            return 0

    def eth_close(self):
        try:
            if self.receiver_connection_status:
                self.socket_inst.sendall(str.encode('q'))
                self.socket_inst.close()
                self.socket_inst = socket.socket()
                exit_message_bytes = "EXIT".encode() + bytearray(124)
                self.ctr_iface_socket.send(exit_message_bytes)
                self.ctr_iface_socket.close()
                self.ctr_iface_socket = socket.socket()
            self.receiver_connection_status = False
        except Exception as e:
            print(f"Error closing Ethernet connections: {e}")
            return -1
        return 0

class IQHeader:
    FRAME_TYPE_DATA = 0
    FRAME_TYPE_DUMMY = 1
    FRAME_TYPE_RAMP = 2
    FRAME_TYPE_CAL = 3
    FRAME_TYPE_TRIGW = 4
    SYNC_WORD = 0x2bf7b95a

    def __init__(self):
        self.header_size = 1024
        self.reserved_bytes = 192
        self.sync_word = self.SYNC_WORD
        self.frame_type = 0
        self.hardware_id = ""
        self.unit_id = 0
        self.active_ant_chs = 0
        self.ioo_type = 0
        self.rf_center_freq = 0
        self.adc_sampling_freq = 0
        self.sampling_freq = 0
        self.cpi_length = 0
        self.time_stamp = 0
        self.daq_block_index = 0
        self.cpi_index = 0
        self.ext_integration_cntr = 0
        self.data_type = 0
        self.sample_bit_depth = 0
        self.adc_overdrive_flags = 0
        self.if_gains = [0] * 32
        self.delay_sync_flag = 0
        self.iq_sync_flag = 0
        self.sync_state = 0
        self.noise_source_state = 0
        self.reserved = [0] * self.reserved_bytes
        self.header_version = 0

    def decode_header(self, iq_header_byte_array):
        iq_header_list = unpack("II16sIIIQQQIQIIQIII" + "I" * 32 + "IIII" + "I" * self.reserved_bytes + "I", iq_header_byte_array)
        self.sync_word = iq_header_list[0]
        self.frame_type = iq_header_list[1]
        self.hardware_id = iq_header_list[2].decode()
        self.unit_id = iq_header_list[3]
        self.active_ant_chs = iq_header_list[4]
        self.ioo_type = iq_header_list[5]
        self.rf_center_freq = iq_header_list[6]
        self.adc_sampling_freq = iq_header_list[7]
        self.sampling_freq = iq_header_list[8]
        self.cpi_length = iq_header_list[9]
        self.time_stamp = iq_header_list[10]
        self.daq_block_index = iq_header_list[11]
        self.cpi_index = iq_header_list[12]
        self.ext_integration_cntr = iq_header_list[13]
        self.data_type = iq_header_list[14]
        self.sample_bit_depth = iq_header_list[15]
        self.adc_overdrive_flags = iq_header_list[16]
        self.if_gains = iq_header_list[17:49]
        self.delay_sync_flag = iq_header_list[49]
        self.iq_sync_flag = iq_header_list[50]
        self.sync_state = iq_header_list[51]
        self.noise_source_state = iq_header_list[52]
        self.header_version = iq_header_list[52 + self.reserved_bytes + 1]

    def encode_header(self):
        iq_header_byte_array = pack("II", self.sync_word, self.frame_type)
        iq_header_byte_array += self.hardware_id.encode() + bytearray(16 - len(self.hardware_id.encode()))
        iq_header_byte_array += pack("IIIQQQIQIIQIII",
                                    self.unit_id, self.active_ant_chs, self.ioo_type, self.rf_center_freq, self.adc_sampling_freq,
                                    self.sampling_freq, self.cpi_length, self.time_stamp, self.daq_block_index, self.cpi_index,
                                    self.ext_integration_cntr, self.data_type, self.sample_bit_depth, self.adc_overdrive_flags)
        for m in range(32):
            iq_header_byte_array += pack("I", self.if_gains[m])
        iq_header_byte_array += pack("IIII", self.delay_sync_flag, self.iq_sync_flag, self.sync_state, self.noise_source_state)
        for m in range(self.reserved_bytes):
            iq_header_byte_array += pack("I", 0)
        iq_header_byte_array += pack("I", self.header_version)
        return iq_header_byte_array

    def dump_header(self):
        print(f"Sync word: {self.sync_word}")
        print(f"Header version: {self.header_version}")
        print(f"Frame type: {self.frame_type}")
        print(f"Hardware ID: {self.hardware_id:16}")
        print(f"Unit ID: {self.unit_id}")
        print(f"Active antenna channels: {self.active_ant_chs}")
        print(f"Illuminator type: {self.ioo_type}")
        print(f"RF center frequency: {self.rf_center_freq/10**6:.2f} MHz")
        print(f"ADC sampling frequency: {self.adc_sampling_freq/10**6:.2f} MHz")
        print(f"IQ sampling frequency {self.sampling_freq/10**6:.2f} MHz")
        print(f"CPI length: {self.cpi_length}")
        print(f"Unix Epoch timestamp: {self.time_stamp}")
        print(f"DAQ block index: {self.daq_block_index}")
        print(f"CPI index: {self.cpi_index}")
        print(f"Extended integration counter {self.ext_integration_cntr}")
        print(f"Data type: {self.data_type}")
        print(f"Sample bit depth: {self.sample_bit_depth}")
        print(f"ADC overdrive flags: {self.adc_overdrive_flags}")
        for m in range(32):
            print(f"Ch: {m} IF gain: {self.if_gains[m]/10:.1f} dB")
        print(f"Delay sync flag: {self.delay_sync_flag}")
        print(f"IQ sync flag: {self.iq_sync_flag}")
        print(f"Sync state: {self.sync_state}")
        print(f"Noise source state: {self.noise_source_state}")

    def check_sync_word(self):
        return 0 if self.sync_word == self.SYNC_WORD else -1
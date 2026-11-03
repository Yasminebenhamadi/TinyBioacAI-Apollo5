#include "ns_core.h"
#include "ns_timer.h"
#include "ns_debug_log.h"
#include "ns_energy_monitor.h"
#include "ns_peripherals_power.h"
#include "ns_pmu_utils.h"

#include <stdint.h>
#include <stddef.h>

#include "arm_math.h"

#include "audio.h"
#include "backbone_tflite_utils.h"

static ns_timer_config_t timer_cfg = {
    .api = &ns_timer_V1_0_0,
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};


#if MEL_MAIN==1

#include "mel_spec.h"
#include "mel_cnn.h"

/*****************************************************************/

// Buffer and addresses used for calculating mel spectogram 

#define NUM_FRAMES (((AUDIO_LEN - FFT_SIZE) / FRAME_SHIFT) + 1)
#define NB_RUNS 200

float fft_buff1[FFT_SIZE];
float fft_buff2[FFT_SIZE];

float * frame_f32=fft_buff1;
float * fft_input=fft_buff2;
float * fft_output=fft_buff1;
float * mag_squared=fft_buff2;

float mel_spec[NB_MELS*NUM_FRAMES];
float* mel_column;


arm_rfft_fast_instance_f32 fft_inst;




int main(void){   
    
    // This code is used to Benchmark inference pipeline on Apollo5 based on Mel spectograms
    
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};

    
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\n");
    NS_TRY(ns_power_config(&ns_audio_default), "Power Init Failed.\n");
    ns_itm_printf_enable();
    ns_interrupt_master_enable();
    ns_lp_printf("Core init successful.\n");
    NS_TRY(ns_timer_init(&timer_cfg), "Timer init failed.\n");
    NS_TRY(ns_set_performance_mode(NS_MAXIMUM_PERF), "Set CPU Perf mode failed.");

    ns_lp_printf("Benchmark start\n");

    // Tested for the following settings
    ns_lp_printf("AUDIO_LEN: %lu\n", AUDIO_LEN);
    ns_lp_printf("NUM_FRAMES: %lu\n", NUM_FRAMES);


    // Time profiling
    uint32_t to_float_us = 0, fft_us = 0, mel_us = 0, melspec_us = 0;

    u_int32_t model_init_us = 0, inference_us=0;

    uint32_t t0, t1; // for finegrained profiling

    // Calculating Mel spectograms
    arm_rfft_fast_init_f32(&fft_inst, FFT_SIZE);

    t0 = ns_us_ticker_read(&timer_cfg);
    model_init(mel_CNN_tflite);
    t1 = ns_us_ticker_read(&timer_cfg);
    model_init_us=t1-t0;


    // Fill input properly
    int8_t * mel_input = (int8_t *) model_input->data.int8;
    float input_scale = model_input->params.scale;
	int input_zero_point = model_input->params.zero_point;
    

    if(1)
    {   
        uint32_t tl0 = ns_us_ticker_read(&timer_cfg);

        for(int run=0; run<NB_RUNS; run++){
            for (int s = 0; s < NUM_FRAMES; s++){
                t0 = ns_us_ticker_read(&timer_cfg);
                arm_q15_to_float(audio_int16 + (s * FRAME_SHIFT), frame_f32, FFT_SIZE);
                t1 = ns_us_ticker_read(&timer_cfg);
                to_float_us += (t1 - t0);

                // FFT ONLY
                t0 = ns_us_ticker_read(&timer_cfg);

                arm_mult_f32(frame_f32, hann, fft_input, FFT_SIZE);
                arm_rfft_fast_f32(&fft_inst, fft_input, fft_output, 0);

                // magnitude (NOT in FFT timing)
                mag_squared[0] = fft_output[0] * fft_output[0];
                arm_cmplx_mag_squared_f32(fft_output + 2,
                                        mag_squared + 1,
                                        (FFT_SIZE / 2) - 2);
                mag_squared[FFT_SIZE / 2] = fft_output[1] * fft_output[1];

                t1 = ns_us_ticker_read(&timer_cfg);
                fft_us += (t1 - t0);

                // MEL
                t0 = ns_us_ticker_read(&timer_cfg);

                mel_column = mel_spec+(s*NB_MELS);
                size_t mel_idx=0;
                for(size_t i=0; i<NB_MELS; i++){
                    size_t idx_win_start=mel_filter_pos[i];
                    size_t mel_size=mel_filter_len[i];
                    arm_dot_prod_f32(mag_squared+idx_win_start, mel_filter_coefs+mel_idx, mel_size, mel_column+i);
                    mel_idx+=mel_size;
                }

                arm_clip_f32(mel_column, mel_column, 1e-10f, FLT_MAX, NB_MELS);
                arm_vlog_f32(mel_column, mel_column, NB_MELS);

                for(size_t i=0; i<NB_MELS; i++){
                    int value = (int)(round((mel_column[i] / input_scale) + input_zero_point));
                    
                    if (value < -128)
                        value = -128;
                    else if (value > 127)
                        value = 127;
                    *(mel_input+(i+NB_MELS*s)) = (int8_t)(value);
                }

                t1 = ns_us_ticker_read(&timer_cfg);
                mel_us += (t1 - t0);
            }

            

             
        }
        uint32_t tl1 = ns_us_ticker_read(&timer_cfg);
        melspec_us = tl1-tl0;
    }


    ns_lp_printf("Mel spectogram - to float: %lu us\n", to_float_us/NB_RUNS);
    ns_lp_printf("Mel spectogram - Hann+FFT+CmplxMag: %lu us\n", fft_us/NB_RUNS);
    ns_lp_printf("Mel spectogram - Filterbanks + log: %lu us\n", mel_us/NB_RUNS);
    ns_lp_printf("Mel spectogram - total time: %lu us\n", melspec_us/NB_RUNS);

    // Warm-up
    interpreter->Invoke();

    // Timed run
    t0 = ns_us_ticker_read(&timer_cfg);

    if(1)
    {
        interpreter->Invoke();
    }

    t1 = ns_us_ticker_read(&timer_cfg);

    inference_us = t1 - t0;

    ns_lp_printf("Model setup time: %lu us\n", model_init_us);
    ns_lp_printf("Inference time: %lu us\n", inference_us);

    return 0;
}
#endif
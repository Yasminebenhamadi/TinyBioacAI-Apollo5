#include "ns_core.h"
#include "ns_timer.h"
#include "ns_debug_log.h"
#include "ns_energy_monitor.h"
#include "ns_peripherals_power.h"


#include "audio.h"
#include <arm_mve.h>
#include <stdint.h>
#include <stddef.h>
#include "arm_math.h"

#include "tflite_utils.h"
#include "gabor_layer.h"


static ns_timer_config_t timer_cfg = {
    .api = &ns_timer_V1_0_0,
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};

#if MEL_MAIN==0

int8_t *audio_low, * audio_high;

#define OUT_LEN 480 * 32

int8_t output_high[OUT_LEN];
int8_t output_low[OUT_LEN];
int16_t output_recon[OUT_LEN];
int16_t output_test[OUT_LEN];


void get_LSB(
    const int16_t * __restrict pSrc,
    int8_t * __restrict pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt = blockSize >> 4;


    u_int16_t * pSrcVec = (uint16_t *) pSrc;

    while (blkCnt > 0U)
    {
        uint16x8_t tmp;

        /* First 8 samples */
        tmp = vld1q((const uint16_t *)pSrcVec);
        pSrcVec += 8;

        tmp = vshrq_n_u16(tmp, 1);
        tmp = vandq_u16(tmp, vdupq_n_u16(0x7F));

        uint8x16_t bytes = vreinterpretq_u8_u16(tmp);

        for(int i =0; i<16; i+=2){
            *pDst++ = vgetq_lane(bytes, i);
        }
        /* Second 8 samples */
        tmp = vld1q((const uint16_t *)pSrcVec);
        pSrcVec += 8;

        tmp = vshrq_n_u16(tmp, 1);
        tmp = vandq_u16(tmp, vdupq_n_u16(0x7F));

        bytes = vreinterpretq_u8_u16(tmp);
        for(int i =0; i<16; i+=2){
            *pDst++ = vgetq_lane(bytes, i);
        }

        blkCnt--;
    }

    blkCnt = blockSize & 0xF;

    while (blkCnt > 0U)
    {
        *pDst++ = (int8_t)(((uint16_t)*pSrcVec++ >> 1) & 0x7F);
        blkCnt--;
    }
}


void reconstruct_int16(
    const int8_t *x_high,
    const int8_t *x_low,
    int16_t *x_recon,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        x_recon[i] =
            ((int16_t)x_high[i] << 8) +
            (int8_t)(x_low[i] << 1);
    }
}

void reconstruct_int16_mve(
  const q7_t * x_low,
  const q7_t * x_high,
  q15_t * pDst,
  uint32_t blockSize)
{
    // @TODO try to fix, it does not give correct results
    uint32_t  blkCnt;           /* loop counters */
    q15x8_t vecLow, vecHigh;
    
    const q7_t *pLow; 
    const q7_t *pHigh;


    pLow = (q7_t *) x_low;
    pHigh = (q7_t *) x_high;

    blkCnt = blockSize >> 3; // Process in blocks of 8
    while (blkCnt > 0U)
    {
        vecLow = vldrbq_s16(pLow);   
        pLow += 8; 
        vecHigh = vldrbq_s16(pHigh);  
        pHigh += 8;
        
        

        vecLow = vshlq_n_s16(vecLow, 1);
        vecHigh = vshlq_n_s16(vecHigh, 8);

        
        vstrhq(pDst, vaddq(vecLow, vecHigh));   
        pDst += 8;
        blkCnt--;
    }

  blkCnt = blockSize & 7;
  while (blkCnt > 0U)
  {
    q15_t v_low = *pLow++;
    q15_t v_high = *pHigh++;
    *pDst++ = ((q15_t)v_high << 8) +
            (q15_t)(v_low << 1);
    blkCnt--;
  }

}


int main(void)
{

    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};

    
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\n");
    NS_TRY(ns_power_config(&ns_audio_default), "Power Init Failed.\n");
    ns_itm_printf_enable();
    ns_interrupt_master_enable();
    ns_lp_printf("Core init successful.\n");
    NS_TRY(ns_timer_init(&timer_cfg), "Timer init failed.\n");
    NS_TRY(ns_set_performance_mode(NS_MAXIMUM_PERF), "Set CPU Perf mode failed.");

    ns_lp_printf("Benchmark start\n");

    uint32_t total0, total1;
    uint32_t t0, t1; // for finegrained profiling

    u_int32_t gabor_init_us = 0, gabor_us=0, get_high=0, get_low=0, re_ticks=0;

    for (int i = 0; i < OUT_LEN; i++)
        output_high[i] = (rand() % 255) - 128;
    
    for (int i = 0; i < OUT_LEN; i++)
        output_low[i] = (rand() % 255) - 128;



    t0 = ns_us_ticker_read(&timer_cfg);
    model_init(gabor_feature_2D_int8_model_tflite);
    t1 = ns_us_ticker_read(&timer_cfg);
    gabor_init_us=t1-t0;

    while(1){
        for(int run=0; run<20; run++){
            audio_high=model_input->data.int8; 

            t0 = ns_us_ticker_read(&timer_cfg);
            total0 = ns_us_ticker_read(&timer_cfg);
            arm_q15_to_q7(audio_int16, audio_high, model_input->bytes / sizeof(int8_t));
            t1 = ns_us_ticker_read(&timer_cfg);
            get_high=t1-t0;

            // Warm-up
            interpreter->Invoke();

            // Timed run
            t0 = ns_us_ticker_read(&timer_cfg);
            interpreter->Invoke();
            t1 = ns_us_ticker_read(&timer_cfg);
            gabor_us = t1 - t0;


            audio_low=model_input->data.int8;

            t0 = ns_us_ticker_read(&timer_cfg);
            get_LSB(audio_int16, audio_low, model_input->bytes / sizeof(int8_t));
            t1 = ns_us_ticker_read(&timer_cfg);
            get_low=t1-t0;

            t0 = ns_us_ticker_read(&timer_cfg);
            reconstruct_int16(output_high,output_low,output_recon,OUT_LEN);
            t1 = ns_us_ticker_read(&timer_cfg);
            re_ticks = t1 - t0;

            total1 = ns_us_ticker_read(&timer_cfg);
        }

        ns_delay_us(100000);
        ns_lp_printf("Returning");

    }

    // Results
    ns_lp_printf("Model setup: %lu us\n", gabor_init_us);

    ns_lp_printf("Getting MSB: %lu us\n", get_high);
    ns_lp_printf("Inference on MSB: %lu us\n", gabor_us);

    ns_lp_printf("Getting LSB: %lu us\n", get_low);
    ns_lp_printf("Inference on LSB: %lu us\n", gabor_us);

    ns_lp_printf("Combing low and high: %lu us\n", re_ticks);

    ns_lp_printf("Total Gabor: %lu us\n", total1-total0);
    
    return 0;
}
#endif
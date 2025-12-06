/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. (now owned by
 * Analog Devices, Inc.),
 * Copyright (C) 2023-2024 Analog Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

 #include <stdlib.h>
 #include <stdint.h>
 #include <string.h>
 #include <stdio.h>
 #include "mxc_device.h"
 #include "mxc_sys.h"
 #include "fcr_regs.h"
 #include "icc.h"
 #include "led.h"
 #include "tmr.h"
 #include "dma.h"
 #include "pb.h"
 #include "cnn.h"
 #include "weights.h"
 #include "mxc_delay.h"
 #include "camera.h"
 #include "gpio.h"
 #include "gcr_regs.h"
 
 #define IMAGE_SIZE_X (64)
 #define IMAGE_SIZE_Y (64)
 
 #define CAMERA_FREQ (10 * 1000 * 1000)
 
 // LED Pin Definitions
 #define LED_RED_LOWER_PORT MXC_GPIO0
 #define LED_RED_LOWER_PIN  MXC_GPIO_PIN_9
 
 #define LED_RED_UPPER_PORT MXC_GPIO0
 #define LED_RED_UPPER_PIN  MXC_GPIO_PIN_8
 
 #define LED_GREEN_PORT    MXC_GPIO0
 #define LED_GREEN_PIN     MXC_GPIO_PIN_11
 
 #define LED_YELLOW_PORT   MXC_GPIO0
 #define LED_YELLOW_PIN    MXC_GPIO_PIN_6
 
 
 // Button Configuration
 #define BUTTON_PORT         MXC_GPIO1
 #define BUTTON_PIN          MXC_GPIO_PIN_1
 #define DEBOUNCE_TIME_MS    150
 #define DEBOUNCE_TICKS      ((SystemCoreClock / 32 / 1000) * DEBOUNCE_TIME_MS)
 
 // Detection Parameters
 #define FRAMES_PER_POSITION 110
 #define CAPTURE_DURATION_MS 5000
 #define FRAME_DELAY_MS      (CAPTURE_DURATION_MS / FRAMES_PER_POSITION)
 #define THRESHOLD_PERCENT   50.0
 #define INTER_POSITION_DELAY_MS 2000
 
 // GPIO Configurations
 const mxc_gpio_cfg_t led_red_lower = {LED_RED_LOWER_PORT, LED_RED_LOWER_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
 const mxc_gpio_cfg_t led_red_upper = {LED_RED_UPPER_PORT, LED_RED_UPPER_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
 const mxc_gpio_cfg_t led_green = {LED_GREEN_PORT, LED_GREEN_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
 const mxc_gpio_cfg_t led_yellow = {LED_YELLOW_PORT, LED_YELLOW_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
 const mxc_gpio_cfg_t button_cfg = {BUTTON_PORT, BUTTON_PIN, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_PULL_UP};
 
 // Class definitions (update based on your CNN model output)
 typedef enum {
     CLASS_PROPER_FMHN = 0,      // ProperFM + ProperHN
     CLASS_IMPROPER_FMHN = 1,    // ImproperFM + ImproperHN
     CLASS_PROPER_FM_IMPROPER_HN = 2,  // ProperFM + ImproperHN
     CLASS_PROPER_HN_IMPROPER_FM = 3,  // ProperHN + ImproperFM
     NUM_CLASSES = 4
 } ClassType;
 
 char *class_names[NUM_CLASSES] = { "improper_fm_hn", "proper_fm_hn", "proper_fm_improper_hn", "proper_hn_improper_fm" };
 
 char *position_names[3] = {"FRONT", "LEFT", "RIGHT"};
 
 // Global Variables
 static int32_t ml_data[CNN_NUM_OUTPUTS];
 static q15_t ml_softmax[CNN_NUM_OUTPUTS];
 volatile uint32_t cnn_time;
 uint8_t data565[IMAGE_SIZE_X * 2];
 static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y];
 
 volatile int button_pressed = 0;
 volatile uint32_t last_button_press_time = 0;
 
 int class_counts[NUM_CLASSES];
 
 /* **************************************************************************** */
 void button_handler(void *cbdata)
 {
     uint32_t current_ticks = MXC_TMR_GetCount(MXC_TMR0);
     if ((current_ticks - last_button_press_time) > DEBOUNCE_TICKS) {
         button_pressed = 1;
         last_button_press_time = current_ticks;
     }
 }
 
 void setup_button(void)
 {
     mxc_tmr_cfg_t tmr_cfg;
     tmr_cfg.pres = TMR_PRES_32;
     tmr_cfg.mode = TMR_MODE_CONTINUOUS;
     tmr_cfg.cmp_cnt = 0;
     tmr_cfg.pol = 0;
 
     MXC_TMR_Init(MXC_TMR0, &tmr_cfg, false);
     MXC_TMR_Start(MXC_TMR0);
 
     MXC_GPIO_Config(&button_cfg);
     MXC_GPIO_RegisterCallback(&button_cfg, button_handler, NULL);
     MXC_GPIO_IntConfig(&button_cfg, MXC_GPIO_INT_FALLING);
     MXC_GPIO_EnableInt(button_cfg.port, button_cfg.mask);
     NVIC_EnableIRQ(MXC_GPIO_GET_IRQ(MXC_GPIO_GET_IDX(button_cfg.port)));
 }
 
 /* **************************************************************************** */
 void setup_leds(void)
 {
     MXC_GPIO_Config(&led_yellow);
     MXC_GPIO_Config(&led_red_upper);
     MXC_GPIO_Config(&led_red_lower);
     MXC_GPIO_Config(&led_green);
     
     // Turn off all LEDs initially
     MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
     MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
     MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
     MXC_GPIO_OutClr(led_green.port, led_green.mask);
 }
 
 void all_leds_off(void)
 {
     MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
     MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
     MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
     MXC_GPIO_OutClr(led_green.port, led_green.mask);
 }
 
 void display_classification_result(int class_idx)
 {
     all_leds_off();
     
     switch(class_idx) {
         case CLASS_PROPER_FMHN:
             // Both green LEDs ON
             MXC_GPIO_OutSet(led_green.port, led_green.mask);
             printf("Result: ProperFM + ProperHN (Both Green LEDs ON)\n");
             break;
             
         case CLASS_IMPROPER_FMHN:
             // Both red LEDs ON
             MXC_GPIO_OutSet(led_red_upper.port, led_red_upper.mask);
             MXC_GPIO_OutSet(led_red_lower.port, led_red_lower.mask);
             printf("Result: ImproperFM + ImproperHN (Both Red LEDs ON)\n");
             break;
             
         case CLASS_PROPER_FM_IMPROPER_HN:
             // Upper red LED ON, Lower red OFF
             MXC_GPIO_OutSet(led_red_upper.port, led_red_upper.mask);
             MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
             printf("Result: ProperFM + ImproperHN (Upper Red LED ON)\n");
             break;
             
         case CLASS_PROPER_HN_IMPROPER_FM:
             // Lower red LED ON, Upper red OFF
             MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
             MXC_GPIO_OutSet(led_red_lower.port, led_red_lower.mask);
             printf("Result: ProperHN + ImproperFM (Lower Red LED ON)\n");
             break;
             
         default:
             all_leds_off();
             printf("Result: No class above threshold (All LEDs OFF)\n");
             break;
     }
 }
 
 /* **************************************************************************** */
 int get_cnn_classification(void)
 {
     int32_t max = ml_data[0];
     int32_t max_index = 0;
 
     for (int i = 1; i < CNN_NUM_OUTPUTS; i++) {
         if (ml_data[i] > max) {
             max = ml_data[i];
             max_index = i;
         }
     }
     
     // Map CNN output to our 4 classes (adjust based on your model)
     if (max_index < NUM_CLASSES) {
         return max_index;
     }
     return -1; // Unknown class
 }
 
 /* **************************************************************************** */
 void cnn_load_input(void)
 {
     const uint32_t *in0 = input_0;
 
     for (int i = 0; i < 4096; i++) {
         while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
         *((volatile uint32_t *)0x50000008) = *in0++;
     }
 }
 
 /* **************************************************************************** */
 void capture_process_camera(void)
 {
     uint8_t *raw;
     uint32_t imgLen;
     uint32_t w, h;
     int cnt = 0;
     uint8_t r, g, b;
     uint16_t rgb;
     int j = 0;
     uint8_t *data = NULL;
     stream_stat_t *stat;
 
     camera_start_capture_image();
     camera_get_image(&raw, &imgLen, &w, &h);
 
     for (int row = 0; row < h; row++) {
         while ((data = get_camera_stream_buffer()) == NULL) {
             if (camera_is_image_rcv()) {
                 break;
             }
         }
         j = 0;
         for (int k = 0; k < 4 * w; k += 4) {
             r = data[k];
             g = data[k + 1];
             b = data[k + 2];
             input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;
             rgb = ((r & 0b11111000) << 8) | ((g & 0b11111100) << 3) | (b >> 3);
             data565[j] = (rgb >> 8) & 0xFF;
             data565[j + 1] = rgb & 0xFF;
             j += 2;
         }
         release_camera_stream_buffer();
     }
 
     stat = get_camera_stream_statistic();
     if (stat->overflow_count > 0) {
         printf("OVERFLOW = %d\n", stat->overflow_count);
     }
 }
 
 /* **************************************************************************** */
 void process_position(int position_idx)
 {
     printf("\n=== Processing %s position ===\n", position_names[position_idx]);
     
     // Reset class counts
     for (int i = 0; i < NUM_CLASSES; i++) {
         class_counts[i] = 0;
     }
     
     // Phase (a): Yellow LED ON - Capture Phase (5 seconds, 110 frames)
     printf("Yellow LED ON - Capturing frames...\n");
     MXC_GPIO_OutSet(led_yellow.port, led_yellow.mask);
     
     for (int frame = 0; frame < FRAMES_PER_POSITION; frame++) {
         // Capture and process camera image
         capture_process_camera();
         
         // Run CNN inference
         cnn_time = 0;
         cnn_start();
         cnn_load_input();
         
         while (cnn_time == 0) {
             __WFI();
         }
         
         cnn_unload((uint32_t *)ml_data);
         softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
         
         // Get classification result
         int class_idx = get_cnn_classification();
         if (class_idx >= 0 && class_idx < NUM_CLASSES) {
             class_counts[class_idx]++;
         }
         
         if ((frame + 1) % 20 == 0) {
             printf("Progress: %d/%d frames\n", frame + 1, FRAMES_PER_POSITION);
         }
         
         MXC_Delay(MXC_DELAY_MSEC(FRAME_DELAY_MS));
     }
     
     // Phase (b): Yellow LED OFF - Classification Phase
     printf("Yellow LED OFF - Computing final classification...\n");
     MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
     
     // Calculate detection success rates
     printf("\nDetection Results:\n");
     double max_rate = 0.0;
     int best_class = -1;
     
     for (int i = 0; i < NUM_CLASSES; i++) {
         double rate = (class_counts[i] * 100.0) / FRAMES_PER_POSITION;
         printf("  %s: %d detections (%.2f%%)\n", class_names[i], class_counts[i], rate);
         
         if (rate > (double)THRESHOLD_PERCENT && rate > max_rate) {
             max_rate = rate;
             best_class = i;
         }
     }
     
     // Phase (c): Display result via LEDs
     printf("\nFinal Classification: ");
     if (best_class >= 0) {
         printf("%s (%.2f%%)\n", class_names[best_class], max_rate);
         display_classification_result(best_class);
     } else {
         printf("No class above %.0f%% threshold\n", (double)THRESHOLD_PERCENT);
         display_classification_result(-1);
     }
     
     // Phase (d): Delay before next position
     printf("Waiting %d seconds before next position...\n", INTER_POSITION_DELAY_MS / 1000);
     MXC_Delay(MXC_DELAY_MSEC(INTER_POSITION_DELAY_MS));
 }
 
 /* **************************************************************************** */
 int main(void)
 {
     int dma_channel;
 
     MXC_Delay(200000);
     Camera_Power(POWER_ON);
     printf("\n\n=== Facemask/Hairnet Detection System ===\n");;
 
     MXC_ICC_Enable(MXC_ICC0);
     MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
     SystemCoreClockUpdate();
 
     // Initialize CNN
     cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
     cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
     cnn_init();
     cnn_load_weights();
     cnn_load_bias();
     cnn_configure();
 
     // Initialize DMA
     MXC_DMA_Init();
     dma_channel = MXC_DMA_AcquireChannel();
 
    printf("Init Camera.\n");
    camera_init(CAMERA_FREQ);
    camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, STREAMING_DMA, dma_channel);
    camera_set_hmirror(0);
    camera_set_vflip(0);
    camera_write_reg(0x11, 0x80);

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);
 
     // Setup hardware
     setup_leds();
     setup_button();
 
     printf("\nSystem ready!\n");
     printf("Press button to start detection sequence...\n\n");
 
     while (1) {
         // Wait for button press
         if (button_pressed) {
             button_pressed = 0;
             
             printf("\n========================================\n");
             printf("Starting detection sequence!\n");
             printf("========================================\n");
             
             all_leds_off();
             
             // Process three positions in sequence
             process_position(0); // FRONT
             process_position(1); // LEFT
             process_position(2); // RIGHT
             
             printf("\n========================================\n");
             printf("Detection sequence complete!\n");
             printf("Press button to start again...\n");
             printf("========================================\n\n");
             
             all_leds_off();
         }
     }
 
     return 0;
 }
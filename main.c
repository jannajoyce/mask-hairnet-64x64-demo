/******************************************************************************
 * main.c -- Restart-safe, interrupt-driven button handling for MAX78000
 * - Starts detection sequence when button pressed (idle)
 * - If button pressed during detection, requests reset and restarts
 * - Each position: capture exactly FRAMES_PER_POSITION frames over CAPTURE_DURATION_MS
 * - Yellow LED on during capture (CAPTURE_DURATION_MS)
 * - Result LED on for 1s, then off; 2s pause between positions
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
#include "wdt.h"

#define IMAGE_SIZE_X (64)
#define IMAGE_SIZE_Y (64)

#define CAMERA_FREQ (10 * 1000 * 1000)

// LED Pin Definitions
#define LED_RED_UPPER_PORT MXC_GPIO0
#define LED_RED_UPPER_PIN  MXC_GPIO_PIN_9

#define LED_RED_LOWER_PORT MXC_GPIO0
#define LED_RED_LOWER_PIN  MXC_GPIO_PIN_8

#define LED_GREEN_PORT    MXC_GPIO0
#define LED_GREEN_PIN     MXC_GPIO_PIN_11

#define LED_YELLOW_PORT   MXC_GPIO0
#define LED_YELLOW_PIN    MXC_GPIO_PIN_6

// Button Configuration - External button on GPIO1 Pin 1
#define BUTTON_PORT         MXC_GPIO1
#define BUTTON_PIN          MXC_GPIO_PIN_1
#define DEBOUNCE_TIME_MS    150
#define DEBOUNCE_TICKS      ((SystemCoreClock / 32 / 1000) * DEBOUNCE_TIME_MS)

// Detection Parameters
#define FRAMES_PER_POSITION 110
#define CAPTURE_DURATION_MS 5000
//#define FRAME_DELAY_MS      (CAPTURE_DURATION_MS / FRAMES_PER_POSITION)
#define THRESHOLD_PERCENT   50.0
#define INTER_POSITION_DELAY_MS 2000

// GPIO Configurations
const mxc_gpio_cfg_t led_red_lower = {LED_RED_LOWER_PORT, LED_RED_LOWER_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_red_upper = {LED_RED_UPPER_PORT, LED_RED_UPPER_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_green = {LED_GREEN_PORT, LED_GREEN_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_yellow = {LED_YELLOW_PORT, LED_YELLOW_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t button_cfg = {BUTTON_PORT, BUTTON_PIN, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_PULL_UP, MXC_GPIO_VSSEL_VDDIOH};

// Class definitions
typedef enum {
    CLASS_IMPROPER_FMHN = 0,
    CLASS_PROPER_FMHN = 1,
    CLASS_PROPER_FM_IMPROPER_HN = 2,
    CLASS_PROPER_HN_IMPROPER_FM = 3,
    NUM_CLASSES = 4
} ClassType;

char *class_names[NUM_CLASSES] = { "improper_fm_hn", "proper_fm_hn", "proper_fm_improper_hn", "proper_hn_improper_fm" };
char *position_names[3] = {"FRONT", "LEFT", "RIGHT"};

// Global Variables (CNN/capture)
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];
volatile uint32_t cnn_time;
uint8_t data565[IMAGE_SIZE_X * 2];
static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y];

volatile int button_pressed = 0;           // Set by ISR to request start (when idle)
volatile int reset_requested = 0;          // Set by ISR to request immediate abort & restart (when detecting)
volatile uint32_t last_button_press_time = 0;
volatile int detecting = 0;                // 1 while detection sequence running
int class_counts[NUM_CLASSES];

/* **************************************************************************** */
/* Button ISR - debounced.                                                       */
/* Behavior:
 *  - If idle (detecting == 0): set button_pressed to request start
 *  - If detecting: set reset_requested so the main loop aborts current run and restarts
 */
/* **************************************************************************** */
void button_handler(void *cbdata)
{
    // Clear IRQ flag first
    MXC_GPIO_ClearFlags(button_cfg.port, button_cfg.mask);

    uint32_t current_ticks = MXC_TMR_GetCount(MXC_TMR0);

    // Debounce: require at least DEBOUNCE_TICKS between accepted presses
    if ((current_ticks - last_button_press_time) > DEBOUNCE_TICKS) {
        last_button_press_time = current_ticks;

        if (detecting) {
            // Abort current detection, request restart
            reset_requested = 1;
            printf("[ISR] Button pressed -> request RESET\n");
        } else {
            // Request start when idle
            button_pressed = 1;
            printf("[ISR] Button pressed -> request START\n");
        }
    }
}

/* **************************************************************************** */
/* Setup button: timer for debounce + GPIO interrupt                             */
/* **************************************************************************** */
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

    printf("Button setup complete. IRQ enabled for GPIO%d Pin %d\n",
           MXC_GPIO_GET_IDX(button_cfg.port), __builtin_ctz(button_cfg.mask));
}

/* **************************************************************************** */
/* LED helpers                                                                  */
/* **************************************************************************** */
void setup_leds(void)
{
    MXC_GPIO_Config(&led_red_lower);
    MXC_GPIO_Config(&led_red_upper);
    MXC_GPIO_Config(&led_green);
    MXC_GPIO_Config(&led_yellow);

    /* Turn off all LEDs initially */
    MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
    MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
    MXC_GPIO_OutClr(led_green.port, led_green.mask);
    MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
}

void all_leds_off(void)
{
    MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
    MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
    MXC_GPIO_OutClr(led_green.port, led_green.mask);
    MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
}

void display_classification_result(int class_idx)
{
    all_leds_off();

    switch(class_idx) {
        case CLASS_PROPER_FMHN:
            MXC_GPIO_OutSet(led_green.port, led_green.mask);
            break;
        case CLASS_IMPROPER_FMHN:
            MXC_GPIO_OutSet(led_red_upper.port, led_red_upper.mask);
            MXC_GPIO_OutSet(led_red_lower.port, led_red_lower.mask);
            break;
        case CLASS_PROPER_FM_IMPROPER_HN:
            MXC_GPIO_OutSet(led_red_upper.port, led_red_upper.mask);
            MXC_GPIO_OutClr(led_red_lower.port, led_red_lower.mask);
            break;
        case CLASS_PROPER_HN_IMPROPER_FM:
            MXC_GPIO_OutClr(led_red_upper.port, led_red_upper.mask);
            MXC_GPIO_OutSet(led_red_lower.port, led_red_lower.mask);
            break;
        default:
            all_leds_off();
            printf("Result: No class above threshold (All LEDs OFF)\n");
            break;
    }
}

/* **************************************************************************** */
/* CNN helper                                                                   */
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

    if (max_index < NUM_CLASSES) {
        return max_index;
    }
    return -1;
}

/* **************************************************************************** */
/* Load input to CNN (unchanged)                                                 */
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
/* Camera capture & process (unchanged)                                          */
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
/* process_position: captures FRAMES_PER_POSITION frames over CAPTURE_DURATION_MS,
 * runs CNN per frame, accumulates counts, decides class, shows LEDs for 1s,
 * then waits INTER_POSITION_DELAY_MS.
 * **************************************************************************** */
void process_position(int position_idx)
{
    printf("\n=== Processing %s position ===\n", position_names[position_idx]);

    for (int i = 0; i < NUM_CLASSES; i++) {
        class_counts[i] = 0;
    }

    /* Yellow LED on for the capture duration */
    MXC_GPIO_OutSet(led_yellow.port, led_yellow.mask);

    for (int frame = 0; frame < FRAMES_PER_POSITION; frame++) {
        // If reset requested, turn off yellow and exit immediately
        if (reset_requested) {
            MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);
            return;
        }

        capture_process_camera();

        /* Run CNN inference */
        cnn_time = 0;
        cnn_start();
        cnn_load_input();

        while (cnn_time == 0) {
            __WFI(); // wait for CNN interrupt (interrupts remain enabled)
        }

        cnn_unload((uint32_t *)ml_data);
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);

        int class_idx = get_cnn_classification();
        if (class_idx >= 0 && class_idx < NUM_CLASSES) {
            class_counts[class_idx]++;
            printf("%d: %s\n", frame + 1, class_names[class_idx]);
        }

        /* Delay to spread captures evenly over CAPTURE_DURATION_MS */
       // MXC_Delay(MXC_DELAY_MSEC(FRAME_DELAY_MS));
    }

    /* Capture done */
    MXC_GPIO_OutClr(led_yellow.port, led_yellow.mask);

    /* Evaluate results */
    printf("\nDetection Results:\n");
    double max_rate = 0.0;
    int best_class = -1;

    for (int i = 0; i < NUM_CLASSES; i++) {
        double rate = ((double)class_counts[i] / FRAMES_PER_POSITION) * 100.0;
        printf("  %s: %d detections (%.2f%%)\n", class_names[i], class_counts[i], rate);
        if (rate > (double)THRESHOLD_PERCENT && rate > max_rate) {
            max_rate = rate;
            best_class = i;
        }
    }

    printf("\nFinal Classification: ");
    if (best_class >= 0) {
        printf("%s (%.2f%%)\n", class_names[best_class], max_rate);
        display_classification_result(best_class);
    } else {
        printf("No class above %.0f%% threshold\n", (double)THRESHOLD_PERCENT);
        display_classification_result(-1);
    }

    /* Show result for 1 second (user requirement), then turn off */
    MXC_Delay(MXC_DELAY_MSEC(1000));
    all_leds_off();

    /* Delay before next position (2 seconds) */
    MXC_Delay(MXC_DELAY_MSEC(INTER_POSITION_DELAY_MS));
}

/* **************************************************************************** */
/* main - initialize hardware and run event loop                                */
/* **************************************************************************** */
int main(void)
{
    int dma_channel;

    /* Disable watchdog timer to prevent reset */
    MXC_WDT_Disable(MXC_WDT0);

    MXC_Delay(2000000);
    Camera_Power(POWER_ON);

    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Initialize CNN */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();

    /* Initialize DMA */
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    camera_init(CAMERA_FREQ);
    camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, STREAMING_DMA, dma_channel);
    camera_set_hmirror(0);
    camera_set_vflip(0);
    camera_write_reg(0x11, 0x80);

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    /* Setup hardware */
    setup_leds();
    setup_button();

    //printf("\nSystem ready!\n");
   // printf("Press button to start detection sequence...\n\n");

    /* Clear state */
    reset_requested = 0;
    detecting = 0;

    while (1) {
        /* Wait for button press - direct polling, no interrupts needed */
        while (MXC_GPIO_InGet(button_cfg.port, button_cfg.mask) != 0) {
            MXC_Delay(MXC_DELAY_MSEC(10));
        }

        /* Button pressed - debounce */
        MXC_Delay(MXC_DELAY_MSEC(DEBOUNCE_TIME_MS));

        /* Verify button is still pressed after debounce */
        if (MXC_GPIO_InGet(button_cfg.port, button_cfg.mask) != 0) {
            continue; // False trigger, go back to waiting
        }

        /* Begin detection sequence */
        reset_requested = 0;   // clear any stale reset
        detecting = 1;         // mark detection active

        /* Wait for physical button release before starting */
        while (MXC_GPIO_InGet(button_cfg.port, button_cfg.mask) == 0) {
            MXC_Delay(MXC_DELAY_MSEC(10));
        }

        printf("\n========================================\n");
        printf("Starting detection sequence!\n");
        printf("========================================\n");

        all_leds_off();

        /* Run three positions. If reset_requested becomes true during any position,
           we abort immediately and go back to the top so ISR can re-request start. */
        process_position(0); /* FRONT */
        if (reset_requested) {
            printf("\n!!!! Detection interrupted - Restarting... !!!!\n");
            all_leds_off();
            detecting = 0;
            MXC_Delay(MXC_DELAY_MSEC(500));
            // keep button_pressed as set by ISR if ISR set it (ISR will set it when idle)
            continue;
        }

        process_position(1); /* LEFT */
        if (reset_requested) {
            printf("\n!!!! Detection interrupted - Restarting... !!!!\n");
            all_leds_off();
            detecting = 0;
            MXC_Delay(MXC_DELAY_MSEC(500));
            continue;
        }

        process_position(2); /* RIGHT */
        if (reset_requested) {
            printf("\n!!!! Detection interrupted - Restarting... !!!!\n");
            all_leds_off();
            detecting = 0;
            MXC_Delay(MXC_DELAY_MSEC(500));
            continue;
        }

        printf("\n========================================\n");
        printf("Detection sequence complete!\n");
        printf("Press button to start again...\n");
        printf("========================================\n\n");

        all_leds_off();

        /* Clear state and mark as idle */
        reset_requested = 0;
        detecting = 0;
    }

    return 0;
}
/*****************************************************************/

#ifndef GABOR_TFLITE_UTILS_H
#define GABOR_TFLITE_UTILS_H

// Tensorflow Lite for Microcontroller headers
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

// Tensorflow Globals (somewhat boilerplate)
static tflite::ErrorReporter *gabor_error_reporter = nullptr;
static const tflite::Model *gabor_model = nullptr;
static tflite::MicroInterpreter *gabor_interpreter = nullptr;
static TfLiteTensor *gabor_model_input = nullptr;
static TfLiteTensor *gabor_model_output = nullptr;
static tflite::MicroProfiler *gabor_profiler = nullptr;

static constexpr int kGaborTensorArenaSize = 1024 * 150;
alignas(16) static uint8_t gabor_tensor_arena[kGaborTensorArenaSize];

static void gabor_model_init(const unsigned char *model_flatbuffer);

static void
gabor_model_init(const unsigned char *model_flatbuffer) {
    tflite::MicroErrorReporter micro_error_reporter;
    gabor_error_reporter = &micro_error_reporter;
    ns_lp_printf("Error reporter initialized.\n");
    tflite::InitializeTarget();
    ns_lp_printf("Target initialized.\n");
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    gabor_model = tflite::GetModel(model_flatbuffer);
    if (gabor_model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(gabor_error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             gabor_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ns_lp_printf("Model initialized.\n");
    // Build mutable resolver with minimum opset
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddMean();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(gabor_model, resolver, gabor_tensor_arena,
                                                       kGaborTensorArenaSize, nullptr, gabor_profiler);
    ns_lp_printf("Interpreter initialized.\n");
    gabor_interpreter = &static_interpreter;
    ns_lp_printf("Interpreter assigned.\n");
    ns_lp_printf("Arena address: 0x%x\n", gabor_tensor_arena);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = gabor_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(gabor_error_reporter, "AllocateTensors() failed\n");
        ns_lp_printf("Arena size computed: %d\n", gabor_interpreter->arena_used_bytes());
        ns_lp_printf("AllocateTensors() failed with status %d\n", allocate_status);
        return;
    }
    ns_lp_printf("AllocateTensors() successful.\n");
    ns_lp_printf("Arena size computed: %d\n", gabor_interpreter->arena_used_bytes());

    // Obtain pointers to the model's input and output tensors.
    gabor_model_input = gabor_interpreter->input(0);
    gabor_model_output = gabor_interpreter->output(0);
    ns_lp_printf("Input tensor: 0x%x\n", gabor_model_input);
    ns_lp_printf("Output tensor: 0x%x\n", gabor_model_output);
}

#endif

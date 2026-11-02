/*****************************************************************/

// Tensorflow Lite for Microcontroller headers
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

// Tensorflow Globals (somewhat boilerplate)
static tflite::ErrorReporter *error_reporter = nullptr;
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *model_input = nullptr;
static TfLiteTensor *model_output = nullptr;
static tflite::MicroProfiler *profiler = nullptr;

static constexpr int kTensorArenaSize = 1024 * 150;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static void model_init(const unsigned char * model_flatbuffer);



static void 
model_init(const unsigned char * model_flatbuffer) {

    tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    ns_lp_printf("Error reporter initialized.\n");
    tflite::InitializeTarget();
    ns_lp_printf("Target initialized.\n");
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(model_flatbuffer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ns_lp_printf("Model initialized.\n");
    // Build mutable resolver with minimum opset
    static tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddMean();
    resolver.AddPack();
    resolver.AddShape();
    resolver.AddStridedSlice();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
                                                       kTensorArenaSize, nullptr, profiler);
    ns_lp_printf("Interpreter initialized.\n");
    interpreter = &static_interpreter;
    ns_lp_printf("Interpreter assigned.\n");
    ns_lp_printf("Arena address: 0x%x\n", tensor_arena);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed\n");
        ns_lp_printf("Arena size computed: %d\n", interpreter->arena_used_bytes());
        ns_lp_printf("AllocateTensors() failed with status %d\n", allocate_status);
        return;
    }
    ns_lp_printf("AllocateTensors() successful.\n");
    ns_lp_printf("Arena size computed: %d\n", interpreter->arena_used_bytes());

    // Obtain pointers to the model's input and output tensors.
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
    ns_lp_printf("Input tensor: 0x%x\n", model_input);
    ns_lp_printf("Output tensor: 0x%x\n", model_output);
}
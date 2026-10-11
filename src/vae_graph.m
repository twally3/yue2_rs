// FP32 Oobleck decoder: a GPU-resident graph, with one host transfer per tile.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    size_t kind, input, residual, channels_in, channels_out;
    size_t kernel, stride, padding, dilation, frames;
    const float *weight, *bias, *alpha, *beta;
} Yue2VaeNode;

@interface Yue2VaeGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input, *output;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLBuffer> inputBuffer;
@property(nonatomic) size_t inputFrames, inputChannels, outputFrames, outputChannels;
@end
@implementation Yue2VaeGraph
@end

static MPSGraphTensor *constant(MPSGraph *g, const float *data, size_t count, NSArray *shape) {
    return [g constantWithData:[NSData dataWithBytes:data length:count*sizeof(float)]
                        shape:shape dataType:MPSDataTypeFloat32];
}

void *yue2_vae_graph_create(const Yue2VaeNode *nodes, size_t count, size_t frames, size_t channels) {
    @autoreleasepool {
        @try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || !count) return NULL;
            Yue2VaeGraph *program = [Yue2VaeGraph new];
            program.queue = [device newCommandQueue];
            program.graph = [MPSGraph new];
            MPSGraph *g = program.graph;
            program.inputFrames = frames;
            program.inputChannels = channels;
            program.input = [g placeholderWithShape:@[@1,@1,@(frames),@(channels)]
                                           dataType:MPSDataTypeFloat32 name:@"latent"];
            NSMutableArray<MPSGraphTensor *> *values = [NSMutableArray arrayWithObject:program.input];
            for (size_t i=0;i<count;++i) {
                const Yue2VaeNode *node = nodes+i;
                if (node->input > i || node->residual > i) return NULL;
                MPSGraphTensor *x = values[node->input], *y = nil;
                switch (node->kind) {
                    case 0: // Conv1d, NHWC data and checkpoint OIHW weights.
                    case 1: { // ConvTranspose1d is the corresponding data gradient.
                        BOOL transpose = node->kind == 1;
                        NSArray *shape = transpose
                            ? @[@(node->channels_in),@(node->channels_out),@1,@(node->kernel)]
                            : @[@(node->channels_out),@(node->channels_in),@1,@(node->kernel)];
                        MPSGraphTensor *w = constant(g,node->weight,
                            node->channels_in*node->channels_out*node->kernel,shape);
                        MPSGraphConvolution2DOpDescriptor *desc =
                            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:node->stride
                                strideInY:1 dilationRateInX:node->dilation dilationRateInY:1
                                groups:1 paddingLeft:node->padding paddingRight:node->padding
                                paddingTop:0 paddingBottom:0 paddingStyle:MPSGraphPaddingStyleExplicit
                                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
                        if (transpose) {
                            y = [g convolution2DDataGradientWithIncomingGradientTensor:x weightsTensor:w
                                outputShape:@[@1,@1,@(node->frames),@(node->channels_out)]
                                forwardConvolutionDescriptor:desc name:nil];
                        } else {
                            y = [g convolution2DWithSourceTensor:x weightsTensor:w descriptor:desc name:nil];
                        }
                        MPSGraphTensor *bias = constant(g,node->bias,node->channels_out,
                            @[@1,@1,@1,@(node->channels_out)]);
                        y = [g additionWithPrimaryTensor:y secondaryTensor:bias name:nil];
                        break;
                    }
                    case 2: { // SnakeBeta, with checkpoint alpha/beta in exponentiated FP32.
                        NSArray *shape = @[@1,@1,@1,@(node->channels_in)];
                        MPSGraphTensor *alpha = constant(g,node->alpha,node->channels_in,shape);
                        MPSGraphTensor *beta = constant(g,node->beta,node->channels_in,shape);
                        MPSGraphTensor *sine = [g sinWithTensor:
                            [g multiplicationWithPrimaryTensor:x secondaryTensor:alpha name:nil] name:nil];
                        MPSGraphTensor *square = [g multiplicationWithPrimaryTensor:sine secondaryTensor:sine name:nil];
                        MPSGraphTensor *epsilon = [g constantWithScalar:1e-9 dataType:MPSDataTypeFloat32];
                        MPSGraphTensor *denominator = [g additionWithPrimaryTensor:beta secondaryTensor:epsilon name:nil];
                        MPSGraphTensor *one = [g constantWithScalar:1 dataType:MPSDataTypeFloat32];
                        MPSGraphTensor *inverse = [g divisionWithPrimaryTensor:one secondaryTensor:denominator name:nil];
                        y = [g additionWithPrimaryTensor:x secondaryTensor:
                            [g multiplicationWithPrimaryTensor:inverse secondaryTensor:square name:nil] name:nil];
                        break;
                    }
                    case 3: { // ELU, alpha=1.
                        MPSGraphTensor *zero = [g constantWithScalar:0 dataType:MPSDataTypeFloat32];
                        MPSGraphTensor *one = [g constantWithScalar:1 dataType:MPSDataTypeFloat32];
                        MPSGraphTensor *positive = [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];
                        MPSGraphTensor *negative = [g subtractionWithPrimaryTensor:
                            [g exponentWithTensor:x name:nil] secondaryTensor:one name:nil];
                        y = [g selectWithPredicateTensor:positive truePredicateTensor:x falsePredicateTensor:negative name:nil];
                        break;
                    }
                    case 4: y = [g tanhWithTensor:x name:nil]; break;
                    case 5: y = [g additionWithPrimaryTensor:x secondaryTensor:values[node->residual] name:nil]; break;
                    default: return NULL;
                }
                if (!y) return NULL;
                [values addObject:y];
            }
            program.output = values.lastObject;
            program.outputFrames = nodes[count-1].frames;
            program.outputChannels = nodes[count-1].channels_out;
            MTLResourceOptions storage = device.hasUnifiedMemory ? MTLResourceStorageModeShared : MTLResourceStorageModeManaged;
            program.inputBuffer = [device newBufferWithLength:frames*channels*sizeof(float) options:storage];
            if (!program.queue || !program.inputBuffer) return NULL;
            return (__bridge_retained void *)program;
        } @catch (NSException *error) {
            fprintf(stderr,"VAE graph construction failed: %s\n",error.reason.UTF8String);
            return NULL;
        }
    }
}

int yue2_vae_graph_run(void *handle, const float *input, float *output) {
    @autoreleasepool {
        @try {
            Yue2VaeGraph *program = (__bridge Yue2VaeGraph *)handle;
            size_t bytes = program.inputFrames*program.inputChannels*sizeof(float);
            memcpy(program.inputBuffer.contents,input,bytes);
            if (program.inputBuffer.storageMode == MTLStorageModeManaged)
                [program.inputBuffer didModifyRange:NSMakeRange(0,bytes)];
            MPSGraphTensorData *feed = [[MPSGraphTensorData alloc] initWithMTLBuffer:program.inputBuffer
                shape:program.input.shape dataType:MPSDataTypeFloat32];
            NSDictionary *results = [program.graph runWithMTLCommandQueue:program.queue
                feeds:@{program.input:feed} targetTensors:@[program.output] targetOperations:nil];
            MPSGraphTensorData *result = results[program.output];
            if (!result || ![result.shape isEqualToArray:@[@1,@1,@(program.outputFrames),@(program.outputChannels)]])
                return 0;
            [[result mpsndarray] readBytes:output strideBytes:nil];
            return 1;
        } @catch (NSException *error) {
            fprintf(stderr,"VAE graph execution failed: %s\n",error.reason.UTF8String);
            return 0;
        }
    }
}

void yue2_vae_graph_free(void *handle) {
    @autoreleasepool { id program = (__bridge_transfer id)handle; (void)program; }
}
